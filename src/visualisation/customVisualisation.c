
/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 
 * Based on templated visualisation.c, adapted and modified for gravitational N-body simulation
 * Author: Laurence James
 * Contact: ljames1@sheffield.ac.uk or laurence.james@gmail.com
 */

// includes, project
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <cutil_math.h>
#include <cudpp.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
    
#include <header.h>
#include <customVisualisation.h>
#include "globalsController.h"


// bo variables
GLuint sphereVerts;
GLuint sphereNormals;

//Simulation output buffers/textures

GLuint simulationVarsAgent_default_tbo;
GLuint simulationVarsAgent_default_displacementTex;

GLuint Particle_testingActive_tbo;
GLuint Particle_testingActive_displacementTex;

GLuint Particle_outputingData_tbo;
GLuint Particle_outputingData_displacementTex;

GLuint Particle_updatingPosition_tbo;
GLuint Particle_updatingPosition_displacementTex;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -VIEW_DISTANCE;

//Add x and y translation
float translate_x = 0;
float translate_y = 0;

// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_displacementMap;
GLuint vs_mapIndex;

//light position
GLfloat LIGHT_POSITION[] = {1.0f, 1.0f, 1.0f, 1.0f};

//timer
GLuint timer;
const int average = 50;
int frame_count;

#ifdef SIMULATION_DELAY 
//delay
int delay_count = 0;
#endif

//custom visualisation params
bool displayingDarkMatter=true;
bool displayingBaryonicMatter=true;
bool paused=true;

// prototypes

void initShader();

void setVertexBufferData();
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);


void initVisualisation()
{
	//Allow user to set window size for different aspect ratios
	
	printf("%s","\nWhen taking user input, your first valid string will be used. A completely invalid string will be prompted for again.\n");
	setWindowSize();
	
	//Handle defaults
	bool set=false;
	char input;
	char buf;
	printf("Use default visualisation settings? y/n \n");
	while(!set){
		input=getchar();
		while(input == '\n') input=getchar();
		switch(input){
		case 'y': case 'Y':
			set=true;
			break;		
		case 'n': case 'N':
			while ((buf = getchar()) != '\n' && buf != EOF);
			setVisualisationVars();
			set=true;
			break;
		case '\n':
			break;
		default:
			printf("Invalid input. Use default visualisation settings? y/n \n");
			break;
		}
		
		//flush any extra input before simulation defaults
		while ((buf = getchar()) != '\n' && buf != EOF);
	}
	
    //set the CUDA GL device: Will cause an error without this since CUDA 3.0
    setCudaDevice();

    // Create GL context
    int   argc   = 1;
    char *argv[] = {"GLUT application", NULL};
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow( "FLAME GPU Visualiser");

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }
	initShader();

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);
    
    // create VBO's
	createVBO( &sphereVerts, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof(float3));
	createVBO( &sphereNormals, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof (float3));
	setVertexBufferData();

	// create TBO
	
	createTBO( &Particle_testingActive_tbo, &Particle_testingActive_displacementTex, xmachine_memory_Particle_MAX * sizeof( float4));

	//set shader uniforms
	glUseProgram(shaderProgram);

    CUT_SAFE_CALL( cutCreateTimer( &timer));
}

void runVisualisation(){
    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
void incrementItNum();

#ifdef SIMULATION_DELAY
    delay_count++;
	if (delay_count == SIMULATION_DELAY){
		delay_count = 0;
		singleIteration();
		//
		incrementItNum();
	}
#else
	//only step simulation if visualisation isn't paused
	if(!paused){
		singleIteration();
		//itNum stored as simulation variable for optimisation approaches. Increment it here
		incrementItNum();
	}
#endif

	//kernals sizes
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
    dim3 threads;
    float3 centralise;

	//pointer
	float4 *dptr;

	//We only draw agents in testingActive state...
	//...bcause this is the only state it should be possible to be in!
	if (get_agent_Particle_testingActive_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
		CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, Particle_testingActive_tbo));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Particle_testingActive_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
		
		//cast FAR_CLIP to float for CUDA compatability
		output_Particle_agent_to_VBO<<< grid, threads>>>(get_device_Particle_testingActive_agents(), dptr, centralise, displayingDarkMatter, displayingBaryonicMatter, (float)(FAR_CLIP));
		CUT_CHECK_ERROR("Kernel execution failed");
		// unmap buffer object
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject(Particle_testingActive_tbo));
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }

    // default initialization
    
    //don't clear color - we want it black!
    //glClearColor( 255.0, 255.0, 255.0, 1.0);
    glEnable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (GLfloat)WINDOW_WIDTH / (GLfloat) WINDOW_HEIGHT, NEAR_CLIP, FAR_CLIP);

    checkGLError();

	//lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GLSL Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void initShader()
{
	const char* v = vertexShaderSource;
	const char* f = fragmentShaderSource;

	//vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &v, 0);
    glCompileShader(vertexShader);

	//fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &f, 0);
    glCompileShader(fragmentShader);

	//program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

	// check for errors
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[1024];
		int len;
		glGetShaderInfoLog(vertexShader, 1024, &len, data); 
		printf("%s", data);
	}
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[1024];
		int len;
		glGetShaderInfoLog(fragmentShader, 1024, &len, data); 
		printf("%s", data);
	}
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	vs_displacementMap = glGetUniformLocation(shaderProgram, "displacementMap");
	vs_mapIndex = glGetAttribLocation(shaderProgram, "mapIndex"); 
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, GLuint size)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create TBO
////////////////////////////////////////////////////////////////////////////////
void createTBO(GLuint* tbo, GLuint* tex, GLuint size)
{
    // create buffer object
    glGenBuffers( 1, tbo);
    glBindBuffer( GL_TEXTURE_BUFFER_EXT, *tbo);

    // initialize buffer object
    glBufferData( GL_TEXTURE_BUFFER_EXT, size, 0, GL_DYNAMIC_DRAW);

	//tex
	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, *tex);
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, *tbo); 
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*tbo));

    checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete TBO
////////////////////////////////////////////////////////////////////////////////
void deleteTBO( GLuint* tbo)
{
    glBindBuffer( 1, *tbo);
    glDeleteBuffers( 1, tbo);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*tbo));

    *tbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Vertex Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereVertex(float3* data, int slice, int stack) {
	float PI = 3.14159265358;
    
	double sl = 2*PI*slice/SPHERE_SLICES;
    double st = 2*PI*stack/SPHERE_STACKS;
 
    data->x = cos(st)*sin(sl) * SPHERE_RADIUS;
    data->y = sin(st)*sin(sl) * SPHERE_RADIUS;
    data->z = cos(sl) * SPHERE_RADIUS;
}


////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Normal Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereNormal(float3* data, int slice, int stack) {
	float PI = 3.14159265358;
    
	double sl = 2*PI*slice/SPHERE_SLICES;
    double st = 2*PI*stack/SPHERE_STACKS;
 
    data->x = cos(st)*sin(sl);
    data->y = sin(st)*sin(sl);
    data->z = cos(sl);
}


////////////////////////////////////////////////////////////////////////////////
//! Set Vertex Buffer Data
////////////////////////////////////////////////////////////////////////////////
void setVertexBufferData()
{
	int slice, stack;
	int i;

	// upload vertex points data
	glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
	float3* verts =( float3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice<SPHERE_SLICES/2; slice++) {
		for (stack=0; stack<=SPHERE_STACKS; stack++) {
			setSphereVertex(&verts[i++], slice, stack);
			setSphereVertex(&verts[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// upload vertex normal data
	glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
	float3* normals =( float3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice<SPHERE_SLICES/2; slice++) {
		for (stack=0; stack<=SPHERE_STACKS; stack++) {
			setSphereNormal(&normals[i++], slice, stack);
			setSphereNormal(&normals[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{

	//CUDA start Timing
	CUT_SAFE_CALL( cutStartTimer( timer));

	// run CUDA kernel to generate vertex positions
    runCuda();
    
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
     //set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

   	//zoom in z; translate x,y
	glTranslatef(translate_x, translate_y, translate_z); 
	//rotation about centre
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.0, 1.0);

	//Set light position
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);

	//At the end of one full layer iteration, Particles can only be in the testingActive state.
	//Draw Particle Agents in testingActive state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Particle_testingActive_displacementTex);
	//loop
	for (int i=0; i< get_agent_Particle_testingActive_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	
	}

	//CUDA stop timing
	cudaThreadSynchronize();
	glFlush();
	CUT_SAFE_CALL( cutStopTimer( timer));

	if(frame_count == average){
		char title [100];
		sprintf(title, "Execution & Rendering Total: %f (FPS)", average/(cutGetTimerValue( timer)/1000.0f));
		glutSetWindowTitle(title);

		//reset
		frame_count = 0;
		CUT_SAFE_CALL( cutResetTimer( timer));
	}else{
		frame_count++;
	}

    glutSwapBuffers();
    glutPostRedisplay();
    }


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {
    case( 27) :
        deleteVBO( &sphereVerts);
		deleteVBO( &sphereNormals);
		deleteTBO( &Particle_testingActive_tbo);
        exit( 0);
        break;
    
    //Extra options
    //Togle dark matter visualisation on/off
    case( 'd') :
        displayingDarkMatter=!displayingDarkMatter;
        break;
    
    //Toggle baryonic matter visualisation on/off
    case( 'b') :
        displayingBaryonicMatter=!displayingBaryonicMatter;
        break;

	//toggle pause
    case ( 'p'):
		paused=!paused;
		break;
	
	//Allow user to update simulation variables.
    case ( 'u'):
		//halt processing while getting info
		if(!paused){
			paused=true;
			updateSimulationVars();
			paused=false;
		}
		else{
			updateSimulationVars();
		}
		break;
	
	//Print a snapshot of simulation information
	case ( 'i'):
		//halt processing while dumping info
		if(!paused){
			paused=true;
			printSimulationInformation();
			paused=false;
		}
		else{
			printSimulationInformation();
		}
	}
    
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

	//left 
    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    }
    //right 
    else if (mouse_buttons & 4) {
		translate_z += dy * VIEW_DISTANCE* 0.005;
	}
	
	//middle
	else if (mouse_buttons & 3) {
		translate_x += dx * 0.001;
		translate_y += dy * -0.001;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	}

void checkGLError(){
  int Error;
  if((Error = glGetError()) != GL_NO_ERROR)
  {
    const char* Message = (const char*)gluErrorString(Error);
    fprintf(stderr, "OpenGL Error : %s\n", Message);
  }
  }