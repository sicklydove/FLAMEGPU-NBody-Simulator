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
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>


extern void set_DELTA_T(float* h_DELTA_T);
extern void set_GRAV_CONST(float* h_GRAV_CONST);
extern void set_VELOCITY_DAMP(float* h_VELOCITY_DAMP);
extern void set_MIN_INTERRACTION_RAD(float* h_MIN_INTERRACTION);
extern void set_NUM_PARTITIONS(float* h_NUM_PARTITIONS);

float dt; 
float gravConstant;
float velocityDamper;
float sphereRadius;
int numPartitions;


void updateSimulationVars(){

	dt=0.001f; 
	gravConstant=1;
    velocityDamper=0.3;
    sphereRadius = 0.0035;
	numPartitions=1;

	/*
	printf("\nInput dt\n");
	scanf("%f", &dt);
	printf("\nInput gravitational constant:\n");
	scanf("%f", &gravConstant);
	printf("\nInput velocity dampening factor:\n");
	scanf("%f", &velocityDamper);
	printf("\nInput minimum radius of interraction:\n");
	scanf("%f", &velocityDamper);
	printf("\nInput number of timestep slices (optimisation):\n");
	scanf("%d", &numPartitions);
	*/

	set_DELTA_T(&dt);
	set_GRAV_CONST(&gravConstant);
	set_VELOCITY_DAMP(&velocityDamper);
	set_MIN_INTERRACTION_RAD(&sphereRadius);
	set_NUM_PARTITIONS(&numPartitions);
}

void printSimulationInformation(int itNum){
	
	printf("\nIteration number: ");
	printf("%d", itNum);
	printf("\nTimestep size: ");
	printf("%f", dt);
	printf("\nGravitational constant: ");
	printf("%f", gravConstant);
	printf("\nCelocity dampening factor: ");
	printf("%f", velocityDamper);
	printf("\nMinimum radius of interraction: ");
	printf("%f", sphereRadius);
	printf("\nNumber of timestep slices: ");
	printf("%d", numPartitions);
}