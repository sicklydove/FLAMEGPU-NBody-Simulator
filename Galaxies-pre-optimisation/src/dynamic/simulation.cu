
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

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <cutil_math.h>
#include <cudpp.h>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"

/* SM padding and offset variables */
int SM_START;
int PADDING;

/* Agent Memory */

/* Particle Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Particle_list* d_Particles;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Particle_list* d_Particles_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Particle_list* d_Particles_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_Particle_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Particle_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Particle_values;  /**< Agent sort identifiers value */
    
/* Particle state variables */
xmachine_memory_Particle_list* h_Particles_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Particle_list* d_Particles_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Particle_default_count;   /**< Agent population size counter */ 


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
CUDPPHandle cudpp_scanplan;   /**< CUDPPHandle*/
CUDPPHandle cudpp_sortplan;   /**< CUDPPHandle*/
int cudpp_last_sum;           /**< Indicates if the position (in message list) of last message*/
int cudpp_last_included;      /**< Indicates if last sum value is included in the total sum count*/
int radix_keybits = 32;

/* Agent function prototypes */

/** Particle_outputdata
 * Agent function prototype for outputdata function of Particle agent
 */
void Particle_outputdata();

/** Particle_inputdata
 * Agent function prototype for inputdata function of Particle agent
 */
void Particle_inputdata();

  
CUDPPHandle* getCUDPPSortPlan(){
    return &cudpp_sortplan;
}


void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
    int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(0);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
    printf("Simulation requires full precision double values\n");
    if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
        printf("Error: Hardware does not support full precision double values!\n");
        exit(0);
    }
    
#endif

    //check 32 or 64bit
    x64_sys = (sizeof(void*)==8);
    if (x64_sys)
    {
        printf("64Bit System Detected\n");
    }
    else
    {
        printf("32Bit System Detected\n");
    }

    //check for FERMI
	if ((deviceProp.major >= 2)){
		printf("FERMI Card detected (compute 2.0)\n");
        if (x64_sys){
            SM_START = 8;
            PADDING = 0;
        }else
        {
            SM_START = 4;
            PADDING = 0;
        }
	}	
    //not fermi
    else{
  	    printf("Pre FERMI Card detected (less than compute 2.0)\n");
        if (x64_sys){
            SM_START = 0;
            PADDING = 4;
        }else
        {
            SM_START = 0;
            PADDING = 4;
        }
    }
  
    //copy padding and offset to GPU
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));

        
}


void initialise(char * inputfile){

    //set the padding and offset values depending on architecture and OS
    setPaddingAndOffset();
  

	printf("Allocating Host and Device memeory\n");
  
	/* Agent memory allocation (CPU) */
	int xmachine_Particle_SoA_size = sizeof(xmachine_memory_Particle_list);
	h_Particles_default = (xmachine_memory_Particle_list*)malloc(xmachine_Particle_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);

    //Exit if agent or message buffer sizes are to small for function outpus


	//read initial states
	readInitialStates(inputfile, h_Particles_default, &h_xmachine_memory_Particle_default_count);
	
	
	/* Particle Agent memory allocation (GPU) */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Particles, xmachine_Particle_SoA_size));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Particles_swap, xmachine_Particle_SoA_size));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Particles_new, xmachine_Particle_SoA_size));
    //continuous agent sort identifiers
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_xmachine_memory_Particle_keys, xmachine_memory_Particle_MAX* sizeof(uint)));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_xmachine_memory_Particle_values, xmachine_memory_Particle_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Particles_default, xmachine_Particle_SoA_size));
	CUDA_SAFE_CALL( cudaMemcpy( d_Particles_default, h_Particles_default, xmachine_Particle_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	CUDA_SAFE_CALL( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
		

	/*Set global condition counts*/

	/* CUDPP Init */
	CUDPPConfiguration cudpp_config;
	cudpp_config.op = CUDPP_ADD;
	cudpp_config.datatype = CUDPP_INT;
	cudpp_config.algorithm = CUDPP_SCAN;
	cudpp_config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
	cudpp_scanplan = 0;
	CUDPPResult result = cudppPlan(&cudpp_scanplan, cudpp_config, buffer_size_MAX, 1, 0);  
	if (CUDPP_SUCCESS != result)
	{
		printf("Error creating CUDPPPlan\n");
		exit(-1);
	}

    /* Radix sort */
	CUDPPConfiguration cudpp_sort_config;
    cudpp_sort_config.algorithm = CUDPP_SORT_RADIX;
    cudpp_sort_config.datatype = CUDPP_UINT;
    cudpp_sort_config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	cudpp_sortplan = 0;
	CUDPPResult sort_result = cudppPlan(&cudpp_sortplan, cudpp_sort_config, buffer_size_MAX, 1, 0);  
	if (CUDPP_SUCCESS != result)
	{
		printf("Error creating CUDPPPlan for radix sort\n");
		exit(-1);
	}

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	CUDA_SAFE_CALL( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	
} 


void sort_Particles_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Particle_list* agents))
{
	dim3 grid;
	dim3 threads;
	int tile_size = (int)ceil((float)h_xmachine_memory_Particle_default_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;

	//generate sort keys
	generate_key_value_pairs<<<grid, threads>>>(d_xmachine_memory_Particle_keys, d_xmachine_memory_Particle_values, d_Particles_default);
	CUT_CHECK_ERROR("Kernel execution failed");
	
	//sort
	cudppSort(cudpp_sortplan, d_xmachine_memory_Particle_keys, d_xmachine_memory_Particle_values, radix_keybits, h_xmachine_memory_Particle_default_count);
	CUT_CHECK_ERROR("Kernel execution failed");

	//reorder agents
	reorder_Particle_agents<<<grid, threads>>>(d_xmachine_memory_Particle_values, d_Particles_default, d_Particles_swap);
	CUT_CHECK_ERROR("Kernel execution failed");

	//swap
	xmachine_memory_Particle_list* d_Particles_temp = d_Particles_default;
	d_Particles_default = d_Particles_swap;
	d_Particles_swap = d_Particles_temp;	
}


void cleanup(){

	/* Agent data free*/
	
	/* Particle Agent variables */
	CUDA_SAFE_CALL(cudaFree(d_Particles));
	CUDA_SAFE_CALL(cudaFree(d_Particles_swap));
	CUDA_SAFE_CALL(cudaFree(d_Particles_new));
	
	free( h_Particles_default);
	CUDA_SAFE_CALL(cudaFree(d_Particles_default));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	CUDA_SAFE_CALL(cudaFree(d_locations));
	CUDA_SAFE_CALL(cudaFree(d_locations_swap));
	
}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/
	h_message_location_count = 0;
	//upload to device constant
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	

	/* Call agent functions in order itterating through the layer functions */
	
	/* Layer 1*/
	Particle_outputdata();
	Particle_inputdata();
	

			
	//Syncronise thread blocks (and relax)
	cudaThreadSynchronize();
}

/* Environment functions */



/* Agent data access functions*/

    
int get_agent_Particle_MAX_count(){
    return xmachine_memory_Particle_MAX;
}


int get_agent_Particle_default_count(){
	//continuous agent
	return h_xmachine_memory_Particle_default_count;
	
}

xmachine_memory_Particle_list* get_device_Particle_default_agents(){
	return d_Particles_default;
}

xmachine_memory_Particle_list* get_host_Particle_default_agents(){
	return h_Particles_default;
}



/* Agent functions */


/** Particle_outputdata
 * Agent function prototype for outputdata function of Particle agent
 */
void Particle_outputdata(){
	dim3 grid;
	dim3 threads;
	int sm_size;
	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Particle_default_count == 0)
	{
		return;
	}
	
	
	//SET GRID AND BLOCK SIZES
	//set tile size depending on agent count, set a 1d grid and block
	int tile_size = (int)ceil((float)h_xmachine_memory_Particle_default_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;
	sm_size = SM_START;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Particle_list* Particles_default_temp = d_Particles;
	d_Particles = d_Particles_default;
	d_Particles_default = Particles_default_temp;
	//set working count to current state count
	h_xmachine_memory_Particle_count = h_xmachine_memory_Particle_default_count;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Particle_count, &h_xmachine_memory_Particle_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Particle_default_count = 0;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Particle_default_count, &h_xmachine_memory_Particle_default_count, sizeof(int)));	
	

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_Particle_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function outputdata\n");
		exit(0);
	}
	
	//SET THE OUTPUT MESSAGE TYPE
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_location_output_type = single_message;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (outputdata)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_outputdata<<<grid, threads, sm_size>>>(d_Particles, d_locations);
	CUT_CHECK_ERROR("Kernel execution failed");
    
    
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_Particle_count;	
	//Copy count to device
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Particle_default_count+h_xmachine_memory_Particle_count > xmachine_memory_Particle_MAX){
		printf("Error: Buffer size of outputdata agents in state default will be exceeded moving working agents to next state in function outputdata\n");
		exit(0);
	}
	//append agents to next state list
	append_Particle_Agents<<<grid, threads>>>(d_Particles_default, d_Particles, h_xmachine_memory_Particle_default_count, h_xmachine_memory_Particle_count);
	CUT_CHECK_ERROR("Kernel execution failed");
	//update new state agent size
	h_xmachine_memory_Particle_default_count += h_xmachine_memory_Particle_count;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Particle_default_count, &h_xmachine_memory_Particle_default_count, sizeof(int)));	
	
	
}



/** Particle_inputdata
 * Agent function prototype for inputdata function of Particle agent
 */
void Particle_inputdata(){
	dim3 grid;
	dim3 threads;
	int sm_size;
	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Particle_default_count == 0)
	{
		return;
	}
	
	
	//SET GRID AND BLOCK SIZES
	//set tile size depending on agent count, set a 1d grid and block
	int tile_size = (int)ceil((float)h_xmachine_memory_Particle_default_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;
	sm_size = SM_START;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Particle_list* Particles_default_temp = d_Particles;
	d_Particles = d_Particles_default;
	d_Particles_default = Particles_default_temp;
	//set working count to current state count
	h_xmachine_memory_Particle_count = h_xmachine_memory_Particle_default_count;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Particle_count, &h_xmachine_memory_Particle_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Particle_default_count = 0;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Particle_default_count, &h_xmachine_memory_Particle_default_count, sizeof(int)));	
	

	//******************************** AGENT FUNCTION *******************************

	
	//UPDATE SHARED MEMEORY SIZE FOR EACH FUNCTION INPUT
	//Continuous agent and message input has no partitioning
	sm_size += (threads.x * sizeof(xmachine_message_location));
	
    //all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (threads.x * PADDING);
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (inputdata)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_inputdata<<<grid, threads, sm_size>>>(d_Particles, d_locations);
	CUT_CHECK_ERROR("Kernel execution failed");
    
    
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Particle_default_count+h_xmachine_memory_Particle_count > xmachine_memory_Particle_MAX){
		printf("Error: Buffer size of inputdata agents in state default will be exceeded moving working agents to next state in function inputdata\n");
		exit(0);
	}
	//append agents to next state list
	append_Particle_Agents<<<grid, threads>>>(d_Particles_default, d_Particles, h_xmachine_memory_Particle_default_count, h_xmachine_memory_Particle_count);
	CUT_CHECK_ERROR("Kernel execution failed");
	//update new state agent size
	h_xmachine_memory_Particle_default_count += h_xmachine_memory_Particle_count;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Particle_default_count, &h_xmachine_memory_Particle_default_count, sizeof(int)));	
	
	
}


