

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

#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include <header.h>
#include <cutil_math.h>




/* Agent count constants */

__constant__ int d_xmachine_memory_Particle_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_Particle_testingActive_count;

__constant__ int d_xmachine_memory_Particle_updatingPosition_count;


/* Message constants */

/* particleVariables Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_particleVariables_count;         /**< message list counter*/
__constant__ int d_message_particleVariables_output_type;   /**< message output type (single or optional)*/

	
    
//include each function file

#include "functions.c"
    
/* Texture bindings */

    
#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) (__mul24((s + d_PADDING), i)+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ int next_cell3D(int3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ int next_cell2D(int3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}


/** broadcastVariables_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Particle_list representing agent i the current state
 * @param nextState xmachine_memory_Particle_list representing agent i the next state
 */
 __global__ void broadcastVariables_function_filter(xmachine_memory_Particle_list* currentState, xmachine_memory_Particle_list* nextState)
 {
	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Particle_count){
	
		//apply the filter
		if (currentState->isActive[index]>0)
		{	//copy agent data to newstate list
			nextState->isDark[index] = currentState->isDark[index];
			nextState->isActive[index] = currentState->isActive[index];
			nextState->particleGroup[index] = currentState->particleGroup[index];
			nextState->mass[index] = currentState->mass[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->z[index] = currentState->z[index];
			nextState->xVel[index] = currentState->xVel[index];
			nextState->yVel[index] = currentState->yVel[index];
			nextState->zVel[index] = currentState->zVel[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Particle agent functions */

/** reset_Particle_scan_input
 * Particle agent reset scan input function
 * @param agents The xmachine_memory_Particle_list agent list
 */
__global__ void reset_Particle_scan_input(xmachine_memory_Particle_list* agents){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Particle_Agents
 * Particle scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Particle_list agent list destination
 * @param agents_src xmachine_memory_Particle_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Particle_Agents(xmachine_memory_Particle_list* agents_dst, xmachine_memory_Particle_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write
		agents_dst->_position[output_index] = output_index;
		agents_dst->isDark[output_index] = agents_src->isDark[index];
		agents_dst->isActive[output_index] = agents_src->isActive[index];
		agents_dst->particleGroup[output_index] = agents_src->particleGroup[index];
		agents_dst->mass[output_index] = agents_src->mass[index];
		agents_dst->x[output_index] = agents_src->x[index];
		agents_dst->y[output_index] = agents_src->y[index];
		agents_dst->z[output_index] = agents_src->z[index];
		agents_dst->xVel[output_index] = agents_src->xVel[index];
		agents_dst->yVel[output_index] = agents_src->yVel[index];
		agents_dst->zVel[output_index] = agents_src->zVel[index];
	}
}

/** append_Particle_Agents
 * Particle scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Particle_list agent list destination
 * @param agents_src xmachine_memory_Particle_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Particle_Agents(xmachine_memory_Particle_list* agents_dst, xmachine_memory_Particle_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->isDark[output_index] = agents_src->isDark[index];
	    agents_dst->isActive[output_index] = agents_src->isActive[index];
	    agents_dst->particleGroup[output_index] = agents_src->particleGroup[index];
	    agents_dst->mass[output_index] = agents_src->mass[index];
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->z[output_index] = agents_src->z[index];
	    agents_dst->xVel[output_index] = agents_src->xVel[index];
	    agents_dst->yVel[output_index] = agents_src->yVel[index];
	    agents_dst->zVel[output_index] = agents_src->zVel[index];
    }
}

/** add_Particle_agent
 * Continuous Particle agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Particle_list to add agents to 
 * @param isDark agent variable of type int
 * @param isActive agent variable of type int
 * @param particleGroup agent variable of type int
 * @param mass agent variable of type float
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param xVel agent variable of type float
 * @param yVel agent variable of type float
 * @param zVel agent variable of type float
 */
template <int AGENT_TYPE>
__device__ void add_Particle_agent(xmachine_memory_Particle_list* agents, int isDark, int isActive, int particleGroup, float mass, float x, float y, float z, float xVel, float yVel, float zVel){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = __mul24(blockDim.x, gridDim.x);
		int2 global_position;
		global_position.x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		global_position.y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
		index = global_position.x + __mul24(global_position.y, width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->isDark[index] = isDark;
	agents->isActive[index] = isActive;
	agents->particleGroup[index] = particleGroup;
	agents->mass[index] = mass;
	agents->x[index] = x;
	agents->y[index] = y;
	agents->z[index] = z;
	agents->xVel[index] = xVel;
	agents->yVel[index] = yVel;
	agents->zVel[index] = zVel;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Particle_agent(xmachine_memory_Particle_list* agents, int isDark, int isActive, int particleGroup, float mass, float x, float y, float z, float xVel, float yVel, float zVel){
    add_Particle_agent<DISCRETE_2D>(agents, isDark, isActive, particleGroup, mass, x, y, z, xVel, yVel, zVel);
}

/** reorder_Particle_agents
 * Continuous Particle agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Particle_agents(unsigned int* values, xmachine_memory_Particle_list* unordered_agents, xmachine_memory_Particle_list* ordered_agents)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->isDark[index] = unordered_agents->isDark[old_pos];
	ordered_agents->isActive[index] = unordered_agents->isActive[old_pos];
	ordered_agents->particleGroup[index] = unordered_agents->particleGroup[old_pos];
	ordered_agents->mass[index] = unordered_agents->mass[old_pos];
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->z[index] = unordered_agents->z[old_pos];
	ordered_agents->xVel[index] = unordered_agents->xVel[old_pos];
	ordered_agents->yVel[index] = unordered_agents->yVel[old_pos];
	ordered_agents->zVel[index] = unordered_agents->zVel[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created particleVariables message functions */


/** add_particleVariables_message
 * Add non partitioned or spatially partitioned particleVariables message
 * @param messages xmachine_message_particleVariables_list message list to add too
 * @param mass agent variable of type float
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 */
__device__ void add_particleVariables_message(xmachine_message_particleVariables_list* messages, float mass, float x, float y, float z){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x + d_message_particleVariables_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_particleVariables_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_particleVariables_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_particleVariables Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->mass[index] = mass;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;

}

/**
 * Scatter non partitioned or spatially partitioned particleVariables message (for optional messages)
 * @param messages scatter_optional_particleVariables_messages Sparse xmachine_message_particleVariables_list message list
 * @param message_swap temp xmachine_message_particleVariables_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_particleVariables_messages(xmachine_message_particleVariables_list* messages, xmachine_message_particleVariables_list* messages_swap){
	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_particleVariables_count;

		//AoS - xmachine_message_particleVariables Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->mass[output_index] = messages_swap->mass[index];
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];				
	}
}

/** reset_particleVariables_swaps
 * Reset non partitioned or spatially partitioned particleVariables message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_particleVariables_swaps(xmachine_message_particleVariables_list* messages_swap){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_particleVariables* get_first_particleVariables_message(xmachine_message_particleVariables_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = __mul24(ceil((float)d_message_particleVariables_count/ blockDim.x), blockDim.x);

	//if no messages then return false
	if (wrap_size == 0)
		return false;

	//global thread index
	int global_index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_particleVariables Coalesced memory read
	xmachine_message_particleVariables temp_message;
	temp_message._position = messages->_position[index];
	temp_message.mass = messages->mass[index];
	temp_message.x = messages->x[index];
	temp_message.y = messages->y[index];
	temp_message.z = messages->z[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_particleVariables));
	xmachine_message_particleVariables* sm_message = ((xmachine_message_particleVariables*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_particleVariables*)&message_share[d_SM_START]);
}

__device__ xmachine_message_particleVariables* get_next_particleVariables_message(xmachine_message_particleVariables* message, xmachine_message_particleVariables_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_particleVariables_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_particleVariables_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP(__mul24(blockDim.x, blockIdx.x), wrap_size))
		return false;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we dont change shared memeory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_particleVariables Coalesced memory read
		int index = __mul24(tile, blockDim.x) + threadIdx.x;
		xmachine_message_particleVariables temp_message;
		temp_message._position = messages->_position[index];
		temp_message.mass = messages->mass[index];
		temp_message.x = messages->x[index];
		temp_message.y = messages->y[index];
		temp_message.z = messages->z[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_particleVariables));
		xmachine_message_particleVariables* sm_message = ((xmachine_message_particleVariables*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we dont start returning messages untill all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_particleVariables));
	return ((xmachine_message_particleVariables*)&message_share[message_index]);
}


	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created GPU kernals  */



/**
 *
 */
__global__ void GPUFLAME_setIsActive(xmachine_memory_Particle_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index > d_xmachine_memory_Particle_count)
        return;
    

	//SoA to AoS - xmachine_memory_setIsActive Coalesced memory read
	xmachine_memory_Particle agent;
	agent.isDark = agents->isDark[index];
	agent.isActive = agents->isActive[index];
	agent.particleGroup = agents->particleGroup[index];
	agent.mass = agents->mass[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.xVel = agents->xVel[index];
	agent.yVel = agents->yVel[index];
	agent.zVel = agents->zVel[index];

	//FLAME function call
	int dead = !setIsActive(&agent);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_setIsActive Coalesced memory write
	agents->isDark[index] = agent.isDark;
	agents->isActive[index] = agent.isActive;
	agents->particleGroup[index] = agent.particleGroup;
	agents->mass[index] = agent.mass;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->xVel[index] = agent.xVel;
	agents->yVel[index] = agent.yVel;
	agents->zVel[index] = agent.zVel;
}

/**
 *
 */
__global__ void GPUFLAME_broadcastVariables(xmachine_memory_Particle_list* agents, xmachine_message_particleVariables_list* particleVariables_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index > d_xmachine_memory_Particle_count)
        return;
    

	//SoA to AoS - xmachine_memory_broadcastVariables Coalesced memory read
	xmachine_memory_Particle agent;
	agent.isDark = agents->isDark[index];
	agent.isActive = agents->isActive[index];
	agent.particleGroup = agents->particleGroup[index];
	agent.mass = agents->mass[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.xVel = agents->xVel[index];
	agent.yVel = agents->yVel[index];
	agent.zVel = agents->zVel[index];

	//FLAME function call
	int dead = !broadcastVariables(&agent, particleVariables_messages	);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_broadcastVariables Coalesced memory write
	agents->isDark[index] = agent.isDark;
	agents->isActive[index] = agent.isActive;
	agents->particleGroup[index] = agent.particleGroup;
	agents->mass[index] = agent.mass;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->xVel[index] = agent.xVel;
	agents->yVel[index] = agent.yVel;
	agents->zVel[index] = agent.zVel;
}

/**
 *
 */
__global__ void GPUFLAME_updatePosition(xmachine_memory_Particle_list* agents, xmachine_message_particleVariables_list* particleVariables_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_updatePosition Coalesced memory read
	xmachine_memory_Particle agent;
	agent.isDark = agents->isDark[index];
	agent.isActive = agents->isActive[index];
	agent.particleGroup = agents->particleGroup[index];
	agent.mass = agents->mass[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.xVel = agents->xVel[index];
	agent.yVel = agents->yVel[index];
	agent.zVel = agents->zVel[index];

	//FLAME function call
	int dead = !updatePosition(&agent, particleVariables_messages);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_updatePosition Coalesced memory write
	agents->isDark[index] = agent.isDark;
	agents->isActive[index] = agent.isActive;
	agents->particleGroup[index] = agent.particleGroup;
	agents->mass[index] = agent.mass;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->xVel[index] = agent.xVel;
	agents->yVel[index] = agent.yVel;
	agents->zVel[index] = agent.zVel;
}

	
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static uint2 RNG_rand48_iterate_single(uint2 Xn, uint2 A, uint2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return make_uint2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = __mul24(blockDim.x, gridDim.x);
		int2 global_position;
		global_position.x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		global_position.y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
		index = global_position.x + __mul24(global_position.y, width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	uint2 state = rand48->seeds[index];
	uint2 A = rand48->A;
	uint2 C = rand48->C;

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_
