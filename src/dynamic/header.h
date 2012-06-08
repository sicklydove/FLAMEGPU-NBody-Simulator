
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

#ifndef __HEADER
#define __HEADER

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__ 



/* Agent population size definifions must be a multiple of THREADS_PER_TILE (defualt 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 131072

//Maximum population size of xmachine_memory_Particle
#define xmachine_memory_Particle_MAX 131072
  
  
/* Message poulation size definitions */
//Maximum population size of xmachine_mmessage_particleVariables
#define xmachine_message_particleVariables_MAX 131072



/* Spatial partitioning grid size definitions */
  
  
/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_Particle
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Particle
{
    int isDark;    /**< X-machine memory variable isDark of type int.*/
    int isActive;    /**< X-machine memory variable isActive of type int.*/
    int particleGroup;    /**< X-machine memory variable particleGroup of type int.*/
    float mass;    /**< X-machine memory variable mass of type float.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float z;    /**< X-machine memory variable z of type float.*/
    float xVel;    /**< X-machine memory variable xVel of type float.*/
    float yVel;    /**< X-machine memory variable yVel of type float.*/
    float zVel;    /**< X-machine memory variable zVel of type float.*/
};



/* Message structures */

/** struct xmachine_message_particleVariables
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_particleVariables
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    float mass;        /**< Message variable mass of type float.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_Particle_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Particle_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Particle_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Particle_MAX];  /**< Used during parallel prefix sum */
    
    int isDark [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list isDark of type int.*/
    int isActive [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list isActive of type int.*/
    int particleGroup [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list particleGroup of type int.*/
    float mass [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list mass of type float.*/
    float x [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list y of type float.*/
    float z [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list z of type float.*/
    float xVel [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list xVel of type float.*/
    float yVel [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list yVel of type float.*/
    float zVel [xmachine_memory_Particle_MAX];    /**< X-machine memory variable list zVel of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_particleVariables_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_particleVariables_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_particleVariables_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_particleVariables_MAX];  /**< Used during parallel prefix sum */
    
    float mass [xmachine_message_particleVariables_MAX];    /**< Message memory variable list mass of type float.*/
    float x [xmachine_message_particleVariables_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_particleVariables_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_particleVariables_MAX];    /**< Message memory variable list z of type float.*/
    
};



/* Spatialy Partitioned Message boundary Matrices */



/* Random */
/** struct RNG_rand48 
 *	structure used to hold list seeds
 */
struct RNG_rand48
{
  uint2 A, C;
  uint2 seeds[buffer_size_MAX];
};


/* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

/**
 * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
 * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
 * not work for DISCRETE_2D agent.
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * setIsActive FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int setIsActive(xmachine_memory_Particle* agent);

/**
 * broadcastVariables FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param particleVariables_messages Pointer to output message list of type xmachine_message_particleVariables_list. Must be passed as an argument to the add_particleVariables_message function ??.
 */
__FLAME_GPU_FUNC__ int broadcastVariables(xmachine_memory_Particle* agent, xmachine_message_particleVariables_list* particleVariables_messages);

/**
 * updatePosition FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param particleVariables_messages  particleVariables_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_particleVariables_message and get_next_particleVariables_message functions.
 */
__FLAME_GPU_FUNC__ int updatePosition(xmachine_memory_Particle* agent, xmachine_message_particleVariables_list* particleVariables_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) particleVariables message implemented in FLAMEGPU_Kernels */

/** add_particleVariables_agent
 * Function for all types of message partitioning
 * Adds a new particleVariables agent to the xmachine_memory_particleVariables_list list using a linear mapping
 * @param agents	xmachine_memory_particleVariables_list agent list
 * @param mass	message variable of type float
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_particleVariables_message(xmachine_message_particleVariables_list* particleVariables_messages, float mass, float x, float y, float z);
 
/** get_first_particleVariables_message
 * Get first message function for non partitioned (brute force) messages
 * @param particleVariables_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_particleVariables * get_first_particleVariables_message(xmachine_message_particleVariables_list* particleVariables_messages);

/** get_next_particleVariables_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param particleVariables_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_particleVariables * get_next_particleVariables_message(xmachine_message_particleVariables* current, xmachine_message_particleVariables_list* particleVariables_messages);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_Particle_agent
 * Adds a new continuous valued Particle agent to the xmachine_memory_Particle_list list using a linear mapping
 * @param agents xmachine_memory_Particle_list agent list
 * @param isDark	agent agent variable of type int
 * @param isActive	agent agent variable of type int
 * @param particleGroup	agent agent variable of type int
 * @param mass	agent agent variable of type float
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param z	agent agent variable of type float
 * @param xVel	agent agent variable of type float
 * @param yVel	agent agent variable of type float
 * @param zVel	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_Particle_agent(xmachine_memory_Particle_list* agents, int isDark, int isActive, int particleGroup, float mass, float x, float y, float z, float xVel, float yVel, float zVel);


  
/* Simulation function prototypes implemented in simulation.cu */

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input	XML file path for agent initial configuration
 */
extern "C" void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern "C" void cleanup();

/** singleIteration
 *	Performs a single itteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern "C" void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	itteration_number
 * @param h_Particles Pointer to agent list on the host
 * @param d_Particles Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Particle_count Pointer to agent counter
 */
extern "C" void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Particle_list* h_Particles_testingActive, xmachine_memory_Particle_list* d_Particles_testingActive, int h_xmachine_memory_Particle_testingActive_count,xmachine_memory_Particle_list* h_Particles_updatingPosition, xmachine_memory_Particle_list* d_Particles_updatingPosition, int h_xmachine_memory_Particle_updatingPosition_count);


/** readInitialStates
 * Reads the current agent data fromt he device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_Particles Pointer to agent list on the host
 * @param h_xmachine_memory_Particle_count Pointer to agent counter
 */
extern "C" void readInitialStates(char* inputpath, xmachine_memory_Particle_list* h_Particles, int* h_xmachine_memory_Particle_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_Particle_MAX_count
 * Gets the max agent count for the Particle agent type 
 * @return		the maximum Particle agent count
 */
extern "C" int get_agent_Particle_MAX_count();


/** get_agent_Particle_testingActive_count
 * Gets the agent count for the Particle agent type in state testingActive
 * @return		the current Particle agent count in state testingActive
 */
extern "C" int get_agent_Particle_testingActive_count();

/** get_device_Particle_testingActive_agents
 * Gets a pointer to xmachine_memory_Particle_list on the GPU device
 * @return		a xmachine_memory_Particle_list on the GPU device
 */
extern "C" xmachine_memory_Particle_list* get_device_Particle_testingActive_agents();

/** get_host_Particle_testingActive_agents
 * Gets a pointer to xmachine_memory_Particle_list on the CPU host
 * @return		a xmachine_memory_Particle_list on the CPU host
 */
extern "C" xmachine_memory_Particle_list* get_host_Particle_testingActive_agents();


/** sort_Particles_testingActive
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Particles_testingActive(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Particle_list* agents));


/** get_agent_Particle_updatingPosition_count
 * Gets the agent count for the Particle agent type in state updatingPosition
 * @return		the current Particle agent count in state updatingPosition
 */
extern "C" int get_agent_Particle_updatingPosition_count();

/** get_device_Particle_updatingPosition_agents
 * Gets a pointer to xmachine_memory_Particle_list on the GPU device
 * @return		a xmachine_memory_Particle_list on the GPU device
 */
extern "C" xmachine_memory_Particle_list* get_device_Particle_updatingPosition_agents();

/** get_host_Particle_updatingPosition_agents
 * Gets a pointer to xmachine_memory_Particle_list on the CPU host
 * @return		a xmachine_memory_Particle_list on the CPU host
 */
extern "C" xmachine_memory_Particle_list* get_host_Particle_updatingPosition_agents();


/** sort_Particles_updatingPosition
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Particles_updatingPosition(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Particle_list* agents));


  
  
/* global constant variables */

__constant__ float DELTA_T;

__constant__ float GRAV_CONST;

__constant__ float VELOCITY_DAMP;

__constant__ float MIN_INTERACTION_RAD;

__constant__ int NUM_PARTITIONS;

__constant__ int SIMULATION_ITNUM;

/** set_DELTA_T
 * Sets the constant variable DELTA_T on the device which can then be used in the agent functions.
 * @param h_DELTA_T value to set the variable
 */
extern "C" void set_DELTA_T(float* h_DELTA_T);

/** set_GRAV_CONST
 * Sets the constant variable GRAV_CONST on the device which can then be used in the agent functions.
 * @param h_GRAV_CONST value to set the variable
 */
extern "C" void set_GRAV_CONST(float* h_GRAV_CONST);

/** set_VELOCITY_DAMP
 * Sets the constant variable VELOCITY_DAMP on the device which can then be used in the agent functions.
 * @param h_VELOCITY_DAMP value to set the variable
 */
extern "C" void set_VELOCITY_DAMP(float* h_VELOCITY_DAMP);

/** set_MIN_INTERACTION_RAD
 * Sets the constant variable MIN_INTERACTION_RAD on the device which can then be used in the agent functions.
 * @param h_MIN_INTERACTION_RAD value to set the variable
 */
extern "C" void set_MIN_INTERACTION_RAD(float* h_MIN_INTERACTION_RAD);

/** set_NUM_PARTITIONS
 * Sets the constant variable NUM_PARTITIONS on the device which can then be used in the agent functions.
 * @param h_NUM_PARTITIONS value to set the variable
 */
extern "C" void set_NUM_PARTITIONS(int* h_NUM_PARTITIONS);

/** set_SIMULATION_ITNUM
 * Sets the constant variable SIMULATION_ITNUM on the device which can then be used in the agent functions.
 * @param h_SIMULATION_ITNUM value to set the variable
 */
extern "C" void set_SIMULATION_ITNUM(int* h_SIMULATION_ITNUM);


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
float3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
float3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in seperate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values fromt the main function used with GLUT
 */
extern "C" void initVisualisation();

extern "C" void runVisualisation();

#endif

#endif //__HEADER

