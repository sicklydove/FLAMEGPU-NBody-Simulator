
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>
#include <stdio.h>

__FLAME_GPU_INIT_FUNC__ void initConstants(){

	float dt=0.0001f; 
	float gravConstant=1;
    float velocityDamper=0.25;
    float sphereRadius = 0.0035;
	int numPartitions=2;

	/*
	printf("Input dt\n");
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

/**
 * outputdata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int outputdata(xmachine_memory_Particle* xmemory, xmachine_message_location_list* location_messages){
    
	add_location_message(location_messages, xmemory->mass, xmemory->x, xmemory->y, xmemory->z);
    
    return 0;
}

/**
optimisation

*/
__FLAME_GPU_FUNC__ int setActive(xmachine_memory_Particle* xmemory, xmachine_message_itNumMessage_list* itNumMessage_messages){

	
	int itNum;
	int offset=xmemory->initialOffset;

	xmachine_message_itNumMessage* current_message = get_first_itNumMessage_message(itNumMessage_messages);
	while(current_message){
	  itNum=current_message->itNum;

	  if((itNum+offset)%NUM_PARTITIONS==0){
	    xmemory->isActive=1;
	  }
	  else{
		  xmemory->isActive=0;
	  }
      current_message = get_next_itNumMessage_message(current_message, itNumMessage_messages);
	}
	xmemory->debug1=xmemory->id;
    return 0;
}

__FLAME_GPU_FUNC__ int notoutputdata(xmachine_memory_Particle* xmemory){
    //Do nothing - this just moves states. hack?
    return 0;
}

__FLAME_GPU_FUNC__ int increaseIterationNum(xmachine_memory_simulationVarsAgent* xmemory, xmachine_message_itNumMessage_list* itNumMessage_messages){
    int currentState=xmemory->itNum;
    currentState++;
	xmemory->itNum=currentState;

	add_itNumMessage_message(itNumMessage_messages, xmemory->itNum);
    return 0;
}

/**
 * inputdata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.
	
	SHOCK HORROR!
	IT'S ITERATION THAT SLOWS IT...
	
	W/O ITERATION: 55
	W/ ITERATION AND NO CALCULATIONS: 18
	W/ ITERATION AND CACLULATIONS: 8

	15K AGENTS
	*/

__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_Particle* xmemory, xmachine_message_location_list* location_messages){

	float3 agent_position = make_float3(xmemory->x, xmemory->y, xmemory->z);
	float3 agent_accn=make_float3(0.0,0.0,0.0);
	xmachine_message_location* current_message = get_first_location_message(location_messages);
	
	while (current_message){
		float3 accn = make_float3(0,0,0);
		float3 currentMessagePosition=make_float3(current_message->x, current_message->y, current_message->z);
		float3 positionDifference=currentMessagePosition-agent_position;
		float abs_distance=sqrt(pow(positionDifference.x,2)+pow(positionDifference.y,2)+pow(positionDifference.z,2));

		float3 topHalfEqn=positionDifference*current_message->mass*GRAV_CONST;

		if(abs_distance>5*MIN_INTERRACTION_RAD){
			float lowerHalfEqn=pow((pow(abs_distance, 2)+pow(VELOCITY_DAMP,2)), (3/2));
			accn=topHalfEqn/lowerHalfEqn;
		}

		agent_accn+=accn;
		current_message = get_next_location_message(current_message, location_messages);
	}
        
	float3 vels=make_float3(xmemory->xVel, xmemory->yVel, xmemory->zVel);
	float varPos=agent_position.x;

	varPos+=(DELTA_T*vels.x);
	varPos+=(0.5*(agent_accn.x)*(DELTA_T*DELTA_T));
	xmemory->x=varPos;		

	varPos=agent_position.y;
	varPos+=(DELTA_T*vels.y);
	varPos+=(0.5*(agent_accn.y)*(DELTA_T*DELTA_T));
	xmemory->y=varPos;

	varPos=agent_position.z;
	varPos+=(DELTA_T*vels.z);
	varPos+=(0.5*(agent_accn.z)*(DELTA_T*DELTA_T));
	xmemory->z=varPos;

	//Velocities
	vels.x+=(agent_accn.x*DELTA_T);
	vels.y+=(agent_accn.y*DELTA_T);
	vels.z+=(agent_accn.z*DELTA_T);

	xmemory->xVel=vels.x;
	xmemory->yVel=vels.y;
	xmemory->zVel=vels.z;
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
