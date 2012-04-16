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
#include <GlobalsController.h>

__FLAME_GPU_INIT_FUNC__ void initConstants(){
	
	char input;
	bool set=false;
	
	input=0;
	set=false;
	//ask for vis settings in visualisation

	printf("Use default simulation parameters y/n \n");
	while(!set){
		input=getchar();
		while(input == '\n') input=getchar();
		switch(input){
		case 'y': case 'Y':
			setSimulationDefaults();
			set=true;
			break;
		case 'n': case 'N':
			updateSimulationVars();
			set=true;
			break;
		default:
			printf("Invalid input. Use default simulation parameters? y/n \n");
			break;
		}
	}
}

__FLAME_GPU_FUNC__ int broadcastAndKeepState(xmachine_memory_Particle* xmemory, xmachine_message_particleVariables_list* particleVariables_messages){
	add_particleVariables_message(particleVariables_messages, xmemory->mass, xmemory->x, xmemory->y, xmemory->z);
    return 0;
}

__FLAME_GPU_FUNC__ int broadcastAndMoveState(xmachine_memory_Particle* xmemory, xmachine_message_particleVariables_list* particleVariables_messages){
	add_particleVariables_message(particleVariables_messages, xmemory->mass, xmemory->x, xmemory->y, xmemory->z);
    return 0;
}

__FLAME_GPU_FUNC__ int setIsActive(xmachine_memory_Particle* xmemory, xmachine_message_itNumMessage_list* itNumMessage_messages){
	int particleGroup=xmemory->particleGroup;
	xmachine_message_itNumMessage* current_message = get_first_itNumMessage_message(itNumMessage_messages);
	int itNum=current_message->itNum;
	if((itNum+particleGroup)%NUM_PARTITIONS==0)
		xmemory->isActive=1;
	else
		xmemory->isActive=0;
    return 0;
}


/**
 * outputdata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param particleVariables_messages Pointer to output message list of type xmachine_message_particleVariables_list. Must be passed as an argument to the add_particleVariables_message function ??.
 */
__FLAME_GPU_FUNC__ int broadcastItNum(xmachine_memory_simulationVarsAgent* xmemory, xmachine_message_itNumMessage_list* itNumMessage_messages){
    int currentState=xmemory->itNum;
    add_itNumMessage_message(itNumMessage_messages, xmemory->itNum);
	
	currentState++;
	xmemory->itNum=currentState;

    return 0;
}

/**
 * updatePosition FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param particleVariables_messages  particleVariables_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_particleVariables_message and get_next_particleVariables_message functions.
	*/

__FLAME_GPU_FUNC__ int updatePosition(xmachine_memory_Particle* xmemory, xmachine_message_particleVariables_list* particleVariables_messages){

	float3 agent_position = make_float3(xmemory->x, xmemory->y, xmemory->z);
	float3 agent_accn=make_float3(0.0, 0.0, 0.0);
	xmachine_message_particleVariables* current_message = get_first_particleVariables_message(particleVariables_messages);
	
	//For each other agent...
	while (current_message){
		float3 accn = make_float3(0,0,0);
		float3 currentMessagePosition=make_float3(current_message->x, current_message->y, current_message->z);
		
		float3 positionDifference=currentMessagePosition-agent_position;
		float abs_distance=sqrt(pow(positionDifference.x,2)+pow(positionDifference.y,2)+pow(positionDifference.z,2));

		float3 topHalfEqn=positionDifference*current_message->mass*GRAV_CONST;

		if(abs_distance>MIN_INTERRACTION_RAD){
			float sum=pow(abs_distance, 2)+pow(VELOCITY_DAMP,2);
			float lowerHalfEqn=pow((float)sum, float(1.5));
			accn=topHalfEqn/lowerHalfEqn;
		}

		agent_accn+=accn;
		current_message = get_next_particleVariables_message(current_message, particleVariables_messages);
	}

      
	//use CUDA float types wherever possible
	float4 velsAndPos=make_float4(xmemory->xVel, xmemory->yVel, xmemory->zVel, agent_position.x);

	//calculate new positions
	velsAndPos.w+=(DELTA_T*velsAndPos.x);
	velsAndPos.w+=(0.5*(agent_accn.x)*(DELTA_T*DELTA_T));
	xmemory->x=velsAndPos.w;		

	velsAndPos.w=agent_position.y;
	velsAndPos.w+=(DELTA_T*velsAndPos.y);
	velsAndPos.w+=(0.5*(agent_accn.y)*(DELTA_T*DELTA_T));
	xmemory->y=velsAndPos.w;

	velsAndPos.w=agent_position.z;
	velsAndPos.w+=(DELTA_T*velsAndPos.z);
	velsAndPos.w+=(0.5*(agent_accn.z)*(DELTA_T*DELTA_T));
	xmemory->z=velsAndPos.w;

	//update velocities
	velsAndPos.x+=(agent_accn.x*DELTA_T);
	velsAndPos.y+=(agent_accn.y*DELTA_T);
	velsAndPos.z+=(agent_accn.z*DELTA_T);

	xmemory->xVel=velsAndPos.x;
	xmemory->yVel=velsAndPos.y;
	xmemory->zVel=velsAndPos.z;
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS