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
 *
 *
 * functions.c - Particle agent interaction function scripts for Grav' NBody simulation
 * Author: Laurence James
 * Contact: laurie@farragar.com
 */
#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>
#include <stdio.h>
#include <globalsController.h>

//Take user input to define initial simulation variables
__FLAME_GPU_INIT_FUNC__ void initConstants(){
	
	char input;
	bool set=false;

	//Get vis settings in customVisualisation.cu so console mode users don't have to set them
	printf("Use default simulation parameters y/n \n");
	while(!set){
		input=getchar();
		switch(input){
			case 'y': case 'Y':
				setSimulationDefaults();
				set=true;
				break;
			case 'n': case 'N':
				//flush input buffer to stop users stepping over
				while ((input = getchar()) != '\n' && input != EOF);
				updateSimulationVars();
				set=true;
				break;
			//And again, to handle side cases...
			case '\n':
				break;
			default:
				printf("Invalid input. Use default simulation parameters? y/n \n");
				//flush any extras
				while ((input = getchar()) != '\n' && input != EOF);
				break;
		}
	}
}

/**
 * broadcastVariables FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param particleVariables_messages Pointer to output message list of type xmachine_message_particleVariables_list. Must be passed as an argument to the add_particleVariables_message function ??.
 */
__FLAME_GPU_FUNC__ int broadcastVariables(xmachine_memory_Particle* agent, xmachine_message_particleVariables_list* particleVariables_messages){
	add_particleVariables_message(particleVariables_messages, agent->mass, agent->x, agent->y, agent->z);
    return 0;
}

/**
 * setIsActive FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 */
__FLAME_GPU_FUNC__ int setIsActive(xmachine_memory_Particle* agent){
	int particleGroup=agent->particleGroup;
	
	agent->isActive=1;

	//Check if brute force now to save potentially N(FLOPS)
	if(NUM_PARTITIONS!=1){
		if((SIMULATION_ITNUM+particleGroup)%NUM_PARTITIONS!=0)
			agent->isActive=0;
	}

    return 0;
}

/**
 * updatePosition FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param particleVariables_messages  particleVariables_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_particleVariables_message and get_next_particleVariables_message functions.
 */
__FLAME_GPU_FUNC__ int updatePosition(xmachine_memory_Particle* agent, xmachine_message_particleVariables_list* particleVariables_messages){

	float3 agent_position = make_float3(agent->x, agent->y, agent->z);
	float3 agent_accn=make_float3(0.0, 0.0, 0.0);
	
	//Do this OUTSIDE O(N^2) particle loop
	float damperSq=pow(VELOCITY_DAMP,2);
	
	xmachine_message_particleVariables* current_message = get_first_particleVariables_message(particleVariables_messages);
	while (current_message){
		float3 accn = make_float3(0,0,0);
		float3 currentMessagePosition=make_float3(current_message->x, current_message->y, current_message->z);
		
		float3 positionDifference=currentMessagePosition-agent_position;
		float abs_distance=sqrt(pow(positionDifference.x,2)+pow(positionDifference.y,2)+pow(positionDifference.z,2));
		
		float3 eqnTop=positionDifference*current_message->mass;

		if(abs_distance>MIN_INTERACTION_RAD){
			float sum=pow(abs_distance, 2)+damperSq;
			float eqnBtm=pow((float)sum, float(1.5));
			accn=eqnTop/eqnBtm;
		}
		
		agent_accn+=accn;
		current_message = get_next_particleVariables_message(current_message, particleVariables_messages);
	}

	//Update this agent for this step
	agent_accn=agent_accn*GRAV_CONST;

	//Current values
	float4 velsAndPos=make_float4(agent->xVel, agent->yVel, agent->zVel, 0.0f);
	float halfDtSq=(DELTA_T*DELTA_T)*0.5;

	//new positions
	velsAndPos.w+=(DELTA_T*velsAndPos.x);
	velsAndPos.w+=(agent_accn.x*halfDtSq);
	agent->x=velsAndPos.w;

	velsAndPos.w=agent_position.y;
	velsAndPos.w+=(DELTA_T*velsAndPos.y);
	velsAndPos.w+=(agent_accn.y*halfDtSq);
	agent->y=velsAndPos.w;

	velsAndPos.w=agent_position.z;
	velsAndPos.w+=(DELTA_T*velsAndPos.z);
	velsAndPos.w+=(agent_accn.z*halfDtSq);
	agent->z=velsAndPos.w;

	//update velocities
	velsAndPos.x+=(agent_accn.x*DELTA_T);
	velsAndPos.y+=(agent_accn.y*DELTA_T);
	velsAndPos.z+=(agent_accn.z*DELTA_T);

	agent->xVel=velsAndPos.x;
	agent->yVel=velsAndPos.y;
	agent->zVel=velsAndPos.z;
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS