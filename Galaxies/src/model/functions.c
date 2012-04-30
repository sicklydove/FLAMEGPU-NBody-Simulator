/*
 * Author: Laurence James
 * Contact: ljames1@sheffield.ac.uk or laurence.james@gmail.com
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>
#include <stdio.h>
#include <GlobalsController.h>

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

__FLAME_GPU_FUNC__ int broadcastVariables(xmachine_memory_Particle* xMemory, xmachine_message_particleVariables_list* particleVariables_messages){
	add_particleVariables_message(particleVariables_messages, xMemory->mass, xMemory->x, xMemory->y, xMemory->z);
    return 0;
}

__FLAME_GPU_FUNC__ int setIsActive(xmachine_memory_Particle* xMemory){
	int particleGroup=xMemory->particleGroup;
	
	xMemory->isActive=1;

	//Check if brute force to perform as few floating point operations as possible.
	if(NUM_PARTITIONS!=1){
		if((SIMULATION_ITNUM+particleGroup)%NUM_PARTITIONS!=0)
			xMemory->isActive=0;
	}

    return 0;
}

__FLAME_GPU_FUNC__ int updatePosition(xmachine_memory_Particle* xMemory, xmachine_message_particleVariables_list* particleVariables_messages){

	//THIS agent's vars
	float3 agent_position = make_float3(xMemory->x, xMemory->y, xMemory->z);
	float3 agent_accn=make_float3(0.0, 0.0, 0.0);
	
	xmachine_message_particleVariables* current_message = get_first_particleVariables_message(particleVariables_messages);
	
	//For each other agent...
	while (current_message){
		//Current message vars
		float3 accn = make_float3(0,0,0);
		float3 currentMessagePosition=make_float3(current_message->x, current_message->y, current_message->z);
		
		float3 positionDifference=currentMessagePosition-agent_position;
		float abs_distance=sqrt(pow(positionDifference.x,2)+pow(positionDifference.y,2)+pow(positionDifference.z,2));
		
		float3 topHalfEqn=positionDifference*current_message->mass;

		if(abs_distance>MIN_INTERACTION_RAD){
			float sum=pow(abs_distance, 2)+pow(VELOCITY_DAMP,2);
			float lowerHalfEqn=pow((float)sum, float(1.5));
			accn=topHalfEqn/lowerHalfEqn;
		}
		
		agent_accn+=accn;
		current_message = get_next_particleVariables_message(current_message, particleVariables_messages);
	}

	//Multiply by grav constant outside of loop to reduce #FLOPS
	agent_accn=agent_accn*GRAV_CONST;

	//use CUDA float types wherever possible
	float4 velsAndPos=make_float4(xMemory->xVel, xMemory->yVel, xMemory->zVel, agent_position.x);

	//calculate new positions, .w used for calculating new position to reduce memory req
	velsAndPos.w+=(DELTA_T*velsAndPos.x);
	velsAndPos.w+=(0.5*(agent_accn.x)*(DELTA_T*DELTA_T));
	xMemory->x=velsAndPos.w;		

	velsAndPos.w=agent_position.y;
	velsAndPos.w+=(DELTA_T*velsAndPos.y);
	velsAndPos.w+=(0.5*(agent_accn.y)*(DELTA_T*DELTA_T));
	xMemory->y=velsAndPos.w;

	velsAndPos.w=agent_position.z;
	velsAndPos.w+=(DELTA_T*velsAndPos.z);
	velsAndPos.w+=(0.5*(agent_accn.z)*(DELTA_T*DELTA_T));
	xMemory->z=velsAndPos.w;

	//update velocities
	velsAndPos.x+=(agent_accn.x*DELTA_T);
	velsAndPos.y+=(agent_accn.y*DELTA_T);
	velsAndPos.z+=(agent_accn.z*DELTA_T);

	xMemory->xVel=velsAndPos.x;
	xMemory->yVel=velsAndPos.y;
	xMemory->zVel=velsAndPos.z;
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
