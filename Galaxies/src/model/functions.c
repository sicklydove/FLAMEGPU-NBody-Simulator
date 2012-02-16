
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


/**
 * outputdata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int outputdata(xmachine_memory_Particle* xmemory, xmachine_message_location_list* location_messages){

	add_location_message(location_messages, xmemory->id, xmemory->mass, xmemory->x, xmemory->y, xmemory->z);
    
    return 0;
}

/**
 * inputdata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_Particle. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.
 */
__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_Particle* xmemory, xmachine_message_location_list* location_messages){
	float dt=0.1;

    //float3 agent_position = make_float3(xmemory->xPos+1, xmemory->yPos, xmemory->zPos);
	float agent_mass = xmemory->mass;

      xmachine_message_location* current_message = get_first_location_message(location_messages);
      while (current_message)
      {
         current_message = get_next_location_message(current_message, location_messages);
	  }

    //float3 dPos = make_float3((xmemory->xVel)*dt, (xmemory->yVel)*dt, (xmemory->zVel)*dt));
    
	xmemory->x =  xmemory->x+(xmemory->xVel)*dt;
    
  
    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
