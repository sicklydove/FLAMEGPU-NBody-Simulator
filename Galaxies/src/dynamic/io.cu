
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
#include <cutil.h>
#include <limits.h>

// include header
#include <header.h>

float3 agent_maximum;
float3 agent_minimum;


    
void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Particle_list* h_Particles_settingActive, xmachine_memory_Particle_list* d_Particles_settingActive, int h_xmachine_memory_Particle_settingActive_count,xmachine_memory_Particle_list* h_Particles_sendingData, xmachine_memory_Particle_list* d_Particles_sendingData, int h_xmachine_memory_Particle_sendingData_count,xmachine_memory_Particle_list* h_Particles_updatingPosition, xmachine_memory_Particle_list* d_Particles_updatingPosition, int h_xmachine_memory_Particle_updatingPosition_count)
{
	//Device to host memory transfer
	
	CUDA_SAFE_CALL( cudaMemcpy( h_Particles_settingActive, d_Particles_settingActive, sizeof(xmachine_memory_Particle_list), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL( cudaMemcpy( h_Particles_sendingData, d_Particles_sendingData, sizeof(xmachine_memory_Particle_list), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL( cudaMemcpy( h_Particles_updatingPosition, d_Particles_updatingPosition, sizeof(xmachine_memory_Particle_list), cudaMemcpyDeviceToHost));
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing itteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("<states>\n<itno>", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("</itno>\n", file);
	fputs("<environment>\n" , file);
	fputs("</environment>\n" , file);

	//Write each Particle agent to xml
	for (int i=0; i<h_xmachine_memory_Particle_settingActive_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Particle</name>\n", file);
		
		fputs("<id>", file);
		sprintf(data, "%i", h_Particles_settingActive->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
		
		fputs("<mass>", file);
		sprintf(data, "%f", h_Particles_settingActive->mass[i]);
		fputs(data, file);
		fputs("</mass>\n", file);
		
		fputs("<isDark>", file);
		sprintf(data, "%i", h_Particles_settingActive->isDark[i]);
		fputs(data, file);
		fputs("</isDark>\n", file);
		
		fputs("<x>", file);
		sprintf(data, "%f", h_Particles_settingActive->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
		
		fputs("<y>", file);
		sprintf(data, "%f", h_Particles_settingActive->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
		
		fputs("<z>", file);
		sprintf(data, "%f", h_Particles_settingActive->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
		
		fputs("<xVel>", file);
		sprintf(data, "%f", h_Particles_settingActive->xVel[i]);
		fputs(data, file);
		fputs("</xVel>\n", file);
		
		fputs("<yVel>", file);
		sprintf(data, "%f", h_Particles_settingActive->yVel[i]);
		fputs(data, file);
		fputs("</yVel>\n", file);
		
		fputs("<zVel>", file);
		sprintf(data, "%f", h_Particles_settingActive->zVel[i]);
		fputs(data, file);
		fputs("</zVel>\n", file);
		
		fputs("<isActive>", file);
		sprintf(data, "%i", h_Particles_settingActive->isActive[i]);
		fputs(data, file);
		fputs("</isActive>\n", file);
		
		fputs("<debug1>", file);
		sprintf(data, "%f", h_Particles_settingActive->debug1[i]);
		fputs(data, file);
		fputs("</debug1>\n", file);
		
		fputs("<debug2>", file);
		sprintf(data, "%f", h_Particles_settingActive->debug2[i]);
		fputs(data, file);
		fputs("</debug2>\n", file);
		
		fputs("<debug3>", file);
		sprintf(data, "%f", h_Particles_settingActive->debug3[i]);
		fputs(data, file);
		fputs("</debug3>\n", file);
		
		fputs("</xagent>\n", file);
	}
	//Write each Particle agent to xml
	for (int i=0; i<h_xmachine_memory_Particle_sendingData_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Particle</name>\n", file);
		
		fputs("<id>", file);
		sprintf(data, "%i", h_Particles_sendingData->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
		
		fputs("<mass>", file);
		sprintf(data, "%f", h_Particles_sendingData->mass[i]);
		fputs(data, file);
		fputs("</mass>\n", file);
		
		fputs("<isDark>", file);
		sprintf(data, "%i", h_Particles_sendingData->isDark[i]);
		fputs(data, file);
		fputs("</isDark>\n", file);
		
		fputs("<x>", file);
		sprintf(data, "%f", h_Particles_sendingData->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
		
		fputs("<y>", file);
		sprintf(data, "%f", h_Particles_sendingData->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
		
		fputs("<z>", file);
		sprintf(data, "%f", h_Particles_sendingData->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
		
		fputs("<xVel>", file);
		sprintf(data, "%f", h_Particles_sendingData->xVel[i]);
		fputs(data, file);
		fputs("</xVel>\n", file);
		
		fputs("<yVel>", file);
		sprintf(data, "%f", h_Particles_sendingData->yVel[i]);
		fputs(data, file);
		fputs("</yVel>\n", file);
		
		fputs("<zVel>", file);
		sprintf(data, "%f", h_Particles_sendingData->zVel[i]);
		fputs(data, file);
		fputs("</zVel>\n", file);
		
		fputs("<isActive>", file);
		sprintf(data, "%i", h_Particles_sendingData->isActive[i]);
		fputs(data, file);
		fputs("</isActive>\n", file);
		
		fputs("<debug1>", file);
		sprintf(data, "%f", h_Particles_sendingData->debug1[i]);
		fputs(data, file);
		fputs("</debug1>\n", file);
		
		fputs("<debug2>", file);
		sprintf(data, "%f", h_Particles_sendingData->debug2[i]);
		fputs(data, file);
		fputs("</debug2>\n", file);
		
		fputs("<debug3>", file);
		sprintf(data, "%f", h_Particles_sendingData->debug3[i]);
		fputs(data, file);
		fputs("</debug3>\n", file);
		
		fputs("</xagent>\n", file);
	}
	//Write each Particle agent to xml
	for (int i=0; i<h_xmachine_memory_Particle_updatingPosition_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Particle</name>\n", file);
		
		fputs("<id>", file);
		sprintf(data, "%i", h_Particles_updatingPosition->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
		
		fputs("<mass>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->mass[i]);
		fputs(data, file);
		fputs("</mass>\n", file);
		
		fputs("<isDark>", file);
		sprintf(data, "%i", h_Particles_updatingPosition->isDark[i]);
		fputs(data, file);
		fputs("</isDark>\n", file);
		
		fputs("<x>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
		
		fputs("<y>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
		
		fputs("<z>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
		
		fputs("<xVel>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->xVel[i]);
		fputs(data, file);
		fputs("</xVel>\n", file);
		
		fputs("<yVel>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->yVel[i]);
		fputs(data, file);
		fputs("</yVel>\n", file);
		
		fputs("<zVel>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->zVel[i]);
		fputs(data, file);
		fputs("</zVel>\n", file);
		
		fputs("<isActive>", file);
		sprintf(data, "%i", h_Particles_updatingPosition->isActive[i]);
		fputs(data, file);
		fputs("</isActive>\n", file);
		
		fputs("<debug1>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->debug1[i]);
		fputs(data, file);
		fputs("</debug1>\n", file);
		
		fputs("<debug2>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->debug2[i]);
		fputs(data, file);
		fputs("</debug2>\n", file);
		
		fputs("<debug3>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->debug3[i]);
		fputs(data, file);
		fputs("</debug3>\n", file);
		
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, xmachine_memory_Particle_list* h_Particles, int* h_xmachine_memory_Particle_count)
{

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_name;
	int in_Particle_id;
	int in_Particle_mass;
	int in_Particle_isDark;
	int in_Particle_x;
	int in_Particle_y;
	int in_Particle_z;
	int in_Particle_xVel;
	int in_Particle_yVel;
	int in_Particle_zVel;
	int in_Particle_isActive;
	int in_Particle_debug1;
	int in_Particle_debug2;
	int in_Particle_debug3;

	/* for continuous agents: set agent count to zero */	
	*h_xmachine_memory_Particle_count = 0;
	
	/* Variables for initial state data */
	int Particle_id;
	float Particle_mass;
	int Particle_isDark;
	float Particle_x;
	float Particle_y;
	float Particle_z;
	float Particle_xVel;
	float Particle_yVel;
	float Particle_zVel;
	int Particle_isActive;
	float Particle_debug1;
	float Particle_debug2;
	float Particle_debug3;
	
	/* Open config file to read-only */
	if((file = fopen(inputpath, "r"))==NULL)
	{
		printf("error opening initial states\n");
		exit(0);
	}
	
	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
	i = 0;
	in_tag = 0;
	in_itno = 0;
	in_name = 0;
	in_Particle_id = 0;
	in_Particle_mass = 0;
	in_Particle_isDark = 0;
	in_Particle_x = 0;
	in_Particle_y = 0;
	in_Particle_z = 0;
	in_Particle_xVel = 0;
	in_Particle_yVel = 0;
	in_Particle_zVel = 0;
	in_Particle_isActive = 0;
	in_Particle_debug1 = 0;
	in_Particle_debug2 = 0;
	in_Particle_debug3 = 0;
	//set all Particle values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Particle_MAX; k++)
	{	
		h_Particles->id[k] = 0;
		h_Particles->mass[k] = 0;
		h_Particles->isDark[k] = 0;
		h_Particles->x[k] = 0;
		h_Particles->y[k] = 0;
		h_Particles->z[k] = 0;
		h_Particles->xVel[k] = 0;
		h_Particles->yVel[k] = 0;
		h_Particles->zVel[k] = 0;
		h_Particles->isActive[k] = 0;
		h_Particles->debug1[k] = 0;
		h_Particles->debug2[k] = 0;
		h_Particles->debug3[k] = 0;
	}
	

	/* Default variables for memory */
	Particle_id = 0;
	Particle_mass = 0;
	Particle_isDark = 0;
	Particle_x = 0;
	Particle_y = 0;
	Particle_z = 0;
	Particle_xVel = 0;
	Particle_yVel = 0;
	Particle_zVel = 0;
	Particle_isActive = 0;
	Particle_debug1 = 0;
	Particle_debug2 = 0;
	Particle_debug3 = 0;

	/* Read file until end of xml */
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);
		
		/* If the end of a tag */
		if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;
			
			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "Particle") == 0)
				{		
					if (*h_xmachine_memory_Particle_count > xmachine_memory_Particle_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Particle exceeded whilst reading data\n", xmachine_memory_Particle_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_Particles->id[*h_xmachine_memory_Particle_count] = Particle_id;
                    
					h_Particles->mass[*h_xmachine_memory_Particle_count] = Particle_mass;
                    
					h_Particles->isDark[*h_xmachine_memory_Particle_count] = Particle_isDark;
                    
					h_Particles->x[*h_xmachine_memory_Particle_count] = Particle_x;
                    //Check maximum x value
                    if(agent_maximum.x < Particle_x)
                        agent_maximum.x = (float)Particle_x;
                    //Check minimum x value
                    if(agent_minimum.x > Particle_x)
                        agent_minimum.x = (float)Particle_x;
                    
					h_Particles->y[*h_xmachine_memory_Particle_count] = Particle_y;
                    //Check maximum y value
                    if(agent_maximum.y < Particle_y)
                        agent_maximum.y = (float)Particle_y;
                    //Check minimum y value
                    if(agent_minimum.y > Particle_y)
                        agent_minimum.y = (float)Particle_y;
                    
					h_Particles->z[*h_xmachine_memory_Particle_count] = Particle_z;
                    //Check maximum z value
                    if(agent_maximum.z < Particle_z)
                        agent_maximum.z = (float)Particle_z;
                    //Check minimum z value
                    if(agent_minimum.z > Particle_z)
                        agent_minimum.z = (float)Particle_z;
                    
					h_Particles->xVel[*h_xmachine_memory_Particle_count] = Particle_xVel;
                    
					h_Particles->yVel[*h_xmachine_memory_Particle_count] = Particle_yVel;
                    
					h_Particles->zVel[*h_xmachine_memory_Particle_count] = Particle_zVel;
                    
					h_Particles->isActive[*h_xmachine_memory_Particle_count] = Particle_isActive;
                    
					h_Particles->debug1[*h_xmachine_memory_Particle_count] = Particle_debug1;
                    
					h_Particles->debug2[*h_xmachine_memory_Particle_count] = Particle_debug2;
                    
					h_Particles->debug3[*h_xmachine_memory_Particle_count] = Particle_debug3;
                    
					(*h_xmachine_memory_Particle_count) ++;
					
					
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
				Particle_id = 0;
				Particle_mass = 0;
				Particle_isDark = 0;
				Particle_x = 0;
				Particle_y = 0;
				Particle_z = 0;
				Particle_xVel = 0;
				Particle_yVel = 0;
				Particle_zVel = 0;
				Particle_isActive = 0;
				Particle_debug1 = 0;
				Particle_debug2 = 0;
				Particle_debug3 = 0;
			}
			if(strcmp(buffer, "id") == 0) in_Particle_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Particle_id = 0;
			if(strcmp(buffer, "mass") == 0) in_Particle_mass = 1;
			if(strcmp(buffer, "/mass") == 0) in_Particle_mass = 0;
			if(strcmp(buffer, "isDark") == 0) in_Particle_isDark = 1;
			if(strcmp(buffer, "/isDark") == 0) in_Particle_isDark = 0;
			if(strcmp(buffer, "x") == 0) in_Particle_x = 1;
			if(strcmp(buffer, "/x") == 0) in_Particle_x = 0;
			if(strcmp(buffer, "y") == 0) in_Particle_y = 1;
			if(strcmp(buffer, "/y") == 0) in_Particle_y = 0;
			if(strcmp(buffer, "z") == 0) in_Particle_z = 1;
			if(strcmp(buffer, "/z") == 0) in_Particle_z = 0;
			if(strcmp(buffer, "xVel") == 0) in_Particle_xVel = 1;
			if(strcmp(buffer, "/xVel") == 0) in_Particle_xVel = 0;
			if(strcmp(buffer, "yVel") == 0) in_Particle_yVel = 1;
			if(strcmp(buffer, "/yVel") == 0) in_Particle_yVel = 0;
			if(strcmp(buffer, "zVel") == 0) in_Particle_zVel = 1;
			if(strcmp(buffer, "/zVel") == 0) in_Particle_zVel = 0;
			if(strcmp(buffer, "isActive") == 0) in_Particle_isActive = 1;
			if(strcmp(buffer, "/isActive") == 0) in_Particle_isActive = 0;
			if(strcmp(buffer, "debug1") == 0) in_Particle_debug1 = 1;
			if(strcmp(buffer, "/debug1") == 0) in_Particle_debug1 = 0;
			if(strcmp(buffer, "debug2") == 0) in_Particle_debug2 = 1;
			if(strcmp(buffer, "/debug2") == 0) in_Particle_debug2 = 0;
			if(strcmp(buffer, "debug3") == 0) in_Particle_debug3 = 1;
			if(strcmp(buffer, "/debug3") == 0) in_Particle_debug3 = 0;
			
			
			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;
			
			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else
			{
				if(in_Particle_id){ 
					Particle_id = (int) atoi(buffer);
				}
				if(in_Particle_mass){ 
					Particle_mass = (float) atof(buffer);
				}
				if(in_Particle_isDark){ 
					Particle_isDark = (int) atoi(buffer);
				}
				if(in_Particle_x){ 
					Particle_x = (float) atof(buffer);
				}
				if(in_Particle_y){ 
					Particle_y = (float) atof(buffer);
				}
				if(in_Particle_z){ 
					Particle_z = (float) atof(buffer);
				}
				if(in_Particle_xVel){ 
					Particle_xVel = (float) atof(buffer);
				}
				if(in_Particle_yVel){ 
					Particle_yVel = (float) atof(buffer);
				}
				if(in_Particle_zVel){ 
					Particle_zVel = (float) atof(buffer);
				}
				if(in_Particle_isActive){ 
					Particle_isActive = (int) atoi(buffer);
				}
				if(in_Particle_debug1){ 
					Particle_debug1 = (float) atof(buffer);
				}
				if(in_Particle_debug2){ 
					Particle_debug2 = (float) atof(buffer);
				}
				if(in_Particle_debug3){ 
					Particle_debug3 = (float) atof(buffer);
				}
				
			}
			
			/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
	/* Close the file */
	fclose(file);
}

float3 getMaximumBounds(){
    return agent_maximum;
}

float3 getMinimumBounds(){
    return agent_minimum;
}

