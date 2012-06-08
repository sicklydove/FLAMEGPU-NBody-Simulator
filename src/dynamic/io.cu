
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


    
void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Particle_list* h_Particles_testingActive, xmachine_memory_Particle_list* d_Particles_testingActive, int h_xmachine_memory_Particle_testingActive_count,xmachine_memory_Particle_list* h_Particles_updatingPosition, xmachine_memory_Particle_list* d_Particles_updatingPosition, int h_xmachine_memory_Particle_updatingPosition_count)
{
	//Device to host memory transfer
	
	CUDA_SAFE_CALL( cudaMemcpy( h_Particles_testingActive, d_Particles_testingActive, sizeof(xmachine_memory_Particle_list), cudaMemcpyDeviceToHost));
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
	for (int i=0; i<h_xmachine_memory_Particle_testingActive_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Particle</name>\n", file);
		
		fputs("<isDark>", file);
		sprintf(data, "%i", h_Particles_testingActive->isDark[i]);
		fputs(data, file);
		fputs("</isDark>\n", file);
		
		fputs("<isActive>", file);
		sprintf(data, "%i", h_Particles_testingActive->isActive[i]);
		fputs(data, file);
		fputs("</isActive>\n", file);
		
		fputs("<particleGroup>", file);
		sprintf(data, "%i", h_Particles_testingActive->particleGroup[i]);
		fputs(data, file);
		fputs("</particleGroup>\n", file);
		
		fputs("<mass>", file);
		sprintf(data, "%f", h_Particles_testingActive->mass[i]);
		fputs(data, file);
		fputs("</mass>\n", file);
		
		fputs("<x>", file);
		sprintf(data, "%f", h_Particles_testingActive->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
		
		fputs("<y>", file);
		sprintf(data, "%f", h_Particles_testingActive->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
		
		fputs("<z>", file);
		sprintf(data, "%f", h_Particles_testingActive->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
		
		fputs("<xVel>", file);
		sprintf(data, "%f", h_Particles_testingActive->xVel[i]);
		fputs(data, file);
		fputs("</xVel>\n", file);
		
		fputs("<yVel>", file);
		sprintf(data, "%f", h_Particles_testingActive->yVel[i]);
		fputs(data, file);
		fputs("</yVel>\n", file);
		
		fputs("<zVel>", file);
		sprintf(data, "%f", h_Particles_testingActive->zVel[i]);
		fputs(data, file);
		fputs("</zVel>\n", file);
		
		fputs("</xagent>\n", file);
	}
	//Write each Particle agent to xml
	for (int i=0; i<h_xmachine_memory_Particle_updatingPosition_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Particle</name>\n", file);
		
		fputs("<isDark>", file);
		sprintf(data, "%i", h_Particles_updatingPosition->isDark[i]);
		fputs(data, file);
		fputs("</isDark>\n", file);
		
		fputs("<isActive>", file);
		sprintf(data, "%i", h_Particles_updatingPosition->isActive[i]);
		fputs(data, file);
		fputs("</isActive>\n", file);
		
		fputs("<particleGroup>", file);
		sprintf(data, "%i", h_Particles_updatingPosition->particleGroup[i]);
		fputs(data, file);
		fputs("</particleGroup>\n", file);
		
		fputs("<mass>", file);
		sprintf(data, "%f", h_Particles_updatingPosition->mass[i]);
		fputs(data, file);
		fputs("</mass>\n", file);
		
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
	int in_Particle_isDark;
	int in_Particle_isActive;
	int in_Particle_particleGroup;
	int in_Particle_mass;
	int in_Particle_x;
	int in_Particle_y;
	int in_Particle_z;
	int in_Particle_xVel;
	int in_Particle_yVel;
	int in_Particle_zVel;

	/* for continuous agents: set agent count to zero */	
	*h_xmachine_memory_Particle_count = 0;
	
	/* Variables for initial state data */
	int Particle_isDark;
	int Particle_isActive;
	int Particle_particleGroup;
	float Particle_mass;
	float Particle_x;
	float Particle_y;
	float Particle_z;
	float Particle_xVel;
	float Particle_yVel;
	float Particle_zVel;
	
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
	in_Particle_isDark = 0;
	in_Particle_isActive = 0;
	in_Particle_particleGroup = 0;
	in_Particle_mass = 0;
	in_Particle_x = 0;
	in_Particle_y = 0;
	in_Particle_z = 0;
	in_Particle_xVel = 0;
	in_Particle_yVel = 0;
	in_Particle_zVel = 0;
	//set all Particle values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Particle_MAX; k++)
	{	
		h_Particles->isDark[k] = 0;
		h_Particles->isActive[k] = 0;
		h_Particles->particleGroup[k] = 0;
		h_Particles->mass[k] = 0;
		h_Particles->x[k] = 0;
		h_Particles->y[k] = 0;
		h_Particles->z[k] = 0;
		h_Particles->xVel[k] = 0;
		h_Particles->yVel[k] = 0;
		h_Particles->zVel[k] = 0;
	}
	

	/* Default variables for memory */
	Particle_isDark = 0;
	Particle_isActive = 0;
	Particle_particleGroup = 0;
	Particle_mass = 0;
	Particle_x = 0;
	Particle_y = 0;
	Particle_z = 0;
	Particle_xVel = 0;
	Particle_yVel = 0;
	Particle_zVel = 0;

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
                    
					h_Particles->isDark[*h_xmachine_memory_Particle_count] = Particle_isDark;
                    
					h_Particles->isActive[*h_xmachine_memory_Particle_count] = Particle_isActive;
                    
					h_Particles->particleGroup[*h_xmachine_memory_Particle_count] = Particle_particleGroup;
                    
					h_Particles->mass[*h_xmachine_memory_Particle_count] = Particle_mass;
                    
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
                    
					(*h_xmachine_memory_Particle_count) ++;
					
					
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
				Particle_isDark = 0;
				Particle_isActive = 0;
				Particle_particleGroup = 0;
				Particle_mass = 0;
				Particle_x = 0;
				Particle_y = 0;
				Particle_z = 0;
				Particle_xVel = 0;
				Particle_yVel = 0;
				Particle_zVel = 0;
			}
			if(strcmp(buffer, "isDark") == 0) in_Particle_isDark = 1;
			if(strcmp(buffer, "/isDark") == 0) in_Particle_isDark = 0;
			if(strcmp(buffer, "isActive") == 0) in_Particle_isActive = 1;
			if(strcmp(buffer, "/isActive") == 0) in_Particle_isActive = 0;
			if(strcmp(buffer, "particleGroup") == 0) in_Particle_particleGroup = 1;
			if(strcmp(buffer, "/particleGroup") == 0) in_Particle_particleGroup = 0;
			if(strcmp(buffer, "mass") == 0) in_Particle_mass = 1;
			if(strcmp(buffer, "/mass") == 0) in_Particle_mass = 0;
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
				if(in_Particle_isDark){ 
					Particle_isDark = (int) atoi(buffer);
				}
				if(in_Particle_isActive){ 
					Particle_isActive = (int) atoi(buffer);
				}
				if(in_Particle_particleGroup){ 
					Particle_particleGroup = (int) atoi(buffer);
				}
				if(in_Particle_mass){ 
					Particle_mass = (float) atof(buffer);
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

