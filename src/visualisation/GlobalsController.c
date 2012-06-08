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
 * Globals Controller --- Manages simulation and visualisation variables
 * Called by visualisation loop, functions.c, and main.cu (if console mode)
 * Author: Laurence James
 * Contact: laurie@farragar.com
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <customVisualisation.h>
#include <globalsController.h>

//Defaults
float dt=0.001f; 
float gravConstant=1.0f;
float velocityDamper=0.05f;
float minInteractionRadius=0.0f;
int numPartitions=1;
int itNum=0;

//Set the defaults into FLAME GPU memory 
//Vis defaults are in customVisualisation.h, and don't need to be stored in FLAME memory 
void setSimulationDefaults(){
	
	set_SIMULATION_ITNUM(&itNum);
	set_DELTA_T(&dt);
	set_GRAV_CONST(&gravConstant);
	set_VELOCITY_DAMP(&velocityDamper);
	set_MIN_INTERACTION_RAD(&minInteractionRadius);
	set_NUM_PARTITIONS(&numPartitions);
}

//Called when specifying user simulation vars, or when updating vars mid-simulation
void updateSimulationVars(){

	printf("\nInput dt\n");
	dt=(float)getAndValidateDouble();

	printf("\nInput gravitational constant:\n");
	gravConstant=(float)getAndValidateDouble();

	printf("\nInput velocity dampening factor:\n");
	velocityDamper=(float)getAndValidateDouble();

	printf("\nInput minimum radius of interaction:\n");
	minInteractionRadius=(float)getAndValidateDouble();

	printf("\nInput number of particle groups:\n");
	numPartitions=(int)getAndValidateDouble();

	//FLAME GPU constant functions
	set_DELTA_T(&dt);
	set_GRAV_CONST(&gravConstant);
	set_VELOCITY_DAMP(&velocityDamper);
	set_MIN_INTERACTION_RAD(&minInteractionRadius);
	set_NUM_PARTITIONS(&numPartitions);
}

//Called when specifying user visualisation vars at launch.
void setVisualisationVars(){

	printf("\nInput near clip\n");
	NEAR_CLIP=(float)getAndValidateDouble();
	
	printf("\nInput far clip\n");
	FAR_CLIP=(float)getAndValidateDouble();

	printf("\nInput number of sphere slices\n");
	SPHERE_SLICES=(int)getAndValidateDouble();
	printf("\nInput number of sphere stacks\n");
	SPHERE_STACKS=(int)getAndValidateDouble();

	printf("\nInput sphere radius\n");
	SPHERE_RADIUS=(float)getAndValidateDouble();

	printf("\nInput initial view distance\n");
	VIEW_DISTANCE=(float)getAndValidateDouble();
}

//Increment iteration number and write to FLAME memory
//Called by visualisation loop or main loop (console)
void incrementItNum(){
	itNum++;
	set_SIMULATION_ITNUM(&itNum);
}

//For visualisation, define aspect ratio
void setWindowSize(){

	printf("Enter window width (px):\n");
	WINDOW_WIDTH=(int)getAndValidateDouble();
	printf("Enter window height (px):\n");
	WINDOW_HEIGHT=(int)getAndValidateDouble();
}

//Dump simulation info to console.
void printSimulationInformation(){

	printf("\nIteration number: ");
	printf("%d", itNum);
	printf("\nTimestep size: ");
	printf("%f", dt);
	printf("\nGravitational constant: ");
	printf("%f", gravConstant);
	printf("\nCelocity dampening factor: ");
	printf("%f", velocityDamper);
	printf("\nMinimum radius of interaction: ");
	printf("%f", minInteractionRadius);
	printf("\nNumber of particle groups: ");
	printf("%d", numPartitions);
}

//Validates against the first substring of digits, then clears the buffer
double getAndValidateDouble(){
	int set=0;
	double output;
	char buf;
	while(!set){
		if (scanf("%lf", &output)==0){
			//Clean before next input
			while (getchar() != '\n');   
			printf("\n Invalid input. Please try again\n");
		}
		else {
			set=1;
			while ((buf = getchar()) != '\n' && buf != EOF);
		}
	}
  return output;
}