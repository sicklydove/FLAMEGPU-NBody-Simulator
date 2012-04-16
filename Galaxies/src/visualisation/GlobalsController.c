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
#include "customVisualisation.h"
#include "globalsController.h"

float dt=0.001; 
float gravConstant=1;
float velocityDamper=0.1;
float minInteractionRadius=0.0;
int numPartitions=1;

void updateSimulationVars(){

	printf("\nInput dt\n");
	scanf("%f", &dt);
	printf("\nInput gravitational constant:\n");
	scanf("%f", &gravConstant);
	printf("\nInput velocity dampening factor:\n");
	scanf("%f", &velocityDamper);
	printf("\nInput minimum radius of interraction:\n");
	scanf("%f", &minInteractionRadius);
	printf("\nInput number of timestep slices (optimisation):\n");
	scanf("%d", &numPartitions);
	
	set_DELTA_T(&dt);
	set_GRAV_CONST(&gravConstant);
	set_VELOCITY_DAMP(&velocityDamper);
	set_MIN_INTERRACTION_RAD(&minInteractionRadius);
	set_NUM_PARTITIONS(&numPartitions);
}

void setVisualisationVars(){

	printf("\nInput near clip\n");
	scanf("%lf", &NEAR_CLIP);
	printf("\nInput far clip\n");
	scanf("%lf", &FAR_CLIP);

	printf("\nInput number of sphere slices\n");
	scanf("%d", &SPHERE_SLICES);
	printf("\nInput number of sphere stacks\n");
	scanf("%d", &SPHERE_STACKS);
	
	printf("\nInput sphere radius\n");
	scanf("%lf", &SPHERE_RADIUS );

	printf("\nInput initial view distance\n");
	scanf("%lf", &VIEW_DISTANCE);
}

void setSimulationDefaults(){

	set_DELTA_T(&dt);
	set_GRAV_CONST(&gravConstant);
	set_VELOCITY_DAMP(&velocityDamper);
	set_MIN_INTERRACTION_RAD(&minInteractionRadius);
	set_NUM_PARTITIONS(&numPartitions);
}

void setWindowSize(){

	printf("Enter window width (px):\n");
	scanf("%d", &WINDOW_WIDTH);
	printf("Enter window height (px):\n");
	scanf("%d", &WINDOW_HEIGHT);
}

void printSimulationInformation(int itNum){

	printf("\nIteration number: ");
	printf("%d", itNum);
	printf("\nTimestep size: ");
	printf("%f", dt);
	printf("\nGravitational constant: ");
	printf("%f", gravConstant);
	printf("\nCelocity dampening factor: ");
	printf("%f", velocityDamper);
	printf("\nMinimum radius of interraction: ");
	printf("%f", minInteractionRadius);
	printf("\nNumber of timestep slices: ");
	printf("%d", numPartitions);
}