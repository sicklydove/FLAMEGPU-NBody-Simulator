#ifndef __VISUALISATION_H
#define __VISUALISATION_H

//Consts defined by user at init; defaults

unsigned int WINDOW_WIDTH = 1600;
unsigned int WINDOW_HEIGHT = 900;

//frustrum
double NEAR_CLIP = 0.01;
double FAR_CLIP = 512;

//Circle model fidelity
int SPHERE_SLICES = 10;
int SPHERE_STACKS = 10;

double SPHERE_RADIUS = 0.005;

//Viewing Distance
double VIEW_DISTANCE = 1;

#endif __VISUALISATION_H
