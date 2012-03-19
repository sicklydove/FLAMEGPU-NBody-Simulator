#ifndef __VISUALISATION_H
#define __VISUALISATION_H

// constants
//const int SIMULATION_DELAY = 5000;
const unsigned int WINDOW_WIDTH = 1600;
const unsigned int WINDOW_HEIGHT = 900;


//frustrum
const double NEAR_CLIP = 0.01;
const double FAR_CLIP = 640;

//Circle model fidelity
const int SPHERE_SLICES = 10;
const int SPHERE_STACKS = 10;
const double SPHERE_RADIUS = 0.0035;

//Viewing Distance
const double VIEW_DISTANCE = 1;

//light position
GLfloat LIGHT_POSITION[] = {1.0f, 1.0f, 1.0f, 1.0f};

#endif __VISUALISATION_H