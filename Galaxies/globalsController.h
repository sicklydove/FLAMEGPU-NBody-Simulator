#ifndef __GLOBALS_CONTROLLER
#define __GLOBALS_CONTROLLER

extern void set_DELTA_T(float* h_DELTA_T);
extern void set_GRAV_CONST(float* h_GRAV_CONST);
extern void set_VELOCITY_DAMP(float* h_VELOCITY_DAMP);
extern void set_MIN_INTERRACTION_RAD(float* h_MIN_INTERRACTION);
extern void set_NUM_PARTITIONS(float* h_NUM_PARTITIONS);

void updateSimulationVars();
void printSimulationInformation(int itNum);
void setSimulationDefaults();
void setVisualisationDefaults();
void setVisualisationVars();
void setWindowSize();

#endif __GLOBALS_CONTROLLER
