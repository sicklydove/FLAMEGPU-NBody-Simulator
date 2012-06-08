#ifndef __GLOBALS_CONTROLLER
#define __GLOBALS_CONTROLLER

extern void set_DELTA_T(float* h_DELTA_T);
extern void set_GRAV_CONST(float* h_GRAV_CONST);
extern void set_VELOCITY_DAMP(float* h_VELOCITY_DAMP);
extern void set_MIN_INTERACTION_RAD(float* h_MIN_INTERRACTION);
extern void set_NUM_PARTITIONS(int* h_NUM_PARTITIONS);
extern void set_SIMULATION_ITNUM(int* h_ITNUM);

void updateSimulationVars();
void printSimulationInformation();
void setSimulationDefaults();
void setVisualisationDefaults();
void setVisualisationVars();
void setWindowSize();
void incrementItNum();
double getAndValidateDouble();

#endif __GLOBALS_CONTROLLER