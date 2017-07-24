#ifndef DETECTOR_TRAIN_H
#define DETECTOR_TRAIN_H

#include "darknet.h"

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear);

#endif