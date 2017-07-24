#ifndef DETECTOR_TEST_H
#define DETECTOR_TEST_H

#include "darknet.h"

void test_detector(char *datacfg, char *cfgfile, char *weightfile, 
	char *filename, double thresh, float hier_thresh, char *outfile, int fullscreen);

#endif