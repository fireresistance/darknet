#ifndef DETECTOR_TEST_WITHCLASS_H
#define DETECTOR_TEST_WITHCLASS_H

#include "darknet.h"

void detector_test_filelist_with_class(char *datacfg, char *cfgfile, char *weightfile, char **filenames, int num_files, double net_thresh, double hier_thresh, char *outfile, int fullscreen, 
    int needed_class, char *datacfg2, char *cfgfile2, char *weightfile2);

#endif