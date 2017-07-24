#ifndef DETECTOR_CLEANDIR_WITHCLASS_H
#define DETECTOR_CLEANDIR_WITHCLASS_H

#include "darknet.h"

void detector_clean_dir_class(char *datacfg, char *cfgfile, char *weightfile, char **filenames, int num_files, double net_thresh, double hier_thresh, char *outfile, int fullscreen, int needed_class);

#endif