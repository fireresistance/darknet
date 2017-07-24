#ifndef SOCK_DEMO_H
#define SOCK_DEMO_H
#include "darknet.h"
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include <sys/time.h>

void socketdemo(char *datacfg, char *cfgfile, char *weightfile, 
    double net_thresh, double hier_thresh, char *outfile, 
    int fullscreen, int needed_class, char* sock_name);

void socketdemowithclass(char *datacfg, char *cfgfile, char *weightfile, 
    double net_thresh, double hier_thresh, char *outfile, 
    int fullscreen, int needed_class,char *datacfg_class, 
    char *cfgfile_class, char *weightfile_class, char* sock_name, double class_thresh);

#endif