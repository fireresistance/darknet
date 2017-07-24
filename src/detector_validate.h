#ifndef DETECTOR_VAL_H
#define DETECTOR_VAL_H

#include "darknet.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

static int get_coco_image_id(char *filename);

static void print_cocos(FILE *fp, char *image_path, box *boxes, float **probs, int num_boxes, int classes, int w, int h);

void print_detector_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h);

void print_imagenet_detections(FILE *fp, int id, box *boxes, float **probs, int total, int classes, int w, int h);

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile);

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile);

void validate_detector_recall(char *cfgfile, char *weightfile);

#endif