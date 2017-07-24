#include "detector_clean_dir_class.h"

void detector_clean_dir_class(char *datacfg, char *cfgfile, char *weightfile, char **filenames, int num_files, double net_thresh, double hier_thresh, char *outfile, int fullscreen, int needed_class)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int deletedclasses=0;
    int img_done=0;

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
    char *filename;
    int k;
    for (k=0;k<num_files;k++)
    {
        filename=filenames[k];
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        get_region_boxes(l, im.w, im.h, net.w, net.h, net_thresh, probs, boxes, 0, 0, hier_thresh, 1);
        if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        int check = bool_detections(l.w*l.h*l.n, net_thresh, probs, l.classes, needed_class);
        img_done++;
        if(check==1)
        {
            fprintf(stderr, "Car Found - ");
            fprintf(stderr, "\t\t%s;\tDONE:%d\tREMAINS:%d\n", filename, img_done,num_files-img_done);
        }
        else
        {
            fprintf(stderr, "Car NOT Found - ");
            fprintf(stderr, "\t%s;\tDONE:%d\tREMAINS:%d\n", filename, img_done,num_files-img_done);
            remove(filename);
            deletedclasses++;
        }

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        if (filename) break;
    }
    }
    fprintf(stderr, "Results:\n");
    fprintf(stderr, "Total images: %d\n",num_files);
    fprintf(stderr, "Deleted: %d\n",deletedclasses);
    fprintf(stderr, "Images with CAR on it: %d\n",num_files-deletedclasses);
}