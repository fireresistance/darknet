#include "detector_test_with_class.h"

void detector_test_filelist_with_class(char *datacfg, char *cfgfile, char *weightfile, char **filenames, int num_files, double net_thresh, double hier_thresh, char *outfile, int fullscreen, 
    int needed_class, char *datacfg2, char *cfgfile2, char *weightfile2)
{  
// NETWORK 1 START ######################################       
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    int deletedclasses=0;
    int counter=0;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
// NETWORK 1 END ######################################## 

// NETWORK 2 START ###################################### 
    network net2 = parse_network_cfg(cfgfile2);
    if(weightfile2){
        load_weights(&net2, weightfile2);
    }
    set_batch_network(&net2, 1);
    srand(2222222);    
    list *options2 = read_data_cfg(datacfg2);
    char *name_list2 = option_find_str(options2, "names", 0);
    if(!name_list2) name_list2 = option_find_str(options2, "labels", "data/labels.list");

    int i = 0;
    char **names2 = get_labels(name_list2);
    int *indexes = calloc(1, sizeof(int));
    int size2 = net2.w;
    image ret;
// NETWORK 2 END ########################################

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
        printf("\n%s\n",filenames[k]);
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

        char savename[256];
        char* p_savename = &savename[0]; 

        for(i = 0; i < l.w*l.h*l.n; ++i)
        {
            int class = max_index(probs[i], l.classes);
            float prob = probs[i][class];
            if (class==needed_class)
            {
                counter=counter+1;
                if(prob > net_thresh){
                    box b = boxes[i];

                    int left  = (b.x-b.w/2.)*im.w;
                    int right = (b.x+b.w/2.)*im.w;
                    int top   = (b.y-b.h/2.)*im.h;
                    int bot   = (b.y+b.h/2.)*im.h;

                    if(left < 0) left = 0;
                    if(right > im.w-1) right = im.w-1;
                    if(top < 0) top = 0;
                    if(bot > im.h-1) bot = im.h-1;
                    ret = crop_image(im,left,top,right-left,bot-top);

                    image r = resize_min(ret, size2);
                    resize_network(&net2, r.w, r.h);

                    float *X2= r.data;
                    time=clock();
                    float *predictions = network_predict(net2, X2);
                    if(net2.hierarchy) hierarchy_predictions(predictions, net2.outputs, net2.hierarchy, 1, 1);
                    top_k(predictions, net2.outputs, 1, indexes);
                    fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));
                    printf("%5.2f%%: %s\n", predictions[indexes[0]]*100, names2[indexes[0]]);
                    sprintf(savename,"carstest/%s-%d",names2[indexes[0]],counter);
                    save_image(ret, p_savename);
                    if(r.data != ret.data) free_image(r);
                    free_image(ret);
                }
            }
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