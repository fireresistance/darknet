#include "detector_clean_dir_class.h"
#include <libgen.h>

void detector_clean_dir_class_save_bbox(char *datacfg, char *cfgfile, char *weightfile, char **filenames, int num_files, double net_thresh, double hier_thresh, char *outfile, int fullscreen, int needed_class)
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
    image ret;
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
        char save_buff[256];
    	char *p_savebuff = &save_buff[0];

        image im = load_image_color(input,0,0);
        remove(input);
        input[strlen(input)-4]=0;
        //remove(input);
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
        img_done++;
        int i;
	    int supacnt=0;

	    int cleft=0;
	    int cright=0;
	    int ctop=0;
	    int cbot=0;

	    for(i = 0; i < l.w*l.h*l.n; ++i){
	        int class = max_index(probs[i], l.classes);
	        float prob = probs[i][class];
	        if(class==needed_class)
	        {
		        if(prob > net_thresh)
		        {
		            supacnt++;

		            //printf("%s: %.0f%%\n", names[class], prob*100);

		            box b = boxes[i];

		            int left  = (b.x-b.w/2.)*im.w;
		            int right = (b.x+b.w/2.)*im.w;
		            int top   = (b.y-b.h/2.)*im.h;
		            int bot   = (b.y+b.h/2.)*im.h;

		            if(left < 0) left = 0;
		            if(right > im.w-1) right = im.w-1;
		            if(top < 0) top = 0;
		            if(bot > im.h-1) bot = im.h-1;

		            if(((right-left)*(bot-top))>((cright-cleft)*(cbot-ctop)))
		            {
			            ret = crop_image(im,left,top,right-left,bot-top);
			            sprintf(save_buff,"%s",input);
			            //sprintf(save_buff,"%s_%d",input,supacnt);
			            //printf("SAVE BBOX FOR %s\n", input);
			            //printf("SAVE BBOX TO %s\n", p_savebuff);
	                	save_image(ret, p_savebuff);
	                	free_image(ret);
	                	int cleft=left;
					    int cright=right;
					    int ctop=top;
					    int cbot=bot;        
                	}
		        }
	    	}
	    }
	    fprintf(stderr, "\t\t%s;\tDONE:%d\tREMAINS:%d\n", filename, img_done,num_files-img_done);


	    /*
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
        */

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