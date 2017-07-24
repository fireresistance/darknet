#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo_with_class.h"
#include <sys/time.h>

#define DEMO 1
#ifdef OPENCV
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

//NETWORK2 ##################################
static network net_class;
static list *options_class;
static char *name_list_class;
static char **names_class;
static int *indexes_class;
static int size_class=0;
static int needed_class=2;
static int top_class = 0;
//NETWORK2 END ############################

static float **probs;
static box *boxes;
static network net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static double demo_thresh = 0;
static double demo_class_thresh = 0.25;
static float demo_hier = .5;
static int running = 0;

static int demo_delay = 0;
static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static float *last_avg2;
static float *last_avg;
static float *avg;
double demo_time;

void *detect_in_thread_class(void *ptr)
{
    running = 1;
    float nms = .4;
    layer l = net.layers[net.n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    float *prediction = network_predict(net, X);
    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = last_avg2;
    if(demo_delay == 0) l.output = avg;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, buff[0].w, buff[0].h, net.w, 
            net.h, demo_thresh, probs, boxes, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];
    top_class = 1;

    int i;
    int supacnt=0;
    for(i = 0; i < demo_detections; ++i){
        int class = max_index(probs[i], demo_classes);
        float prob = probs[i][class];
        if(prob > .24){
            supacnt++;
            if(class==needed_class){

                int width = display.h * .006;

                if(0){
                    width = pow(prob, 1./2.)*10+1;
                    demo_alphabet = 0;
                }

                int offset = class*123457 % demo_classes;
                float red = get_color(2,offset,demo_classes);
                float green = get_color(1,offset,demo_classes);
                float blue = get_color(0,offset,demo_classes);
                float rgb[3];

                rgb[0] = red;
                rgb[1] = green;
                rgb[2] = blue;
                box b = boxes[i];

                int left  = (b.x-b.w/2.)*display.w;
                int right = (b.x+b.w/2.)*display.w;
                int top   = (b.y-b.h/2.)*display.h;
                int bot   = (b.y+b.h/2.)*display.h;

                if(left < 0) left = 0;
                if(right > display.w-1) right = display.w-1;
                if(top < 0) top = 0;
                if(bot > display.h-1) bot = display.h-1;


                image ret = crop_image(display,left,top,right-left,bot-top);
                image r = resize_min(ret, size_class);
                resize_network(&net_class, r.w, r.h);

                float *x_class = r.data;
                float *predictions_class = network_predict(net_class, x_class);
                
                if(net_class.hierarchy) hierarchy_predictions(predictions_class, 
                    net_class.outputs, net_class.hierarchy, 1, 1);
                top_k(predictions_class, net_class.outputs, 
                    top_class, indexes_class);
                
                if(predictions_class[indexes_class[0]]+
                    predictions_class[indexes_class[1]]+
                    predictions_class[indexes_class[2]]>demo_class_thresh)
                {
                    printf("%5.2f%%: %s\n", predictions_class[indexes_class[0]]*100, names_class[indexes_class[0]]);
	                draw_box_width(display, left, top, right, bot, width, red, green, blue);
	                if (demo_alphabet) {
	                    image label = get_label(demo_alphabet, names_class[indexes_class[0]], (display.h*.03)/10);
	                    draw_label(display, top + width, left, label, rgb);
	                    free_image(label);
	                }
                }
                if(r.data != ret.data) free_image(r);
                free_image(ret);
            }
        }
    }
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread_class(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net.w, net.h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread_class(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 10){
        if(demo_delay == 0) demo_delay = 60;
        else if(demo_delay == 5) demo_delay = 0;
        else if(demo_delay == 60) demo_delay = 5;
        else demo_delay = 0;
    } else if (c == 27) {
        demo_done = 1;
        return 0;
    } 
    else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } 
    else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void demowithclass(char *cfgfile, char *weightfile, float thresh, 
    int cam_index, const char *filename, char **names, int classes, 
    int delay, char *prefix, int avg_frames, double hier, int w, int h, 
    int frames, int fullscreen, char *datacfg_class, char *cfgfile_class, 
    char *weightfile_class, double class_thresh)
{
    demo_delay = delay;
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_class_thresh = class_thresh;

    demo_hier = hier;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

//NETWORK 2###############################
    net_class = parse_network_cfg(cfgfile_class);
    if(weightfile_class){
        load_weights(&net_class, weightfile_class);
    }
    set_batch_network(&net_class, 1);
    srand(2222222);

    options_class = read_data_cfg(datacfg_class);

    name_list_class = option_find_str(options_class, "names", 0);
    if(!name_list_class) name_list_class = option_find_str(options_class, "labels", "data/labels.list");
    if(top_class == 0) top_class = option_find_int(options_class, "top", 1);

    names_class = get_labels(name_list_class);
    indexes_class = calloc(top_class, sizeof(int));
    size_class = net_class.w;
//NETWORK2 ################################

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    last_avg  = (float *) calloc(l.outputs, sizeof(float));
    last_avg2 = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[1] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[2] = letterbox_image(buff[0], net.w, net.h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = get_wall_time();

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread_class, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread_class, 0)) error("Thread creation failed");
        if(!prefix){
            if(count % (demo_delay+1) == 0){
                fps = 1./(get_wall_time() - demo_time);
                demo_time = get_wall_time();
                float *swap = last_avg;
                last_avg  = last_avg2;
                last_avg2 = swap;
                memcpy(last_avg, avg, l.outputs*sizeof(float));
            }
            display_in_thread_class(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
}
#else
void demowithclass(char *cfgfile, char *weightfile, float thresh, 
    int cam_index, const char *filename, char **names, int classes, 
    int delay, char *prefix, int avg_frames, float hier, int w, int h, 
    int frames, int fullscreen,
    char *datacfg_class, char *cfgfile_class, char *weightfile_class)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif