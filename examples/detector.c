#include "darknet.h"

void run_detector(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    //Arguments parcer
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    double net_thresh = find_float_arg(argc, argv, "-thresh", .24);
    double class_thresh = find_float_arg(argc, argv, "-classthresh", .25);
    double hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    int num_files = find_int_arg(argc, argv, "-numfiles", 0);
    int needed_class = find_int_arg(argc, argv, "-neededclass", 0);
    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;

    char *datacfg2 = argv[7];
    char *cfg2 = argv[8];
    char *weights2 = (argc > 9) ? argv[9] : 0;

    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    //Args parcer prints
    if(prefix!=0) printf("Prefix - %s\n", prefix);
    if(net_thresh!=0.24) printf("Net thresh - %f\n", net_thresh);
    if(class_thresh!=0.24) printf("Class thresh - %f\n", class_thresh);
    if(hier_thresh!=0.5) printf("Hier Thresh - %f\n", hier_thresh);
    if(cam_index!=0) printf("Cam index - %d\n", cam_index);
    if(frame_skip!=0) printf("Frame skip - %d\n", frame_skip);
    if(avg!=3) printf("Avg - %d\n", avg);
    if(num_files!=0) printf("Num files in filelist - %d\n", num_files);
    if(needed_class!=0) printf("Needed Class - %d\n", needed_class);
    if(clear!=0) printf("Clear - %d\n", clear);
    if(fullscreen!=0) printf("Fullscreen - %d\n", fullscreen);
    if(width!=0) printf("Width - %d\n", width);
    if(height!=0) printf("Height - %d\n", height);
    if(fps!=0) printf("FPS - %d\n", fps);
    if(gpu_list!=0) printf("Gpus - %s\n", gpu_list);
    if(outfile!=0) printf("Outfile - %s\n", outfile);
    
    if(0==strcmp(argv[2], "test")) 
        test_detector(datacfg, cfg, weights, filename, net_thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) 
        train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) 
        validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) 
        validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) 
        validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, net_thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    else if(0==strcmp(argv[2], "demowithclass")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demowithclass(cfg, weights, net_thresh, cam_index, filename, names, 
            classes, frame_skip, prefix, avg, hier_thresh, width, height, 
            fps, fullscreen, datacfg2, cfg2, weights2, class_thresh, needed_class, outfile);
    }
    else if(0==strcmp(argv[2], "cleandir")) {
        char **names = get_directory_paths(filename);
        detector_clean_dir_class(datacfg, cfg, weights, names, num_files, net_thresh, hier_thresh, outfile, fullscreen, needed_class);  
    }
    else if(0==strcmp(argv[2], "cleandirbbox")) {
        char **names = get_directory_paths(filename);
        detector_clean_dir_class_save_bbox(datacfg, cfg, weights, names, num_files, net_thresh, hier_thresh, outfile, fullscreen, needed_class);  
    }
    else if(0==strcmp(argv[2], "testfilelistwithclass")) {
        char **names = get_directory_paths(filename);
        detector_test_filelist_with_class(datacfg, cfg, weights, names, num_files, net_thresh, 
            hier_thresh, outfile, fullscreen, needed_class, 
            datacfg2, cfg2, weights2);  
    }
}
