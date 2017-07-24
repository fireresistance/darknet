#include "darknet.h"

void run_socket(int argc, char **argv)
{
    double net_thresh = find_float_arg(argc, argv, "-thresh", .24);
    double class_thresh = find_float_arg(argc, argv, "-classthresh", .25);
    double hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int needed_class = find_int_arg(argc, argv, "-neededclass", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *sock_name = find_char_arg(argc, argv, "-sockname",0);
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [demo/demowithclass] [data] [cfg] [weights] [optional arguments]\n", argv[0], argv[1]);
        return;
    }

    if(net_thresh!=0.24) printf("Net thresh - %f\n", net_thresh);
    if(class_thresh!=0.24) printf("Class thresh - %f\n", class_thresh);
    if(hier_thresh!=0.5) printf("Hier Thresh - %f\n", hier_thresh);
    if(needed_class!=0) printf("Needed Class - %d\n", needed_class);
    if(fullscreen!=0) printf("Fullscreen - %d\n", fullscreen);
    if(outfile!=0) printf("Gpus - %s\n", outfile);
    if(sock_name!=0) printf("Socket name - %s\n", sock_name);

    if (sock_name == 0)
    {
    	sock_name = "/tmp/demo_sock";
    }

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;

    char *datacfg2 = argv[6];
    char *cfg2 = argv[7];
    char *weights2 = (argc > 8) ? argv[8] : 0;
    if(0==strcmp(argv[2], "demo")) {
        socketdemo(datacfg, cfg, weights, net_thresh, hier_thresh, 
            outfile, fullscreen, needed_class, sock_name);
    }
    else if(0==strcmp(argv[2], "demowithclass")) {
        socketdemowithclass(datacfg, cfg, weights, net_thresh, hier_thresh, 
            outfile, fullscreen, needed_class, datacfg2, cfg2, weights2, 
            sock_name, class_thresh);
    }
}