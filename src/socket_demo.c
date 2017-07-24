#include "socket_demo.h"
#include "darknet.h"

static list *options;
static list *options_class;

void socketdemo(char *datacfg, char *cfgfile, char *weightfile, 
    double net_thresh, double hier_thresh, char *outfile, 
    int fullscreen, int needed_class, char* sock_name)
{
    struct sockaddr_un address;
    int socket_fd, connection_fd;
    int recv_num;
    socklen_t address_length;
    int n;

    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd < 0){
        printf("socket() failed\n");
        return 1;
    }

    unlink(sock_name);
    memset(&address, 0, sizeof(struct sockaddr_un));

    address.sun_family = AF_UNIX;
    snprintf(address.sun_path, 100, sock_name);

    if (bind(socket_fd, (struct sockaddr *) &address, sizeof(struct sockaddr_un)) != 0) {
        printf("bind() failed\n");
        return 1;
    }
    if(listen(socket_fd, 5) != 0) {
        printf("listen() failed\n");
        return 1;
    }
    address_length = sizeof(address);    

    options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

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

    while((connection_fd = accept(socket_fd,(struct sockaddr *) &address,&address_length)) > -1)
    {
        while(1)
        {
            char buffer[256];
            bzero(buffer,256);

            write(connection_fd, "RDY", 3);

            n = read(connection_fd,&recv_num,sizeof(int));
            if (n < 0)
            {
                printf("ERROR writing to socket\n");
                break;
            }
            n = read(connection_fd,&buffer,recv_num);
            input = &buffer;
            printf("Input image: %s\n", input);
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
            
            int i;
            int boxes_counter=0;
            intbox *boxes_ret=calloc(1, sizeof(box));
            char ** classes_ret=(char**)malloc(sizeof(char*));

            for(i = 0; i < l.w*l.h*l.n; ++i){
                int class = max_index(probs[i], l.classes);
                float prob = probs[i][class];
                if(prob > net_thresh){
                    //if(class==needed_class)
                    //{

                    printf("%s: %.0f%%\n", names[class], prob*100);
                    box b = boxes[i];

                    int left  = (b.x-b.w/2.)*im.w;
                    int right = (b.x+b.w/2.)*im.w;
                    int top   = (b.y-b.h/2.)*im.h;
                    int bot   = (b.y+b.h/2.)*im.h;

                    if(left < 0) left = 0;
                    if(right > im.w-1) right = im.w-1;
                    if(top < 0) top = 0;
                    if(bot > im.h-1) bot = im.h-1;

                    intbox ib;
                    ib.x=left;
                    ib.y=top;
                    ib.w=right-left;
                    ib.h=bot-top;

                    boxes_ret[boxes_counter]=ib;
                    classes_ret[boxes_counter]=names[class];
                    boxes_counter++;
                    boxes_ret=realloc(boxes_ret,(sizeof(intbox))*(boxes_counter+1));
                    classes_ret=realloc(classes_ret,(sizeof(char*))*(boxes_counter+1));
                //}
                }
            }

            int si=0;
            int len_class=0;
            printf("Boxes Counter: %d\n", boxes_counter);
            write(connection_fd, &boxes_counter, 4);
            for(si=0;si<boxes_counter;si++)
            {
                write(connection_fd,(void*)&boxes_ret[si].x,4);
                write(connection_fd,(void*)&boxes_ret[si].y,4);
                write(connection_fd,(void*)&boxes_ret[si].w,4);
                write(connection_fd,(void*)&boxes_ret[si].h,4);
                len_class=strlen(classes_ret[si]);
                write(connection_fd,(void*)&len_class,4);
                write(connection_fd,classes_ret[si],len_class);
                printf("Box [%d] - [x,y,w,h] - %d,%d,%d,%d\n", si,boxes_ret[si].x,boxes_ret[si].y,boxes_ret[si].w,boxes_ret[si].h);
                printf("Class Name Length: %d\n", len_class);
                printf("Class Name: %s\n\n", classes_ret[si]);
            }

            free_image(im);
            free_image(sized);
            free(boxes);
            free_ptrs((void **)probs, l.w*l.h*l.n);


            if (n < 0)
            {
                printf("ERROR writing to socket\n");
                break;
            }
        }
    }
    close(socket_fd);
    close(socket_fd);
    return;
}

void socketdemowithclass(char *datacfg, char *cfgfile, char *weightfile, 
    double net_thresh, double hier_thresh, char *outfile, 
    int fullscreen, int needed_class, char *datacfg_class, 
    char *cfgfile_class, char *weightfile_class, char* sock_name, double class_thresh)
{
    struct sockaddr_un address;
    int socket_fd, connection_fd;
    int recv_num;
    socklen_t address_length;
    int n;
    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd < 0){
        printf("socket() failed\n");
        return;
    }
    unlink(sock_name);
    memset(&address, 0, sizeof(struct sockaddr_un));
    address.sun_family = AF_UNIX;
    snprintf(address.sun_path, 100, sock_name);
    if (bind(socket_fd, (struct sockaddr *) &address, sizeof(struct sockaddr_un)) != 0) {
        printf("bind() failed\n");
        return;
    }
    if(listen(socket_fd, 5) != 0) {
        printf("listen() failed\n");
        return;
    }
    address_length = sizeof(address);

    //NETWORK 1 #####################################
    options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

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
    //NETWORK 1 END #################################

    //NETWORK 2###############################
    network net_class = parse_network_cfg(cfgfile_class);
    if(weightfile_class){
        load_weights(&net_class, weightfile_class);
    }
    set_batch_network(&net_class, 1);
    srand(2222222);
    options_class = read_data_cfg(datacfg_class);

    //Always 1 Due to label contains only top class
    int top_class=1;
    char *name_list_class = option_find_str(options_class, "names", 0);
    if(!name_list_class) name_list_class = option_find_str(options_class, "labels", "data/labels.list");
    if(top_class == 0) top_class = option_find_int(options_class, "top", 1);

    char **names_class = get_labels(name_list_class);
    int *indexes_class = calloc(top_class, sizeof(int));
    int size_class = net_class.w;
    //NETWORK2 ################################

    while((connection_fd = accept(socket_fd,(struct sockaddr *) &address,&address_length)) > -1)
    {
        while(1)
        {
            char buffer[256];
            bzero(buffer,256);
            write(connection_fd, "RDY", 3);
            n = read(connection_fd,&recv_num,sizeof(int));
            if (n < 0)
            {
                printf("ERROR writing to socket\n");
                break;
            }
            n = read(connection_fd,&buffer,recv_num);
            input = &buffer;
            printf("Input image: %s\n", input);
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
            
            int i;
            int boxes_counter=0;
            intbox *boxes_ret=calloc(1, sizeof(box));
            char ** classes_ret=(char**)malloc(sizeof(char*));

            for(i = 0; i < l.w*l.h*l.n; ++i){
                int class = max_index(probs[i], l.classes);
                float prob = probs[i][class];
                //printf("preclass-%d\n", class );
                if(prob > net_thresh){
                    if(class==needed_class)
                    {
                        printf("%s: %.0f%%\n", names[class], prob*100);
                        box b = boxes[i];

                        int left  = (b.x-b.w/2.)*im.w;
                        int right = (b.x+b.w/2.)*im.w;
                        int top   = (b.y-b.h/2.)*im.h;
                        int bot   = (b.y+b.h/2.)*im.h;

                        if(left < 0) left = 0;
                        if(right > im.w-1) right = im.w-1;
                        if(top < 0) top = 0;
                        if(bot > im.h-1) bot = im.h-1;

                        intbox ib;
                        ib.x=left;
                        ib.y=top;
                        ib.w=right-left;
                        ib.h=bot-top;

                        boxes_ret[boxes_counter]=ib;
                        printf("Predicted RETBOXES - %d,%d,%d,%d\n", boxes_ret[boxes_counter].x,boxes_ret[boxes_counter].y,boxes_ret[boxes_counter].w,boxes_ret[boxes_counter].h);

                        //NETWORK CLASS
                        image ret = crop_image(im,left,top,right-left,bot-top);
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
                            predictions_class[indexes_class[2]]>class_thresh)
                        {
                            printf("%5.2f%%: %s\n", predictions_class[indexes_class[0]]*100, names_class[indexes_class[0]]);
                            classes_ret[boxes_counter]=names_class[indexes_class[0]];
	                        boxes_counter++;
	                        boxes_ret=realloc(boxes_ret,(sizeof(intbox))*(boxes_counter+1));
	                        classes_ret=realloc(classes_ret,(sizeof(char*))*(boxes_counter+1));
                        }
                        if(r.data != ret.data) free_image(r);
                        free_image(ret);
                        //NETWORK CLASS END
                    }
                }
            }

            int si=0;
            int len_class=0;
            printf("Boxes Counter: %d\n", boxes_counter);
            write(connection_fd, &boxes_counter, 4);
            for(si=0;si<boxes_counter;si++)
            {
                write(connection_fd,(void*)&boxes_ret[si].x,4);
                write(connection_fd,(void*)&boxes_ret[si].y,4);
                write(connection_fd,(void*)&boxes_ret[si].w,4);
                write(connection_fd,(void*)&boxes_ret[si].h,4);
                len_class=strlen(classes_ret[si]);
                write(connection_fd,(void*)&len_class,4);
                write(connection_fd,classes_ret[si],len_class);
                printf("Box [%d] - [x,y,w,h] - %d,%d,%d,%d\n", si,boxes_ret[si].x,boxes_ret[si].y,boxes_ret[si].w,boxes_ret[si].h);
                printf("Class Name Length: %d\n", len_class);
                printf("Class Name: %s\n\n", classes_ret[si]);
            }

            free_image(im);
            free_image(sized);
            free(boxes);
            free_ptrs((void **)probs, l.w*l.h*l.n);

            if (n < 0)
            {
                printf("ERROR writing to socket\n");
                break;
            }
        }
    }
    close(socket_fd);
    close(socket_fd);
    return;
}