#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int i, c, h, w;
    int fr, fc;
    int outr, outc;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);
    float val;

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (i = 0; i < in.rows; i++) {
        // get 1 image
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        
        // loop over original image by channel, rows, then collumn
        for (c = 0; c < example.c; c++) {
            for (h = 0, outr = 0; h < example.h; h += l.stride, outr++) {
                for (w = 0, outc = 0; w < example.w; w += l.stride, outc++) {
                    float max = FLT_MIN;
                    
                    // loop over the maxpool filter
                    for (fr = 0; fr < l.size; fr++) {
                        for (fc = 0; fc < l.size; fc++) {
                            if (w + fc >= 0 && h + fr >= 0 && w + fc < example.w && h + fr < example.h) {
                                val = get_pixel(example, w + fc, h + fr, c);
                                if (val > max) { max = val;}
                            }
                        }
                    }

                    set_matrix(out, i, (outh * outw * c) + (outw * outr) + outc, max);
                }
            }
        }
    }


    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    int i, c, h, w;
    int fr, fc;
    int outr, outc;

    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    for (i = 0; i < in.rows; i++) {
        // get 1 image
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        
        // loop over original image by channel, rows, then collumn
        for (c = 0; c < example.c; c++) {
            for (h = 0, outr = 0; h < example.h; h += l.stride, outr++) {
                for (w = 0, outc = 0; w < example.w; w += l.stride, outc++) {
                    float max = FLT_MIN;
                    int idxr = 0;
                    int idxc = 0;
                    float val;
                    
                    // loop over the maxpool filter
                    for (fr = 0; fr < l.size; fr++) {
                        for (fc = 0; fc < l.size; fc++) {
                            if (w + fc >= 0 && h + fr >= 0 && w + fc < example.w && h + fr < example.h) {
                                val = get_pixel(example, w + fc, h + fr, c);
                                if (val > max) {
                                    max = val;
                                    idxr = fr;
                                    idxc = fc;
                                }
                            }
                        }
                    }

                    val = get_matrix(dy, i, (outh * outw * c) + (outw * outr) + outc);
                    set_matrix(dx, i, (l.height * l.width * c) + (l.width * (h + idxr)) + (w + idxc), val);
                }
            }
        }
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

