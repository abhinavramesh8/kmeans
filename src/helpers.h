#ifndef _HELPERS_H
#define _HELPERS_H

#include <cstdlib>
#include "argparse.h"

struct kmeans_args_t {
    int n_clusters;
    int dims;
    int max_iters;
    float threshold;
    bool print_centroids;
    float *centroid_elements;
    int n_points;
    float *point_elements;
    int *labels;
    int iters;
    float iter_time;
    
    ~kmeans_args_t() {
        free(centroid_elements);
        free(point_elements);
        free(labels);
    }
};

void fill_args(kmeans_args_t&, const options_t&);

#endif