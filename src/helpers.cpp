#include <cstdlib>
#include <cstring>

#include "helpers.h"
#include "io.h"

static void init_centroids(int seed, kmeans_args_t& args) {
    int n_cent_eles = args.n_clusters * args.dims;
    args.centroid_elements = (float*) malloc(n_cent_eles * sizeof(float));
    
    unsigned long next = static_cast<unsigned long>(seed);
    unsigned long kmeans_rmax = 32767;
    
    for (int cent_ele_idx = 0; cent_ele_idx < n_cent_eles; cent_ele_idx += args.dims) {
        next = next * 1103515245 + 12345;
        auto kmeans_rand = (next / 65536) % (kmeans_rmax + 1);
        auto pt_ele_idx = (kmeans_rand % args.n_points) * args.dims;
        memcpy(args.centroid_elements + cent_ele_idx, args.point_elements + pt_ele_idx, 
               args.dims * sizeof(float));
    }
}

void fill_args(kmeans_args_t& args, const options_t& opts) {
    args.n_clusters = opts.n_clusters;
    args.dims = opts.dims;
    args.max_iters = opts.max_iters;
    args.threshold = opts.threshold;
    args.print_centroids = opts.print_centroids;
    read_file(opts.in_file, args);
    args.labels = (int*) malloc(args.n_points * sizeof(int));
    init_centroids(opts.seed, args);
}
