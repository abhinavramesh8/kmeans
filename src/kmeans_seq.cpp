#include <limits>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

#include "helpers.h"
#include "kmeans_seq.h"

using namespace std;

static float distance(float *x, float *y, int dims) {
    float sq_dist = 0.0f;
    for (int i = 0; i < dims; i++) {
        float diff = x[i] - y[i];
        sq_dist += (diff * diff);
    }
    return sqrtf(sq_dist);
}

static void add_vect(float *to_vect, float *from_vect, int dims) {
    for (int i = 0; i < dims; ++i) {
        to_vect[i] += from_vect[i];
    }
}

static void div_by_int(float *x, int val, int dims) {
    if (val != 0) {
        for (int i = 0; i < dims; ++i) {
            x[i] /= val;
        }
    }
}

static bool converged(const kmeans_args_t& args, float *old_centroid_elements) {
    static int n_cent_eles = args.n_clusters * args.dims;
    for (int i = 0; i < n_cent_eles; i += args.dims) {
        auto dist = distance(args.centroid_elements + i, 
                             old_centroid_elements + i, args.dims);
        if (dist > args.threshold) {
            return false;
        }
    }
    return true;
}

static void assign_centroids(kmeans_args_t& args) {
    static int n_pt_eles = args.n_points * args.dims;
    static int n_cent_eles = args.n_clusters * args.dims;
    for (int i = 0; i < n_pt_eles; i += args.dims) {
        float min_dist = std::numeric_limits<float>::max();
        for (int j = 0; j < n_cent_eles; j += args.dims) {
            auto dist = distance(args.point_elements + i, 
                                 args.centroid_elements + j, args.dims);
            if (dist < min_dist) {
                min_dist = dist;
                args.labels[i / args.dims] = j / args.dims;
            }
        }
    }
}

static void recompute_centroids(kmeans_args_t& args, int *counts) {
    static int n_cent_eles = args.n_clusters * args.dims;
    memset(args.centroid_elements, 0.0f, n_cent_eles * sizeof(float));
    memset(counts, 0, args.n_clusters * sizeof(int));
    for (int i = 0; i < args.n_points; ++i) {
        int centroid_index = args.labels[i];
        add_vect(args.centroid_elements + centroid_index * args.dims, 
                 args.point_elements + i * args.dims, args.dims);
        counts[centroid_index]++;
    }
    for (int i = 0; i < args.n_clusters; i++) {
        div_by_int(args.centroid_elements + i * args.dims, counts[i], args.dims);
    }
}

void kmeans_seq(kmeans_args_t& args) {
    // auto all_start = chrono::high_resolution_clock::now();
    int n_cent_eles = args.n_clusters * args.dims;
    float *old_centroid_elements = (float*) malloc(n_cent_eles * sizeof(float));
    int *counts = (int*) malloc(args.n_clusters * sizeof(int));
    bool done = false;
    int iters = 0;
    chrono::milliseconds iter_time(0);
    while (!done) {
        auto start = chrono::high_resolution_clock::now();
        memcpy(old_centroid_elements, args.centroid_elements, n_cent_eles * sizeof(float));
        iters++;
        assign_centroids(args);
        recompute_centroids(args, counts);
        done = (iters >= args.max_iters) || converged(args, old_centroid_elements);
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<std::chrono::milliseconds>(end - start);
        iter_time += diff;
    }
    args.iters = iters;
    args.iter_time = static_cast<float>(iter_time.count());
    args.iter_time /= args.iters;
    free(old_centroid_elements);
    free(counts);
    // auto all_end = chrono::high_resolution_clock::now();
    // auto diff = chrono::duration_cast<std::chrono::milliseconds>(all_end - all_start);
    // cout << "End-to-end : " << diff.count() << " ms\n";
}