#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/find.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>

#include <chrono>

#include "helpers.h"

struct unary_modulus {
    int a;
    
    __host__ __device__
        int operator()(const int& x) const {
            return x % a;
        }
};

struct unary_divide {
    int a;
    
    __host__ __device__
        int operator()(const int& x) const {
            return x / a;
        }
};

struct binary_divide {
    __host__ __device__
        float operator()(const float& x, const int& y) const {
            if (y != 0) {
                return x / y;
            }
            return x;
        }
};

struct point_element_index {
    int dims;
    int n_cent_eles;
    
    __host__ __device__
        int operator()(const int& i) const {
            return (dims * (i / n_cent_eles) + i % dims);
        }
};

struct squared_difference {
    __host__ __device__
        float operator()(const thrust::tuple<float, float>& x) const {
            float diff = thrust::get<0>(x) - thrust::get<1>(x);
            return diff * diff;
        }
};

struct arg_min {
    __host__ __device__
        thrust::tuple<int, float> operator()(const thrust::tuple<int, float>& x, 
                                             const thrust::tuple<int, float>& y) const {
            if (thrust::get<1>(x) < thrust::get<1>(y)) {
                return x;
            }
            else {
                return y;
            }
        }
};

struct first_element {
    __host__ __device__
        int operator()(const thrust::tuple<int, float>& x) const {
            return thrust::get<0>(x);
        }
};

struct centroid_element_index {
    int dims;
    
    __host__ __device__
        int operator()(const thrust::tuple<int, int>& x) const {
            return (thrust::get<0>(x) + dims * thrust::get<1>(x));
        }
};

struct square_root {
    __host__ __device__
        float operator()(const float& x) const {
            return sqrtf(x);
        }
};

struct greater_than {
    float a;
    
    __host__ __device__
        bool operator()(const float& x) const {
            return (x > a);
        }
};

static void assign_centroids(const thrust::device_vector<float>& pt_eles,
                             const thrust::device_vector<float>& cent_eles,
                             thrust::device_vector<int>& labels, int n_points,
                             int n_clusters, int dims) {
    static int n_cent_eles = n_clusters * dims;
    static int n_pt_cent_pairs = n_points * n_clusters;
    static int n_pt_cent_eles = n_pt_cent_pairs * dims;
    static thrust::counting_iterator<int> counter_beg(0);
    static auto pt_ele_idx_beg = thrust::make_transform_iterator(counter_beg, 
                                                                 point_element_index{dims, 
                                                                                     n_cent_eles});
    static auto pt_ele_beg = thrust::make_permutation_iterator(pt_eles.begin(), 
                                                               pt_ele_idx_beg);
    static auto cent_ele_idx_beg = thrust::make_transform_iterator(counter_beg, 
                                                                   unary_modulus{n_cent_eles});
    static auto cent_ele_beg = thrust::make_permutation_iterator(cent_eles.begin(), 
                                                                 cent_ele_idx_beg);
    static auto pt_cent_ele_beg = thrust::make_zip_iterator(thrust::make_tuple(pt_ele_beg, 
                                                                               cent_ele_beg));
    static auto sq_diff_beg = thrust::make_transform_iterator(pt_cent_ele_beg, 
                                                              squared_difference{});
    static auto idx_beg = thrust::make_transform_iterator(counter_beg, unary_divide{dims});
    thrust::device_vector<float> sq_dists(n_pt_cent_pairs);
    thrust::device_vector<int> output_keys(n_pt_cent_pairs);
    thrust::reduce_by_key(idx_beg, idx_beg + n_pt_cent_eles, sq_diff_beg, 
                          output_keys.begin(), sq_dists.begin());
    output_keys.resize(n_points);
    static auto pt_idx_beg = thrust::make_transform_iterator(counter_beg, 
                                                             unary_divide{n_clusters});
    static auto cent_idx_beg = thrust::make_transform_iterator(counter_beg, 
                                                               unary_modulus{n_clusters});
    auto cent_idx_sq_dist_beg = thrust::make_zip_iterator(thrust::make_tuple(
        cent_idx_beg, sq_dists.begin()));
    thrust::device_vector<thrust::tuple<int, float> > labels_sq_dists(n_points);
    thrust::reduce_by_key(pt_idx_beg, pt_idx_beg + n_pt_cent_pairs, cent_idx_sq_dist_beg,
                          output_keys.begin(), labels_sq_dists.begin(), 
                          thrust::equal_to<int>(), arg_min{});
    thrust::transform(labels_sq_dists.begin(), labels_sq_dists.end(), 
                      labels.begin(), first_element{});
}

static void recompute_centroids(thrust::device_vector<float>& cent_eles,
                                thrust::device_vector<int>& labels,
                                const thrust::device_vector<float>& pt_eles,
                                int n_clusters, int n_points, int dims) {
    static thrust::counting_iterator<int> counter_beg(0);
    static auto idx_beg = thrust::make_transform_iterator(counter_beg, 
                                                          unary_divide{dims});
    static auto cent_idx_beg = thrust::make_permutation_iterator(labels.begin(), 
                                                                 idx_beg);
    static auto dim_beg = thrust::make_transform_iterator(counter_beg, 
                                                          unary_modulus{dims});
    static auto dim_cent_idx_beg = thrust::make_zip_iterator(thrust::make_tuple(
        dim_beg, cent_idx_beg));
    static auto cent_ele_idx_beg = thrust::make_transform_iterator(dim_cent_idx_beg, 
                                                                   centroid_element_index{dims});
    thrust::device_vector<float> dev_pt_eles = pt_eles;
    static int n_pt_eles = n_points * dims;
    thrust::device_vector<int> cent_ele_idx(cent_ele_idx_beg, cent_ele_idx_beg + n_pt_eles);
    thrust::sort_by_key(cent_ele_idx.begin(), cent_ele_idx.end(), dev_pt_eles.begin());
    static int n_cent_eles = n_clusters * dims;
    thrust::device_vector<int> output_keys(n_cent_eles);
    thrust::reduce_by_key(cent_ele_idx.begin(), cent_ele_idx.end(), dev_pt_eles.begin(),
                          output_keys.begin(), cent_eles.begin());
    thrust::device_vector<int> counts(n_clusters);
    thrust::sort(labels.begin(), labels.end());
    thrust::upper_bound(labels.begin(), labels.end(), counter_beg, 
                        counter_beg + n_clusters, counts.begin());
    thrust::adjacent_difference(counts.begin(), counts.end(), counts.begin());
    auto cent_count_beg = thrust::make_permutation_iterator(counts.begin(), idx_beg);
    thrust::transform(cent_eles.begin(), cent_eles.end(), cent_count_beg, 
                      cent_eles.begin(), binary_divide{});
}

static bool converged(const thrust::device_vector<float>& cent_eles, 
                      thrust::device_vector<float>& old_cent_eles, float threshold, 
                      int n_clusters, int dims) {
    static int n_cent_eles = n_clusters * dims;
    static auto cent_old_cent_ele_beg = thrust::make_zip_iterator(thrust::make_tuple(
        cent_eles.begin(), old_cent_eles.begin()));
    static auto sq_diff_beg = thrust::make_transform_iterator(cent_old_cent_ele_beg, 
                                                              squared_difference{});
    static thrust::counting_iterator<int> counter_beg(0);
    static auto cent_idx_beg = thrust::make_transform_iterator(counter_beg, 
                                                               unary_divide{dims});
    thrust::device_vector<float> distances(n_clusters);
    thrust::device_vector<int> cent_idx(n_clusters);
    thrust::reduce_by_key(cent_idx_beg, cent_idx_beg + n_cent_eles, 
                          sq_diff_beg, cent_idx.begin(), 
                          distances.begin());
    thrust::transform(distances.begin(), distances.end(), distances.begin(), 
                      square_root{});
    auto iter = thrust::find_if(distances.begin(), distances.end(), 
                                greater_than{threshold});
    return (iter == distances.end());
}

void kmeans_thrust(kmeans_args_t& args) {
    // auto all_start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<float> pt_eles(args.point_elements, 
                                         args.point_elements + args.n_points * args.dims);
    int n_cent_eles = args.n_clusters * args.dims;
    // auto init_start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<float> cent_eles(args.centroid_elements, 
                                           args.centroid_elements + n_cent_eles);
    thrust::device_vector<float> old_cent_eles(n_cent_eles);
    thrust::device_vector<int> labels(args.n_points);
    // auto init_end = std::chrono::high_resolution_clock::now();
    bool done = false;
    int iters = 0;
    std::chrono::milliseconds iter_time(0);
    while (!done) {
        auto start = std::chrono::high_resolution_clock::now();
        thrust::copy(cent_eles.begin(), cent_eles.end(), old_cent_eles.begin());
        iters++;
        assign_centroids(pt_eles, cent_eles, labels, args.n_points, 
                         args.n_clusters, args.dims);
        recompute_centroids(cent_eles, labels, pt_eles, args.n_clusters, 
                            args.n_points, args.dims);
        done = (iters >= args.max_iters) || converged(cent_eles, old_cent_eles, 
                                                      args.threshold, 
                                                      args.n_clusters, args.dims);
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        iter_time += diff;
    }
    // auto fin_start = std::chrono::high_resolution_clock::now();
    thrust::copy(cent_eles.begin(), cent_eles.end(), args.centroid_elements);
    thrust::copy(pt_eles.begin(), pt_eles.end(), args.point_elements);
    thrust::copy(labels.begin(), labels.end(), args.labels);
    // auto fin_end = std::chrono::high_resolution_clock::now();
    args.iters = iters;
    args.iter_time = static_cast<float>(iter_time.count());
    args.iter_time /= args.iters;
//     auto all_end = std::chrono::high_resolution_clock::now();
//     auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(all_end - all_start);
//     cout << "End-to-end: " << diff.count() << " ms\n";
//     diff = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);
//     cout << "Initial data transfer: " << diff.count() << " ms\n";
//     diff = std::chrono::duration_cast<std::chrono::milliseconds>(fin_end - fin_start);
//     cout << "Final data transfer: " << diff.count() << " ms\n";
}