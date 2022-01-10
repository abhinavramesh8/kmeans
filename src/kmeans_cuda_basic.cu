#include <cmath>
#include <cuda.h>
#include <math_constants.h>

#include "helpers.h"
#include "kmeans_cuda_basic.h"

__device__ __host__ float distance(float *x, float *y, int dims) {
    float sq_dist = 0.0f;
    for (int i = 0; i < dims; i++) {
        float diff = x[i] - y[i];
        sq_dist += (diff * diff);
    }
    return sqrtf(sq_dist);
}

__device__ void add_vector(float *to_vect, float *from_vect, int dims) {
    for (int i = 0; i < dims; i++) {
        atomicAdd(to_vect + i, from_vect[i]);
    }
}

__global__ void assign_centroid(float *pt_eles, float *cent_eles, int *labels, 
                                int n_points, int n_clusters, int dims) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx < n_points) {
        float *point = pt_eles + pt_idx * dims;
        float min_dist = CUDART_INF_F;
        int min_idx;
        for (int i = 0; i < n_clusters; i++) {
            float dist = distance(point, cent_eles + dims * i, dims);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }
        labels[pt_idx] = min_idx;
    }
}

__global__ void zero_centroid_element(float *cent_eles, int n_cent_eles) {
    int cent_ele_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cent_ele_idx < n_cent_eles) {
        cent_eles[cent_ele_idx] = 0.0f;
    }
}

__global__ void add_point_to_centroid(float *cent_eles, float *pt_eles, int *labels, 
                                      int *counts, int n_points, int dims) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx < n_points) {
        int cent_idx = labels[pt_idx];
        add_vector(cent_eles + dims * cent_idx, pt_eles + pt_idx * dims, dims);
        atomicAdd(counts + cent_idx, 1);
    }
}

__global__ void average_centroid_element(float *cent_eles, int *counts, 
                                         int n_cent_eles, int dims) {
    int cent_ele_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cent_ele_idx < n_cent_eles) {
        int count = counts[cent_ele_idx / dims];
        if (count != 0) {
            cent_eles[cent_ele_idx] /= count;
        }
    }
}

__global__ void centroid_converged(float *cent_eles, float *old_cent_eles, 
                                   bool *converged, float threshold, 
                                   int n_clusters, int dims) {
    int cent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cent_idx < n_clusters) {
        int offset = dims * cent_idx;
        float dist = distance(cent_eles + offset, old_cent_eles + offset, dims);
        if (dist > threshold) {
            converged[cent_idx] = false;
        } else {
            converged[cent_idx] = true;
        }
    }
}

void kmeans_cuda_basic(kmeans_args_t& args) {
    float iter_diff = 0.0f;
    cudaEvent_t iter_start, iter_stop;
//     float all_diff = 0.0f, init_diff = 0.0f, end_diff = 0.0f, tran1_diff = 0.0f, 
//           tran2_diff = 0.0f;
//     cudaEvent_t all_start, all_stop, init_start, init_stop, end_start, end_stop, 
//                 tran1_start, tran1_stop, tran2_start, tran2_stop;
//     cudaEventCreate(&all_start);
//     cudaEventCreate(&all_stop);
//     cudaEventCreate(&init_start);
//     cudaEventCreate(&init_stop);
//     cudaEventCreate(&end_start);
//     cudaEventCreate(&end_stop);
//     cudaEventCreate(&tran1_start);
//     cudaEventCreate(&tran1_stop);
//     cudaEventCreate(&tran2_start);
//     cudaEventCreate(&tran2_stop);
    cudaEventCreate(&iter_start);
    cudaEventCreate(&iter_stop);
//     cudaEventRecord(all_start);
//     cudaEventRecord(init_start);
    
    int n_cent_eles = args.n_clusters * args.dims;
    int cent_eles_sz = n_cent_eles * sizeof(float);
    float *dev_centroid_elements;
    cudaMalloc((void**)&dev_centroid_elements, cent_eles_sz);
    cudaMemcpy(dev_centroid_elements, args.centroid_elements, cent_eles_sz,
               cudaMemcpyHostToDevice);
    
    float *old_centroid_elements;
    cudaMalloc((void**)&old_centroid_elements, cent_eles_sz);
     
    int pt_eles_sz = args.n_points * args.dims * sizeof(float);
    float *dev_point_elements;
    cudaMalloc((void**)&dev_point_elements, pt_eles_sz);
    cudaMemcpy(dev_point_elements, args.point_elements, pt_eles_sz,
               cudaMemcpyHostToDevice);
      
    int *dev_counts;
    int counts_sz = args.n_clusters * sizeof(int);
    cudaMalloc((void**)&dev_counts, counts_sz);
    
    int *dev_labels;
    int labels_sz = args.n_points * sizeof(int);
    cudaMalloc((void**)&dev_labels, labels_sz);
    
    int conv_sz = args.n_clusters * sizeof(bool);
    bool *converged = (bool*) malloc(conv_sz);
    bool *dev_converged;
    cudaMalloc((void**)&dev_converged, conv_sz);
    
//     cudaEventRecord(init_stop);
//     cudaEventSynchronize(init_stop);
//     cudaEventElapsedTime(&init_diff, init_start, init_stop);
    
    int iters = 0;
    bool done = false;
    while (!done) {
        cudaEventRecord(iter_start);
//         cudaEventRecord(tran1_start);
        cudaMemcpy(old_centroid_elements, dev_centroid_elements, cent_eles_sz,
                   cudaMemcpyDeviceToDevice);
//         cudaEventRecord(tran1_stop);
//         cudaEventSynchronize(tran1_stop);
        float diff;
//         cudaEventElapsedTime(&diff, tran1_start, tran1_stop);
//         tran1_diff += diff;
        
        iters++;
        
        cudaMemset(dev_counts, 0, counts_sz);
        
        int n_threads_per_block = 256;
        int n_blocks_pts = (int)ceil(args.n_points / (float)n_threads_per_block);
        int n_blocks_cent_eles = (int)ceil(n_cent_eles / (float)n_threads_per_block);
        int n_blocks_cents = (int)ceil(args.n_clusters / (float)n_threads_per_block);
        
        assign_centroid<<<n_blocks_pts, n_threads_per_block>>>(
            dev_point_elements, dev_centroid_elements, dev_labels, args.n_points, 
            args.n_clusters, args.dims);
        
        zero_centroid_element<<<n_blocks_cent_eles, n_threads_per_block>>>(
            dev_centroid_elements, n_cent_eles);
        
        add_point_to_centroid<<<n_blocks_pts, n_threads_per_block>>>(
            dev_centroid_elements, dev_point_elements, dev_labels, dev_counts, 
            args.n_points, args.dims);
        
        average_centroid_element<<<n_blocks_cent_eles, n_threads_per_block>>>(
            dev_centroid_elements, dev_counts, n_cent_eles, args.dims);
        
        centroid_converged<<<n_blocks_cents, n_threads_per_block>>>(
            dev_centroid_elements, old_centroid_elements, dev_converged, 
            args.threshold, args.n_clusters, args.dims);
        
//         cudaEventRecord(tran2_start);
        cudaMemcpy(converged, dev_converged, conv_sz, cudaMemcpyDeviceToHost);
//         cudaEventRecord(tran2_stop);
//         cudaEventSynchronize(tran2_stop);
//         cudaEventElapsedTime(&diff, tran2_start, tran2_stop);
//         tran2_diff += diff;
        cudaEventRecord(iter_stop);
        cudaEventSynchronize(iter_stop);
        cudaEventElapsedTime(&diff, iter_start, iter_stop);
        iter_diff += diff;
        bool centroids_converged = true;
        for (int i = 0; i < args.n_clusters; i++) {
            if (!converged[i]) {
                centroids_converged = false;
                break;
            }
        }
        done = (iters >= args.max_iters) || centroids_converged;
    }
    args.iters = iters;
    args.iter_time = iter_diff;
    args.iter_time /= iters;
    
//     cudaEventRecord(end_start);
    cudaMemcpy(args.point_elements, dev_point_elements, pt_eles_sz, 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(args.centroid_elements, dev_centroid_elements, cent_eles_sz, 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(args.labels, dev_labels, labels_sz, 
               cudaMemcpyDeviceToHost);
//     cudaEventRecord(end_stop);
//     cudaEventSynchronize(end_stop);
//     cudaEventElapsedTime(&end_diff, end_start, end_stop);
//     cudaEventRecord(all_stop);
//     cudaEventSynchronize(all_stop);
//     cudaEventElapsedTime(&all_diff, all_start, all_stop);
    
    cudaFree(dev_centroid_elements);
    cudaFree(old_centroid_elements);
    cudaFree(dev_point_elements);
    cudaFree(dev_counts);
    
//     cudaEventDestroy(all_start);
//     cudaEventDestroy(all_stop);
//     cudaEventDestroy(init_start);
//     cudaEventDestroy(init_stop);
//     cudaEventDestroy(end_start);
//     cudaEventDestroy(end_stop);
    cudaEventDestroy(iter_start);
    cudaEventDestroy(iter_stop);
//     cudaEventDestroy(tran1_start);
//     cudaEventDestroy(tran1_stop);
//     cudaEventDestroy(tran2_start);
//     cudaEventDestroy(tran2_stop);
//     cout << "End-to-end: " << all_diff << " ms\n";
//     cout << "Initial transfer: " << init_diff << " ms\n";
//     cout << "Final transfer: " << end_diff << " ms\n";
//     cout << "Total tran1: " << tran1_diff << " ms\n";
//     cout << "Total tran2: " << tran2_diff << " ms\n";
//     cout << "Total iter: " << iter_diff << " ms\n";
}