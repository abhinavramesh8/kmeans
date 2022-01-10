# K-Means

## Introduction

K-Means is a machine-learning algorithm most commonly used for unsupervised learning. In the clustering problem, we are given a training set *x(1),...,x(m)*, and want to group the data into cohesive "clusters." We are given feature vectors for each data point *x(i)* encoded as floating-point vectors in D-dimensional space. But we have no labels. Our goal is to predict k centroids and a label *c(i)* for each datapoint.

## Requirements

- A C++ compiler that supports the C++14 standard.

- An NVIDIA GPU with the whole CUDA toolchain setup along with `nvcc` added to `$PATH` so that it can be invoked directly

## Input

The program assumes that the input first contains the number of data points. After that, each data point appears on its own line with its index(starting from 1) followed by the coordinates that correspond to the *d* dimensions, all separated by spaces. The `input` directory contains some sample inputs.

## Usage

Run `make` to compile the program, which results in an executable named `kmeans` within the `bin` directory. This can be executed with the following command-line arguments:

- `-k num_clusters`: an integer specifying the number of clusters
- `-d dims`: an integer specifying the dimension of the points
- `-i inputfilename`: a string specifying the input filename
- `-m max_num_iters`: an integer specifying the maximum number of iterations
- `-t threshold`: a float specifying the threshold for convergence test
- `-c`: a flag to control the output of the program. If `-c` is specified, it outputs the centroids of all clusters. If `-c` is not specified, it outputs the labels of all points.
- `-s seed`: an integer specifying the seed for the random generation of the initial centroids.
- `-v version`: an integer from 0-3 specifying the version of kmeans to execute. The versions correspond to sequential, Thrust, CUDA and CUDA with shared memory, in that order.

The program will output the total number of iterations for kmeans to converge and the average elapsed time per iteration in milliseconds, after which it outputs either all the centroids or the labels of all the points, depending on the `-c` flag.