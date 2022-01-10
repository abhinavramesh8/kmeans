#ifndef _ARGPARSE_H
#define _ARGPARSE_H

enum Implementation { Seq, Thrust, Cuda, Shmem };

struct options_t {
    char *in_file;
    int n_clusters;
	int dims;
	int max_iters;
	int seed;
	float threshold;
    bool print_centroids;
    Implementation implementation;
};

void get_opts(int, char **, options_t &);

#endif
