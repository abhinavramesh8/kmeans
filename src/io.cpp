#include <fstream>
#include <cstdlib>
#include <cstdio>

#include "io.h"
#include "helpers.h"

void read_file(char *in_file, kmeans_args_t& args) {
	std::ifstream in;
	in.open(in_file);
	
	in >> args.n_points;
	args.point_elements = (float*) malloc(args.n_points * args.dims * sizeof(float));
	for (int i = 0; i < args.n_points; ++i) {
		in.ignore(256, ' '); // ignore the line number
		for (int j = 0; j < args.dims; ++j) {
			in >> args.point_elements[args.dims * i + j];
		}
	}
}

void write_to_stdout(const kmeans_args_t& args) {
    printf("%d,%f\n", args.iters, args.iter_time);
    
    if (args.print_centroids) {
        for (int cluster_id = 0; cluster_id < args.n_clusters; cluster_id++) {
            printf("%d ", cluster_id);
            for (int d = 0; d < args.dims; d++) {
                printf("%f ", args.centroid_elements[cluster_id * args.dims + d]);
            }
            printf("\n");
        }
    }
    else {
        printf("clusters:");
        for (int i = 0; i < args.n_points; i++) {
            printf(" %d", args.labels[i]);
        }
    }
}