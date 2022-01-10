#include "argparse.h"
#include "io.h"
#include "helpers.h"
#include "kmeans_seq.h"
#include "kmeans_thrust.h"
#include "kmeans_cuda_basic.h"
#include "kmeans_cuda_shmem.h"

int main(int argc, char **argv) {
    options_t opts;
    get_opts(argc, argv, opts);
    
    kmeans_args_t args;
    fill_args(args, opts);
    
    switch(opts.implementation) {
        case Seq:
            kmeans_seq(args);
            break;
        case Thrust:
            kmeans_thrust(args);
            break;
        case Cuda:
            kmeans_cuda_basic(args);
            break;
        case Shmem:
            kmeans_cuda_shmem(args);
            break;
    }
    
    write_to_stdout(args);
        
    return 0;
}