#include <cstdlib>
#include <iostream>
#include <unistd.h>

#include "argparse.h"

void get_opts(int argc, char **argv, options_t& opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-k <num_cluster>" << std::endl;
        std::cout << "\t-d <dims>" << std::endl;
        std::cout << "\t-i <file_path>" << std::endl;
        std::cout << "\t-m <max_num_iter>" << std::endl;
		std::cout << "\t-t <threshold>" << std::endl;
		std::cout << "\t-s <seed>" << std::endl;
        std::cout << "\t-v <implementation>" << std::endl;
        std::cout << "\t[Optional] -c" << std::endl;
        exit(0);
    }

    opts.print_centroids = false;

    int ch;
	extern char *optarg;
	extern int optopt;
    while ((ch = getopt(argc, argv, "k:d:i:m:t:s:v:c")) != -1)
    {
        switch (ch)
        {
        case 0:
            break;
		case 'k':
			opts.n_clusters = atoi(optarg);
			break;
		case 'd':
			opts.dims = atoi(optarg);
			break;
        case 'i':
            opts.in_file = optarg;
            break;
        case 'm':
            opts.max_iters = atoi(optarg);
            break;
        case 't':
            opts.threshold = strtof(optarg, NULL);
            break;
        case 's':
            opts.seed = atoi(optarg);
            break;
        case 'v':
            opts.implementation = static_cast<Implementation>(atoi(optarg));
            break;
        case 'c':
            opts.print_centroids = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
