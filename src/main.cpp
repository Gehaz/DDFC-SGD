#include <iostream>
#include <random>
#include <string.h>
#include <cstdlib>
#include "seq_sgd/seq_sgd.hpp"
#include "parallel_sgd/parallel_sgd.hpp"
#include "obj_function.hpp"


// generate toy data
// x_i = i, y_i = 2x_i-1 + eps, eps ~ N(0,0.5)
void gen_toy_data(std::vector< std::vector<double> >& X, std::vector<double>& Y) {
    unsigned int n = X.size(), d = X[0].size()-1;
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0,0.5);
    for (unsigned int i = 0; i < n; i++) { 
        X[i][d-1] = i+1; 
        X[i][d] = 1; // intercept
        Y[i] = 2*(i+1)-1 + dist(gen);
    }
}
    

int main(int argc, char* argv[]) {

    // perform linear regression 
    unsigned int d = 1, n = 100;
    double loss = 0.0;
	
	bool parallel = false;
	float lr = 0.00005
	unsigned int num_iters = 2000, num_batches = 10, num_threads = 4;
	
	// load user parameters
	for (unsigned int i = 1; i < argc; i++) {
		arg = argv[i]
		if (strcmp(arg, "parallel") == 0)
			parallel = true;
		else if (strncmp(arg, "d=", strlen("d=")))
			d = atoi(arg.substr(strlen("d=")))
		else if (strncmp(arg, "n=", strlen("n=")))
			n = atoi(arg.substr(strlen("n=")))
		else if (strncmp(arg, "lr=", strlen("lr=")))
			lr = atoi(arg.substr(strlen("lr=")))
		else if (strncmp(arg, "iters=", strlen("iters=")))
			num_iters = atoi(arg.substr(strlen("iters=")))
		else if (strncmp(arg, "batches=", strlen("batches=")))
			num_batches = atoi(arg.substr(strlen("batches=")))
		else if (strncmp(arg, "threads=", strlen("threads=")))
			num_threads = atoi(arg.substr(strlen("threads=")))
	}
	
    std::vector<double> weights(d+1);
    std::vector< std::vector<double> > X(n, std::vector<double>(d+1));
    std::vector<double> Y(n);

    // toy data
    gen_toy_data(X, Y);
    std::cout << "X = [ ";
    for (unsigned int i = 0; i < n; i++)
        std::cout << X[i][0] << " ";
    std::cout << "]" << std::endl;
    std::cout << "Y = [ ";
    for (auto& e : Y)
        std::cout << e << " ";
    std::cout << "]" << std::endl;

    // init optimizer and perform sgd
    if (argc > 1 && parallel) {
        parallel_sgd optimizer(&linear_reg_obj, &linear_reg_obj_grad, weights, X, Y, lr, num_iters, num_batches, num_threads);
        optimizer.update(100);
        weights = optimizer.get_weights();
        loss = optimizer.get_loss();
    }
    else { 
        seq_sgd optimizer(&linear_reg_obj, &linear_reg_obj_grad, weights, X, Y, lr, num_iters, num_batches);
        optimizer.update(100);
        weights = optimizer.get_weights();
        loss = optimizer.get_loss();
    }

    // print
    std::cout << "w = [ ";
    for (auto& e : weights)
        std::cout << e << " ";
    std::cout << "]" << std::endl;
    std::cout << "loss: " << loss << std::endl;

    return 0;
}