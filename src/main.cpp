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
	float lr = 0.00005;
	unsigned int num_iters = 2000, num_batches = 10, num_threads = 4;
	
	// load user parameters
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "parallel") == 0)
			parallel = true;
		else if (strcmp(argv[i], "--d") == 0)
			d = atoi(argv[i+1]);
		else if (strcmp(argv[i], "--n") == 0)
			n = atoi(argv[i+1]);
		else if (strcmp(argv[i], "--lr") == 0)
			lr = atoi(argv[i+1]);
		else if (strcmp(argv[i], "--iters") == 0)
			num_iters = atoi(argv[i+1]);
		else if (strcmp(argv[i], "--batches") == 0)
			num_batches = atoi(argv[i+1]);
		else if (strcmp(argv[i], "--threads") == 0)
			num_threads = atoi(argv[i+1]);
	}
    printf("d=%d, n=%d, lr=%f, iters=%d, batches=%d, threads=%d\n", d, n, lr, num_iters, num_batches, num_threads);
    printf("parallel run: %i\n", parallel);
	
    std::vector<double> weights(d+1);
    std::vector< std::vector<double> > X(n, std::vector<double>(d+1));
    std::vector<double> Y(n);

    // toy data
    gen_toy_data(X, Y);
    if (false)
    {
        std::cout << "X = [ ";
        for (unsigned int i = 0; i < n; i++)
            std::cout << X[i][0] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Y = [ ";
        for (auto& e : Y)
            std::cout << e << " ";
        std::cout << "]" << std::endl;
    }

    // init optimizer and perform sgd
    if (parallel) {
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