#pragma once
#include <iostream>
#include <Eigen\Dense> 
#include <vector>
#include "types.h"

class Weights
{
private:
	std::string const mode; // range ["zeros"], plan to impletement "randon" mode later

public:
	MatXXd weights_mat;
	int debug = 1;

public:
	Weights() {};
	Weights(int debug) :debug(debug) {};
	Weights(std::string const init_mode)
		:mode(init_mode) {}
	Weights(std::string const init_mode, int debug)
		:mode(init_mode), debug(debug) {}
	MatXXd initiate_weights(std::string const& mode, int const& feature_num, int const& row_num);
	MatXXd Weights::update_weights(std::string const& alg_name, MatXXd weights, double lr, double target, double pred, std::vector<double> input);
	
};