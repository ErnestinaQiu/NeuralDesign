#include "Weights.h"
#include <iostream>
#include <Eigen\Dense>
#include "types.h"
#include "loss.h"

MatXXd Weights::initiate_weights(std::string const& mode, int const& feature_num, int const& row_num)
{
	if (mode == "zeros")
	{
		return MatXXd::Zero(row_num, feature_num);
	}
	else
	{
		if (debug) { std::cout << "Not implement yet" << std::endl; }
	}
	return MatXXd::Zero(row_num, feature_num);
}

MatXXd Weights::update_weights(std::string const& alg_name, MatXXd weights, double lr, double target, double pred, std::vector<double> input)
{
	if (alg_name == "LMS")
	{
		loss 
	}

}

