#pragma once
#include <iostream>
#include <vector>
#include <Eigen\Dense>
#include "types.h"

class loss
{
public:
	std::string mode;
	double target;
	double pred;

public:
	loss() {};
	loss(std::string const mode)
		:mode(mode) {};
	
	double get_loss(double target, double pred) { double loss = target - pred; return loss; };

	RowVecXd get_gradient(std::string const mode, double target, double pred, std::vector<double> input, double lr);
	RowVecXd get_LMS_gradient(double target, double pred, std::vector<double> input, double lr);
};

