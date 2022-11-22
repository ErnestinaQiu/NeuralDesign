#include "loss.h"
#include <iostream>
#include "types.h"

RowVecXd loss::get_gradient(std::string const mode, double target, double pred, std::vector<double> input, double lr)
{
	if (mode == "LMS") { return this->get_LMS_gradient(target, pred, input); }
	else { std::cout << "Not implement yet." << std::endl; }
}

RowVecXd loss::get_LMS_gradient(double target, double pred, std::vector<double> input, double lr)
{
	double err = this->get_loss(target, pred);
	RowVecXd input_mat;
	for (int i = 0; i < input.size(); i++) { input_mat[1, i] = input[i]; }
	RowVecXd gradient = 2 * lr * err * input_mat;
	return gradient;
}