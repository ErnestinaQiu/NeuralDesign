#include "Weights.h"
#include <iostream>
#include <Eigen\Dense>
#include "types.h"
#include "loss.h"
#include "types.h"

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

MatXXd Weights::update_weights(std::string const& alg_name, MatXXd weights, double lr, std::vector<double> targets, std::vector<double> preds, std::vector<double> input)
/* @alg_name, ["LMS", ] */
{
	for (int i = 0; i < targets.size(); i++)
	{
		double target = targets[i];
		double pred = preds[i];
		RowVecXd grad = m_loss.get_gradient(alg_name, target, pred, preds, input, lr);

	}


}

