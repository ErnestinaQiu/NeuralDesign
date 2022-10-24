#pragma once
#include <iostream>
#include <Eigen\Dense>
#include <vector>
#include "types.h"

class learning_rate
{
protected:
	int fea_dim;
	MatXXd m_R;
	VecXcd R_eigenvalue;

public:
	std::string m_type_name; // include "max_stable_lr", "fixed_value"
	float lr;

protected:
	MatXXd get_sin_R(std::vector<double>& vec);

public:
	learning_rate(){};
	learning_rate(std::string& type_name)
		:m_type_name(type_name) {}
	
	double get_upper_limit(MatXXd& R);

	double get_lr(std::vector<double>& vec);

	MatXXd get_R(std::vector<std::vector<double>>& vects);
};