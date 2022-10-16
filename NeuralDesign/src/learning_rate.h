#pragma once
#include <iostream>
#include <Eigen\Dense>
#include <vector>
#include "types.h"

class learning_rate
{
protected:
	std::string m_type_name; // include "max_stable_lr", "fixed_value"
	float lr;


public:
	learning_rate(){};
	learning_rate(std::string& type_name)
		:m_type_name(type_name) {};

	float get_lr(std::string);
	MatXXd get_R(std::vector<double>& vec);
};

