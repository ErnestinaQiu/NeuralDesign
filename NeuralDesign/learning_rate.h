#pragma once
#include <iostream>
#include <Eigen\Dense>
#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXXf;

class learning_rate
{
protected:
	std::string m_type_name; // include "max_stable_lr", "fixed_value"
	float lr;
	MatXXf R;
	
protected:
	MatXXf get_R(std::vector<std::vector<float>> in_vectors);


public:
	learning_rate(std::string& type_name)
		:m_type_name(type_name) {};

	float get_lr(std::string);

};

