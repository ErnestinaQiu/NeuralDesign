#pragma once
#include <iostream>
#include <Eigen\Dense> 
#include <vector>
#include "types.h"

class Weights
{
private:
	std::string mode; // range ["zeros"], plan to impletement "randon" mode later

public:
	MatXXd weights_mat;


public:
	Weights() {};
	Weights(std::string init_mode)
		:mode(init_mode) {}
	MatXXd initiate_weights(std::string& mode);
	MatXXd update_weights();

};

