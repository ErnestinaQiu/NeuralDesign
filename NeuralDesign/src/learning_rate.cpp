#include "learning_rate.h"
#include <iostream>

MatXXd learning_rate::get_R(std::vector<double>& vec)
{
	int vec_num = vec.size();
	Eigen::Matrix<double, Eigen::Dynamic, 1> a(vec_num, 1);
	Eigen::Matrix<double, 1, Eigen::Dynamic> b(1, vec_num);
	MatXXd R(vec_num, vec_num);
	
	for (int i = 0; i < vec_num; i++)
	{
		a[i] = vec[i];
	}
	b = a.transpose();
	R = a * b;
	return R;
}
