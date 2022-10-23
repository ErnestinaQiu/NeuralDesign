#include "learning_rate.h"
#include <iostream>

MatXXd learning_rate::get_R(std::vector<double>& vec)
{
	int vec_num = vec.size();
	learning_rate::fea_dim = vec_num;

	Eigen::Matrix<double, Eigen::Dynamic, 1> a(vec_num, 1);
	Eigen::Matrix<double, 1, Eigen::Dynamic> b(1, vec_num);
	MatXXd R(vec_num, vec_num);
	
	for (int i = 0; i < vec_num; i++)
	{
		a[i] = vec[i];
	}
	b = a.transpose();
	learning_rate::m_R = a * b;

	return R;
}

double learning_rate::get_upper_limit(MatXXd& R)
{
	VecXcd eigen_vals = R.eigenvalues(); 
	//learning_rate::R_eigenvalue = eigen_vals;
	double min_eigenval = 0;
	for (int i = 0; i < eigen_vals.rows(); i++)
	{
		if (eigen_vals(i, 0).real() > min_eigenval) { min_eigenval = eigen_vals(i).real(); }
	}
	return min_eigenval;
}

double learning_rate::get_lr(std::vector<double>& vec)
{
	MatXXd R = learning_rate::get_R(vec);
	double eigenvalue= learning_rate::get_upper_limit(R);
	return 1;
}