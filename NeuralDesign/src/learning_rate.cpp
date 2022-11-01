#include "learning_rate.h"
#include <iostream>

MatXXd learning_rate::get_sin_R(std::vector<double>& vec)
{
	int vec_num = vec.size();
	this->fea_dim = vec_num;

	Eigen::Matrix<double, Eigen::Dynamic, 1> a(vec_num, 1);
	Eigen::Matrix<double, 1, Eigen::Dynamic> b(1, vec_num);
	
	for (int i = 0; i < vec_num; i++)
	{
		a[i] = vec[i];
	}
	b = a.transpose();
	this->m_R = a * b;

	return this->m_R;
}

double learning_rate::get_upper_limit(MatXXd& R)
{
	VecXcd eigen_vals = R.eigenvalues(); 
	//learning_rate::R_eigenvalue = eigen_vals;
	double max_eigenval = 0;
	for (int i = 0; i < eigen_vals.rows(); i++)
	{
		if (eigen_vals(i, 0).real() > max_eigenval) { max_eigenval = eigen_vals(i).real(); }
	}
	return max_eigenval;
}

double learning_rate::get_lr(std::vector<std::vector<double>>& vects)
{
	MatXXd R = this->get_R(vects);
	double eigenvalue= this->get_upper_limit(R);
	if (eigenvalue < 0) return 0;
	double max_stable_lr = 1 / eigenvalue;
	return max_stable_lr;
}

MatXXd learning_rate::get_R(std::vector<std::vector<double>>& vects)
{
	int vec_num = vects.size();
	MatXXd total_R;
	for (int i = 0; i < vec_num; i++)
	{	
		std::cout << "i: " << i << std::endl;
		std::vector<double> tmp_vec = vects[i];
		for (int j = 0; j < tmp_vec.size(); j++)
		{
			std::cout << tmp_vec[j];
		}
		std::cout << " " << std::endl;
		MatXXd tmp_r = this->get_sin_R(tmp_vec);
		std::cout << "tmp_r " << std::endl;
		std::cout << tmp_r << std::endl;
		if (i == 0) { total_R = tmp_r; continue; }
		else if (i != 0) { total_R += tmp_r; continue; }
	}
	total_R = total_R / vec_num;
	return total_R;
}

