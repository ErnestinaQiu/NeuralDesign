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

//RowVecXd learning_rate::get_upper_limit(MatXXd& R)
//{
//	learning_rate::R_eigenvalue(1, learning_rate::fea_dim) = R.eigenvalues();
//	return learning_rate::R_eigenvalue;
//}
//
//double learning_rate::get_lr(std::vector<double>& vec)
//{
//	MatXXd R = learning_rate::get_R(vec);
//	RowVecXd eigenvalue= learning_rate::get_upper_limit(R);
//	return 1;
//}