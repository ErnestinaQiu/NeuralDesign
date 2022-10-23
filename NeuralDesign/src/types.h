#pragma once
#include <iostream>
#include <Eigen\Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXXd;
typedef Eigen::Matrix<int, 2, 2> Mat22i;
typedef Eigen::Matrix<double, 1, 4 > RowVec4d;
typedef Eigen::Matrix<double, 4, 1 > Vec4d;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVecXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecXd;
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> VecXcd;

class types
{
};

