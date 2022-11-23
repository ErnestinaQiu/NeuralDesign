#include <iostream>
#include <Eigen\Dense>
#include <vector>
#include "types.h"
#include "learning_rate.h"
#include "Weights.h"
#include "loss.h"

using namespace std;

/*test loss.get_LMS_gradient*/
int  main()
{
	loss m_loss;
	double target = -1;
	double pred = 0;
	std::string mode = "LMS";
	std::vector<double> in_vec = {1,-1,-1};
	double lr = 0.2;
	RowVecXd grad = m_loss.get_LMS_gradient(target, pred, in_vec, lr);
	RowVecXd grad_2 = m_loss.get_gradient(mode, target, pred, in_vec, lr);
	std::cout << grad << std::endl;
	std::cout << grad_2 << std::endl;


	return 0;
}

/*test loss.get_loss*/
//int main()
//{
//	loss m_loss;
//	double target = -1;
//	double pred = 0;
//	double m_loss_value = m_loss.get_loss(target, pred);
//	std::cout << m_loss_value << std::endl;
//}

/*test  */
//int main()
//{
//	Weights wei;
//	std::string const mode = "zeros";
//	MatXXd zero = wei.initiate_weights(mode, 2, 1);
//	std::cout << zero << std::endl;
//}

/* test get_lr */
//int main()
//{
//	std::vector<double> vec1{ 1,-1,-1 };
//	std::vector<double> vec2{ 1, 1,-1 };
//	std::vector<std::vector<double>> total_vec{ vec1, vec2 };
//	learning_rate l_r;
//	MatXXd R = l_r.get_R(total_vec);
//	std::cout << R << std::endl;
//	double lr = l_r.get_lr(total_vec);
//	std::cout << lr << std::endl;
//}

//int main()
//{
//	MatXXd mat{{1,1,1}, {0,1,1}, {0,0,-1}};
//	MatXXd mat1{ {1,1,1}, {1,1,1}, {1,1,1} };
//
//	std::cout << mat + mat1 << std::endl;
//	std::cout << mat1/2 << std::endl;


	//learning_rate l_r;
	//double eigen_vals;
	//eigen_vals = l_r.get_upper_limit(mat);
	//VecXcd eigen_vals;
	//eigen_vals = mat.eigenvalues();
	//std::cout << "eigen_values: " << eigen_vals << std::endl;
//}

//int main()
//{
	//Mat22i a;
	//std::vector<int> vec {0,1,2,3};
	//int arr[] = {0, 1, 2, 3};
	//double arrs[] = { 1.1, 2.1, 3.1, 4.1 };

	//Eigen::Map<Mat22i>b(arr);
	//Eigen::Map<RowVec4d>a(arrs);

	//Vec4d c = a.transpose();
	//Eigen::Matrix<double, 4, 4> d = c * a;

	////std::cout << vec1 << std::endl;

	//std::cout << "c: " << c << std::endl <<  "d: " << d << std::endl;

	//std::cout << "c[2] " << c[2] << std::endl;

	//for (int i = 0; i < 1; i++)
	//{
	//	for (int j = 0; j < 4; j++)
	//	{
	//		std::cout << "arr(" << i << ", " << j << ") = " << a(i, j) << std::endl;
	//		b(i, j) = arr[2 * i + j];
	//		std::cout << "b(" << i << ", " << j << ") = " << b(i, j) << std::endl;
	//	}
	//}
	//std::cout << "b: " << b << std::endl;

//}