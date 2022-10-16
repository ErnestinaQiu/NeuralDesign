#include <iostream>
#include <Eigen\Dense>
#include <vector>
#include "types.h"
#include "learning_rate.h"

using namespace std;

int main()
{
	std::vector<double> a={0,1,2,3};
	learning_rate l_r;
	MatXXd R = l_r.get_R(a);
	std::cout << "R: " << R << std::endl;
}

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