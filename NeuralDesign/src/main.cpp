#include <iostream>
#include <Eigen\Dense>
#include <vector>

using namespace std;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> MatXXf;
typedef Eigen::Matrix<int, 2, 2> Mat22i;


int main()
{
	//Mat22i a;
	std::vector<int> vec {0,1,2,3};
	int arr[] = {0, 1, 2, 3};

	Mat22i b;

	Eigen::Map<Mat22i> a(arr);

	std::cout << "a: " <<  a << std::endl;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			std::cout << "arr(" << i << ", " << j << ") = " << a(i, j) << std::endl;
			b(i, j) = arr[2 * i + j];
			std::cout << "b(" << i << ", " << j << ") = " << b(i, j) << std::endl;
		}

	}
	std::cout << "b: " << b << std::endl;

}


