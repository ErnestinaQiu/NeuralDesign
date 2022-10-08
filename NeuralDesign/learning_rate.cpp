#include "learning_rate.h"
#include <iostream>
#include <Eigen/Dense>

//MatXXf learning_rate::get_R(std::vector<int std::array> in_vectors)
//{
//	int vec_num = in_vectors.size();
//	int vec_size = in_vectors[0].size();
//	std::vector<Eigen::VectorXf> m_in_vecs(vec_num);
//	// check the dimension
//	for (int i=1; i < vec_num; i++)
//	{
//		if (in_vectors[i].size() != vec_size) { std::cout << "VALUE ERROR: in_vector[" << i << "] != in_vector[" << i + 1 << "]" << std::endl; MatXXf mat(0, 1); mat(0, 1) = 0; }
//		//m_in_vecs[i] = 
//	}
//	
//}