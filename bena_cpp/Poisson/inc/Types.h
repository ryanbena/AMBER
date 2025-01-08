# pragma once
#include <Eigen/Dense>

using scalar_t = double;

using vector_2t = Eigen::Matrix<scalar_t, 2, 1>;
using vector_3t = Eigen::Matrix<scalar_t, 3, 1>;
using vector_4t = Eigen::Matrix<scalar_t, 4, 1>;
using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
using float_matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using int_matrix_t = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using int_vector_t = Eigen::Matrix<int, Eigen::Dynamic, 1>;