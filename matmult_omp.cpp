// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acsmatmult/matmult.h"
#include <omp.h>  // OpenMP support.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

Matrix<float> multiplyMatricesOMP(Matrix<float> a,
                                  Matrix<float> b,
                                  int num_threads) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */
        // Create a matrix to store the res
        Matrix<float> res(a.rows, b.columns);

        // Set the number of threads
        omp_set_num_threads(num_threads);
        //very fast matrix multiplication with openmp
        #pragma omp parallel for
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < b.columns; j++){
                float sum = 0;
        #pragma omp parallel for reduction(+:sum)
                for(int k = 0; k < a.columns; k++){
                    sum += a(i, k) * b(k, j);
                }
                res(i, j) = sum;
            }
        }
        return res;
}

Matrix<double> multiplyMatricesOMP(Matrix<double> a,
                                   Matrix<double> b,
                                   int num_threads) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */

          // Create a matrix to store the res
            Matrix<double> res(a.rows, b.columns);

            // Set the number of threads
            omp_set_num_threads(num_threads);
            //very fast matrix multiplication with openmp
            #pragma omp parallel for
            for(int i = 0; i < a.rows; i++){
                for(int j = 0; j < b.columns; j++){
                    double sum = 0;
            #pragma omp parallel for reduction(+:sum)
                    for(int k = 0; k < a.columns; k++){
                        sum += a(i, k) * b(k, j);
                    }
                    res(i, j) = sum;
                }
            }
            return res;
          //a second method to do the code
        #pragma omp parallel for
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < b.columns; j++){
                double sum = 0;
                for(int k = 0; k < a.columns; k++){
                    sum += a(i, k) * b(k, j);
                }
                res(i, j) = sum;
            }
        }

}
#pragma GCC pop_options