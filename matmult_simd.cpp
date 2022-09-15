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
#include <immintrin.h>  // Intel intrinsics for SSE/AVX.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

typedef union _avxd {
  __m256d val;
  double arr[4];
} avxd;

typedef union _avxf {
    __m256 val;
    float arr[8];
} avxf;





Matrix<float> multiplyMatricesSIMD(Matrix<float> a, Matrix<float> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  // Create a matrix to store the results
     std::cout<<"SIMD float"<<std::endl;
    Matrix<float> result(a.rows, b.columns);
    //Create a vector to store the results of a row of a and a column of b


    //
    //u can transpos in another matrix and then multiply
    //Matrix<float> bT = b.transpose();
    //std::cout<<"bT"<<std::endl;
    //std::cout<<bT<<std::endl;

    //multiply matrices
    //use SIMD for fast matrix multiplication
    __m256 row, col, res;
    avxf avxf_res;
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < b.columns; j++){
            res=_mm256_setzero_ps();
            for(int k = 0; k < a.columns; k+=8){
                //store a row of a and a column of b in vectors
                row = _mm256_loadu_ps(&a(i, k));
                float bT[8]={0,0,0,0,0,0,0,0};
                for(int l = 0; l <(a.columns-k); l++){
                    bT[l] = b(k+l, j);
                }
                //load bT into a vector
                col = _mm256_loadu_ps(bT);
                //add vectors in res and multiply
                res = _mm256_add_ps(res, _mm256_mul_ps(row, col));
            }
            //store the result in a matrix
            _mm256_storeu_ps(avxf_res.arr, res);
            result(i, j) = avxf_res.arr[0] + avxf_res.arr[1] + avxf_res.arr[2] + avxf_res.arr[3] + avxf_res.arr[4] + avxf_res.arr[5] + avxf_res.arr[6] + avxf_res.arr[7];
        }
    }
        return result;
}



Matrix<double> multiplyMatricesSIMD(Matrix<double> a,
                                  Matrix<double> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  // Create a matrix to store the results

  std::cout<<"SIMD double"<<std::endl;
    Matrix<double> result(a.rows, b.columns);
    //Create a vector to store the results of a row of a and a column of b

//multiply matrices
    //use SIMD for fast matrix multiplication
    __m256d row, col, res;
    avxd avxd_res;
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < b.columns; j++){
            res=_mm256_setzero_pd();
            for(int k = 0; k < a.columns; k+=4){
                //store a row of a and a column of b in vectors
                if(k+4>a.columns) {
                    double aT[4] = {0, 0, 0, 0};
                    for (int l = 0; l < (a.columns - k); l++)
                        aT[l] = a(i, k + l);
                        row = _mm256_loadu_pd(aT);
                }
                    else {
                        row = _mm256_loadu_pd(&a(i, k));
                    }
                double bT[4]={0,0,0,0};
                for(int l = 0; l <(a.columns-k); l++){
                    bT[l] = b(k+l, j);
                }
                //load bT into a vector
                col = _mm256_loadu_pd(bT);
                //add vectors in res and multiply
                res = _mm256_add_pd(res, _mm256_mul_pd(row, col));
            }
            //store the result in a matrix
            _mm256_storeu_pd(avxd_res.arr, res);
            result(i, j) = avxd_res.arr[0] + avxd_res.arr[1] + avxd_res.arr[2] + avxd_res.arr[3];
        }
    }
        return result;

}
/*************************************/
#pragma GCC pop_options