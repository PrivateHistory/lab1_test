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

// OpenCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "acsmatmult/matmult.h"

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")

/*************************************/


typedef union {
    Matrix<float> matrix;
    float *data;
} matrix_float;


// Macro to check clFunction outputs.
// Throw an error if not successful, to make debugging easier.
#define CHECK(err) if (err != CL_SUCCESS) { \
  throw std::runtime_error("OpenCL error: " + std::to_string(err) + \
  " in " + __FILE__ + " line " + std::to_string(__LINE__) ); \
}

///@brief A little enum class to help us parse clDeviceInfo
enum class ClInfoType {
    CHAR, SIZE_T, //... add your own info types
};

/// @brief Function to discover OpenCL devices and print some info on stdout.
static std::vector<cl_device_id> discoverDevices(cl_platform_id platform_id) {
    std::vector<cl_device_id> devices;
    // Discover devices for each platform
    cl_uint num_devices = 0;
    // Get number of devices of this type, we will only discover GPU devices for now.
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceIDs.html
    int err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);

    std::cout << "\tDevices: " << num_devices << std::endl;

    if ((err != CL_DEVICE_NOT_FOUND) || (num_devices != 0)) {
        // Get the devices of this type and insert them into the final list
        std::vector<cl_device_id> platform_type_devices(num_devices);
        CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, platform_type_devices.data(), &num_devices));
        // Insert the found devices into the final result
        devices.insert(std::end(devices), std::begin(platform_type_devices), std::end(platform_type_devices));

        // Many infos exist for devices. Also see:
        // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
        //
        // DISCLAIMER: IT IS HIGHLY RECOMMENDED TO DISCOVER SOME MORE STUFF ABOUT YOUR DEVICE WHEN YOU ARE GOING TO
        // USE IT MORE INTENSELY

        for (auto platform_type_device : platform_type_devices) {
            std::vector<cl_device_info> info_queries = {CL_DEVICE_NAME, CL_DEVICE_MAX_WORK_GROUP_SIZE};
            std::vector<ClInfoType> info_types = {ClInfoType::CHAR, ClInfoType::SIZE_T};
            size_t info_size = 0;
            for (unsigned int i = 0; i < info_queries.size(); i++) {
                // Get the query size
                CHECK(clGetDeviceInfo(platform_type_device, info_queries[i], 0, nullptr, &info_size));
                auto query = new char[info_size];
                CHECK(clGetDeviceInfo(platform_type_device, info_queries[i], info_size, query, &info_size));
                switch (info_types[i]) {
                    case ClInfoType::SIZE_T: std::cout << *reinterpret_cast<size_t *>(query) << std::endl;
                        break;
                    default:std::cout << query << std::endl;
                        break;
                }
                delete[] query;

            }
        }
    }
    return devices;
}

/// @brief Function to discover OpenCL platforms and print some info on stdout.
static std::vector<cl_platform_id> discoverPlatforms() {
    cl_uint num_platforms = 0;

    // Obtain the number of OpenCL platforms
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetPlatformIDs.html
    CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));

    // OpenCL sometimes outputs some stuff on cerr. Flush this stuff from the stream.
    std::cerr.flush();

    std::cout << "Found " << num_platforms << " OpenCL platform(s)." << std::endl;

    // Create an array to hold platform IDs.
    auto platform_ids = std::vector<cl_platform_id>(num_platforms);

    // Get the actual platform IDs
    CHECK(clGetPlatformIDs(num_platforms, platform_ids.data(), &num_platforms));

    // Identify the platform info that we would like to discover (more infos exist, but they are not interesting for us)
    const std::vector<cl_platform_info> platform_queries = {CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION};

    // Iterate over all platforms
    for (unsigned int p = 0; p < num_platforms; p++) {
        std::cout << "Platform " << p << std::endl;

        // Iterate over all platform infos we want to inquire
        for (auto platform_query : platform_queries) {
            size_t query_size = 0;

            // Get the current platform info length
            CHECK(clGetPlatformInfo(platform_ids[p], platform_query, 0, nullptr, &query_size));
            auto query = new char[query_size];

            // Get the actual info
            CHECK(clGetPlatformInfo(platform_ids[p], platform_query, query_size, query, &query_size));

            std::cout << '\t' << query << std::endl;

            delete[] query;
        }
    }

    return platform_ids;
}

Matrix<float> multiplyMatricesOCL(Matrix<float> a,Matrix<float> b){

    //Make a matrix to hold the result
    Matrix<float> result(a.rows,b.columns);


    int err;      // error code returned from api calls

    //Discover the platforms
    auto platforms = discoverPlatforms();

    if(platforms.empty()){
        throw std::runtime_error("No OpenCL platforms detected.");
    }

    //Discover the devices
    auto devices = discoverDevices(platforms[0]);

    if(devices.empty()){
        throw std::runtime_error("No OpenCL devices detected.");
    }

    //Create a context
    cl_context context = clCreateContext(nullptr,1,&devices[0],nullptr,nullptr,&err);



    //Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context,devices[0],0,nullptr);
    //Create a program
    auto kernelSource="__kernel void multiplyMatrices(__global float *a, __global float *b, __global float *c, int rows, int columns, int columnsA){"
                      "int i = get_global_id(0);"
                      "int j = get_global_id(1);"
                      "float sum = 0;"
                      "for(int k=0;k<columnsA;k++){"
                      "sum += a[i*columnsA+k]*b[k*columns+j];"
                        "}"
                      "c[i*columns+j] = sum;"
                      "}";

    cl_program program = clCreateProgramWithSource(context,1,(const char**)&kernelSource,nullptr,&err);
    //Build the program
    CHECK(clBuildProgram(program,1,&devices[0],nullptr,nullptr,nullptr));
    //Create the kernel
    cl_kernel kernel = clCreateKernel(program,"multiplyMatrices",nullptr);
    //Create the buffers
    cl_mem aBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY,a.rows*a.columns*sizeof(float),nullptr,nullptr);
    cl_mem bBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY,b.rows*b.columns*sizeof(float),nullptr,nullptr);
    cl_mem cBuffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,result.rows*result.columns*sizeof(float),nullptr,nullptr);
    //Write the data to the buffers
    //Convert from Matrix q to float*
    float* a_data = new float[a.rows*a.columns];
    float* b_data = new float[b.rows*b.columns];
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.columns; j++){
            a_data[i*a.columns+j] = a(i,j);
        }
    }
    for(int i = 0; i < b.rows; i++){
        for(int j = 0; j < b.columns; j++){
            b_data[i*b.columns+j] = b(i,j);
        }
    }

    CHECK(clEnqueueWriteBuffer(queue,aBuffer,CL_TRUE,0,a.rows*a.columns*sizeof(float),a_data,0,nullptr,nullptr));
    CHECK(clEnqueueWriteBuffer(queue,bBuffer,CL_TRUE,0,b.rows*b.columns*sizeof(float),b_data,0,nullptr,nullptr));
    //Set the kernel arguments
    CHECK(clSetKernelArg(kernel,0,sizeof(cl_mem),&aBuffer));
    CHECK(clSetKernelArg(kernel,1,sizeof(cl_mem),&bBuffer));
    CHECK(clSetKernelArg(kernel,2,sizeof(cl_mem),&cBuffer));
    CHECK(clSetKernelArg(kernel,3,sizeof(int),&a.rows));
    CHECK(clSetKernelArg(kernel,4,sizeof(int),&b.columns));
    CHECK(clSetKernelArg(kernel,5,sizeof(int),&a.columns));
    //Execute the kernel
    size_t globalWorkSize[2] = {a.rows,b.columns};
    CHECK(clEnqueueNDRangeKernel(queue,kernel,2,nullptr,globalWorkSize,nullptr,0,nullptr,nullptr));
    //Read the result
    float *result_data = new float[result.rows*result.columns];
    //make the data in the buffer result data 0
    for(int i = 0; i < result.rows*result.columns; i++){
        result_data[i] = 0;
    }
    CHECK(clEnqueueReadBuffer(queue,cBuffer,CL_TRUE,0,result.rows*result.columns*sizeof(float),result_data,0,nullptr,nullptr));
    //Release the buffers
    CHECK(clReleaseMemObject(aBuffer));
    CHECK(clReleaseMemObject(bBuffer));
    CHECK(clReleaseMemObject(cBuffer));
    //Release the kernel
    CHECK(clReleaseKernel(kernel));
    //Release the program
    CHECK(clReleaseProgram(program));
    //Release the command queue
    CHECK(clReleaseCommandQueue(queue));
    //Release the context
    CHECK(clReleaseContext(context));

//Convert from float* to Matrix
    for(int i = 0; i < result.rows; i++){
        for(int j = 0; j < result.columns; j++){
            result(i,j) = result_data[i*result.columns+j];
        }
    }

    return result;

}

Matrix<double> multiplyMatricesOCL(Matrix<double> a,Matrix<double> b) {

        //Make a matrix to hold the result
        Matrix<double> result(a.rows,b.columns);


        int err;      // error code returned from api calls

        //Discover the platforms
        auto platforms = discoverPlatforms();

        if(platforms.empty()){
            throw std::runtime_error("No OpenCL platforms detected.");
        }

        //Discover the devices
        auto devices = discoverDevices(platforms[0]);

        if(devices.empty()){
            throw std::runtime_error("No OpenCL devices detected.");
        }

        //Create a context
        cl_context context = clCreateContext(nullptr,1,&devices[0],nullptr,nullptr,&err);



        //Create a command queue
        cl_command_queue queue = clCreateCommandQueue(context,devices[0],0,nullptr);
        //Create a program
        auto kernelSource="__kernel void multiplyMatrices(__global double *a, __global double *b, __global double *c, int rows, int columns, int columnsA){"
                          "int i = get_global_id(0);"
                          "int j = get_global_id(1);"
                          "double sum = 0;"
                          "for(int k = 0; k < columnsA; k++){"
                          "sum += a[i*columnsA+k]*b[k*columns+j];"
                          "}"
                          "c[i*columns+j] = sum;"
                          "}";

        cl_program program = clCreateProgramWithSource(context,1,(const char**)&kernelSource,nullptr,&err);
        //Build the program
        CHECK(clBuildProgram(program,1,&devices[0],nullptr,nullptr,nullptr));
        //Create the kernel
        cl_kernel kernel = clCreateKernel(program,"multiplyMatrices",nullptr);
        //Create the buffers
        cl_mem aBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY,a.rows*a.columns*sizeof(double),nullptr,nullptr);
        cl_mem bBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY,b.rows*b.columns*sizeof(double),nullptr,nullptr);
        cl_mem cBuffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,result.rows*result.columns*sizeof(double),nullptr,nullptr);
        //Write the data to the buffers
        //Convert from Matrix q to float*
        double* a_data = new double[a.rows*a.columns];
        double* b_data = new double[b.rows*b.columns];
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < a.columns; j++){
                a_data[i*a.columns+j] = a(i,j);
            }
        }
        for(int i = 0; i < b.rows; i++){
            for(int j = 0; j < b.columns; j++){
                b_data[i*b.columns+j] = b(i,j);
            }
        }

        CHECK(clEnqueueWriteBuffer(queue,aBuffer,CL_TRUE,0,a.rows*a.columns*sizeof(double),a_data,0,nullptr,nullptr));
        CHECK(clEnqueueWriteBuffer(queue,bBuffer,CL_TRUE,0,b.rows*b.columns*sizeof(double),b_data,0,nullptr,nullptr));
        //Set the kernel arguments
        CHECK(clSetKernelArg(kernel,0,sizeof(cl_mem),&aBuffer));
        CHECK(clSetKernelArg(kernel,1,sizeof(cl_mem),&bBuffer));
        CHECK(clSetKernelArg(kernel,2,sizeof(cl_mem),&cBuffer));
        CHECK(clSetKernelArg(kernel,3,sizeof(int),&a.rows));
        CHECK(clSetKernelArg(kernel,4,sizeof(int),&b.columns));
        CHECK(clSetKernelArg(kernel,5,sizeof(int),&a.columns));

        //Execute the kernel
        size_t globalWorkSize[2] = {a.rows,b.columns};
        CHECK(clEnqueueNDRangeKernel(queue,kernel,2,nullptr,globalWorkSize,nullptr,0,nullptr,nullptr));

        //Read the result
        double *result_data = new double[result.rows*result.columns];
        CHECK(clEnqueueReadBuffer(queue,cBuffer,CL_TRUE,0,result.rows*result.columns*sizeof(double),result_data,0,nullptr,nullptr));
        //Release the buffers
        CHECK(clReleaseMemObject(aBuffer));
        CHECK(clReleaseMemObject(bBuffer));
        CHECK(clReleaseMemObject(cBuffer));
        //Release the kernel
        CHECK(clReleaseKernel(kernel));
        //Release the program
        CHECK(clReleaseProgram(program));
        //Release the command queue
        CHECK(clReleaseCommandQueue(queue));
        //Release the context
        CHECK(clReleaseContext(context));

//Convert from double* to Matrix
        for(int i = 0; i < result.rows; i++){
            for(int j = 0; j < result.columns; j++){
                result(i,j) = result_data[i*result.columns+j];
            }
        }

        return result;

    }

/*************************************/
#pragma GCC pop_options
/*************************************/