#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double cpuSecond(){
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ( (double)tp.tv_sec + (double)tp.tv_usec * 1e-6 );
}

#define CHECK(call){	\
	const cudaError_t error = call;	\
	if( error != cudaSuccess ){	\
		printf("Error: %s:%d\n", __FILE__, __LINE__);	\
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(1);	\
	}	\
}

void checkResult(float* hostRef, float* gpuRef, const int N){
	double epsilon = 1.0E-8;
	int match = 1;
	for( int idx = 0; idx != N; ++idx ){
		if(abs(gpuRef[idx] - hostRef[idx]) > epsilon){
			match = 0;
			printf("Arrays don't match.\n");
			printf("gpu: %5.2f host: %5.2f at current %d\n", gpuRef[idx], hostRef[idx], idx);
			break;
		}
	}
	if(match){ printf("Arrays match.\n"); }
	return;
}

void initializeData(float* ptr, const int size){
	time_t t;
	srand( (unsigned) time(&t) );
	for(int idx = 0; idx != size; ++idx ){
		ptr[idx] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void sumArraysOnHost(float* A, float* B, float* C, const int N){
	for(int idx = 0; idx != N; ++idx)
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnDevice(float* A, float* B, float* C){
		int idx = threadIdx.x + blockIdx.x * blockDim.x; // assuming 1D ?
		C[idx] = A[idx] + B[idx];
}


int main(int argc, char** argv){
	if(argc != 2){
		printf("Invalid arguments\n");
		exit(2);
	}


	printf("Starting...\n");
	double iStart, iElapse;

	
	int dev = 0; 
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("using device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));


	int nElem = 1<<24;
	printf("Vector size: %d\n", nElem);

	size_t nBytes = nElem * sizeof(float);

	float* h_A = (float *) malloc(nBytes);
	float* h_B = (float *) malloc(nBytes);
	float* hostRef = (float *) malloc(nBytes);
	float* gpuRef = (float *) malloc(nBytes);

	initializeData(h_A, nElem);
	initializeData(h_B, nElem);

	memset(hostRef, 0, nElem);
	memset(gpuRef, 0, nElem);

	float *d_A, *d_B, *d_C;
	CHECK(cudaMalloc((float**) &d_A, nBytes));
	CHECK(cudaMalloc((float**) &d_B, nBytes));
	CHECK(cudaMalloc((float**) &d_C, nBytes));

	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

	
	dim3 block(atoi(argv[1]));
	dim3 grid((nElem + block.x -1) / block.x);

	iStart = cpuSecond();
	sumArraysOnDevice<<< grid, block >>>(d_A, d_B, d_C);
	CHECK(cudaDeviceSynchronize());
	iElapse = cpuSecond() - iStart;
	printf("sumArraysOnDevice() <<< %d, %d >>> time: %5.6f sec\n", grid.x, block.x, iElapse);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost ));

	

	iStart = cpuSecond();
	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	iElapse = cpuSecond() - iStart;
	printf("sumArraysOnHost(): time: %5.6f sec\n", iElapse);

	checkResult(hostRef, gpuRef, nElem);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));

	return(0);
}
