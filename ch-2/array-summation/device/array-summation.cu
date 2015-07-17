#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

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
	printf("Starting...\n");

	int dev = 0;
	cudaSetDevice(dev);

	int nElem = 32;
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
	cudaMalloc((float**) &d_A, nBytes);
	cudaMalloc((float**) &d_B, nBytes);
	cudaMalloc((float**) &d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	
	dim3 block(1);
	dim3 grid(nElem / block.x);
	sumArraysOnDevice<<< grid, block >>>(d_A, d_B, d_C);
	printf("Execution configuration: <<< %d, %d >>> \n", grid.x, block.x);

	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost );

	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	checkResult(hostRef, gpuRef, nElem);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return(0);
}
