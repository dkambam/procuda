#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) \
{ \
	cudaError_t error = call; \
	if(error != cudaSuccess){ \
		printf("ERROR: %s:%d\n", __FILE__, __LINE__); \
		printf("error: %d reason:%s\n", error,  cudaGetErrorString(error)); \
	} \
}


void initIntArray(int *ip, int size){
	for(int idx=0; idx<size; ++idx){
		ip[idx] = idx;
	}
}

void printMatrix(int* C, const int nx, const int ny){
	int *ic = C;
	printf("Matrix: (%d, %d)\n", ny, nx);
	for(int iy=0; iy<ny; ++iy){
		for(int ix=0; ix<nx; ++ix){
			printf("%d ", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
}

__global__ void printThreadInfo(int *A, const int nx, const int ny){
	// compute matrix index
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	// compute global index
	unsigned int idx = ix + iy * nx;

	printf("thread_id(%d, %d) block_id(%d, %d) coordinate(%d, %d) global_idx %2d ival %2d\n",
			threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);

}



int main(int argc, char **argv){
	printf("%s Starting ... \n", argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));


	int nx = 8;
	int ny = 6;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(int);


	int *h_A = (int*) malloc(nBytes);
	initIntArray(h_A, nxy);
	printMatrix(h_A, nx, ny);


	int *d_A;
	CHECK(cudaMalloc((void **)&d_A, nBytes));
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

	dim3 block(4,2);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	printThreadInfo<<< grid, block >>>(d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());

	free(h_A);
	CHECK(cudaFree(d_A));

	CHECK(cudaDeviceReset());

	return 0;
}
