#include <stdio.h>
#include <cuda_runtime.h>

__global__ void checkIndex(void){
	printf("ThreadIdx: (%d,%d,%d) BlockIdx: (%d,%d,%d) BockDim: (%d,%d,%d) gridDim: (%d,%d,%d)\n", \
		    threadIdx.x,threadIdx.y,threadIdx.z,
		    blockIdx.x,blockIdx.y,blockIdx.z,
		    blockDim.x,blockDim.y,blockDim.z,
		    gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc, char** argv){
	int nElem = 6;
	dim3 block(3);
	dim3 grid((nElem + block.x) / block.x); // round it to the correct size

	printf("no. elements: %d\n", nElem);
	printf("no. blocks: (%d,%d,%d)\n", grid.x, grid.y, grid.z);
	printf("no. threads: (%d,%d,%d)\n", block.x, block.y, block.z);

	checkIndex <<< grid, block >>>();
	cudaDeviceReset();
}
