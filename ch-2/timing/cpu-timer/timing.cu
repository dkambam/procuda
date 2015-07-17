#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>


#define CHECK(call)																\
{																				\
	cudaError_t error = call;													\
	if(error != cudaSuccess){													\
		printf("ERROR: %s:%d\n", __FILE__, __LINE__);							\
		printf("error_num: %d reason:%s\n", error, cudaGetErrorString(error));	\
		exit(1);																\
	}																			\
}

double cpuSecond(){
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ( (double)tp.tv_sec + (double)tp.tv_usec * 1e-6 );
}

__global__ void helloFromGPU(void){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Hello from GPU! %d\n", idx);
}

int main(void){
	int nElem = 6;
	dim3 block(6);
	dim3 grid( (nElem + block.x -1) / block.x );

	double iStart, iElapse;
	iStart = cpuSecond();
	helloFromGPU<<< grid, block >>>();
	CHECK(cudaDeviceSynchronize());
	iElapse = cpuSecond() - iStart;

	printf("Elapsed time: %5.6f sec\n", iElapse);

	return 0;

}
