#include <stdio.h>
#include <cuda.h>

__global__ void helloFromGPU(void){
	printf("Hello from GPU! %d\n", threadIdx.x);
}

int main(void){
	printf("Hello! from CPU\n");
	
	helloFromGPU <<< 1,10 >>>();

	// error handling
	cudaError_t res = cudaDeviceReset(); // enumerated error-code type
	if(res == cudaSuccess){  // 
		printf("success!\n");
	}

}
