/* 
Demo for the following: 
	cudaError_t
	cudaGetErrorString
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(void){
	printf("Hello from GPU! %d\n", threadIdx.x);
}

int main(void){
	printf("Hello from CPU!\n");
	
	helloFromGPU <<< 1,10 >>>();

	// error handling
	cudaError_t res; // enumerated error-code type
	res = cudaDeviceReset(); 
	if(res != cudaSuccess){
		printf("%s\n", cudaGetErrorString(res)); // get description of error
	}else{
		printf("Success!\n");
	}
}
