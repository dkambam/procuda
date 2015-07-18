#include <stdio.h>
#include <cuda.h> // include cuda-api header

__global__ void helloFromGPU(void){
	printf("Hello from GPU! %d\n", threadIdx.x);	// accessing thread id
}

int main(void){
	printf("Hello! from CPU\n");
	
	helloFromGPU <<< 1,10 >>>();
	cudaDeviceReset();
}
