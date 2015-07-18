#include <stdio.h>
#include <cuda_runtime.h>


#define CHECK(call){	\
	const cudaError_t error = call;	\
	if( error != cudaSuccess ){	\
		printf("Error: %s:%d\n", __FILE__, __LINE__);	\
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(1);	\
	}	\
}


__global__ void helloFromGPU(void){
	printf("Hello from GPU! %d\n", threadIdx.x);
}

int main(void){
	printf("Hello from CPU!\n");
	
	helloFromGPU <<< 1,10 >>>();
	CHECK(cudaDeviceReset());
}
