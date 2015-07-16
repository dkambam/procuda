#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sumArraysOnHost(float* A, float* B, float* C, const int N){
	for(int idx = 0; idx != N; ++idx)
		C[idx] = A[idx] + B[idx];
}

void initializeData(float* ptr, const int size){
	time_t t;
	srand( (unsigned int) time(&t) );
	for(int idx = 0; idx != size; ++idx ){
		ptr[idx] = (float)(rand() & 0xFF) / 10.0f;
	}
}

int main(void){
	int nElem = 1024;
	size_t nBytes = nElem * sizeof(float);

	float* h_A = (float *) malloc(nBytes);
	float* h_B = (float *) malloc(nBytes);
	float* h_C = (float *) malloc(nBytes);

	initializeData(h_A, nElem);
	initializeData(h_B, nElem);

	sumArraysOnHost(h_A, h_B, h_C, nElem);

	free(h_A);
	free(h_B);
	free(h_C);

	return(0);
}
