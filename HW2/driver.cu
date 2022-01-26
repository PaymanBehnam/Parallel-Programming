// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
#include<stdio.h>
#include "cuda.h"
#include<string.h>
#include<stdlib.h>

#define BLOCK_SIZE 512 //@@ You can change this

char *inputFile,*outputFile;
void _errorCheck(cudaError_t e){
	if(e != cudaSuccess){
		printf("Failed to run statement \n");
	}
}


// simple sequential add in global memory
__global__ void total0 (float *input, float *output, int len) { // Morre sequential: One thread(threadIdx.x == 0)  each block do addition
  //@@ Compute reduction for a segment of the input vector
  int start =  blockDim.x * blockIdx.x ;
  int end;
  float sumation = 0;

  if (len < start+blockDim.x)
    end = len;
  else
    end = start+blockDim.x;


  if (threadIdx.x == 0) {
    for (int i = start; i < end; i+=1)
      sumation += input[i];
    output[blockIdx.x] = sumation;
  }

}

// tree reduction in shared memory
__global__ void total1 (float *input, float *output, int len) {//Reduction#1 with shared Memory
	//@@ Compute reduction for a segment of the input vector
  			unsigned int tx = threadIdx.x;
  			unsigned int idt = blockIdx.x*blockDim.x + threadIdx.x;

			__shared__ float  PartialSum [BLOCK_SIZE];

			PartialSum[tx] = (idt < len) ? input[idt] : 0;


			 for (unsigned int stride=1; stride < blockDim.x; stride*=2){
				 __syncthreads ();
				 if (tx% (2*stride)==0){
					 PartialSum[tx]+=PartialSum[tx+ stride];
				 }
			}
			if (tx == 0) output[blockIdx.x] = PartialSum[0];
}
__global__ void total2(float *input, float *output, int len) { //Reduction#2 with shared Memory
  //@@ Compute reduction for a segment of the input vector
 				 unsigned int tx = threadIdx.x;
  				unsigned int idt = blockIdx.x*blockDim.x + threadIdx.x;

				__shared__ float  PartialSum[BLOCK_SIZE];;

  				PartialSum[tx] = (idt < len) ? input[idt] : 0;


			 	for (unsigned int stride= blockDim.x>>1; stride>0 ; stride>>=1){
				 __syncthreads ();
				 if (tx < stride){
					 PartialSum[tx]+=PartialSum[tx+stride];
				 }
			}
			if (tx == 0) output[blockIdx.x] = PartialSum[0];
}

__global__ void total3 (float *input, float *output, int len) { //Reduction#1 with global Memory
	//@@ Compute reduction for a segment of the input vector
  			unsigned int tx = threadIdx.x;
  			unsigned int idt = blockIdx.x*blockDim.x + threadIdx.x;

			//__shared__ float  PartialSum [BLOCK_SIZE];

			//PartialSum[tx] = (idt < len) ? input[idt] : 0;


			 for (unsigned int stride=1; stride < blockDim.x; stride*=2){
				 __syncthreads ();
				 if (tx% (2*stride)==0){
					 if (idt < len){
				 		input[idt]+=input[idt+stride];
				 	}
				 }
			}
			if (tx == 0) output[blockIdx.x] = input[idt];
}

__global__ void total4(float *input, float *output, int len) { //Reduction#2 with global Memory
  //@@ Compute reduction for a segment of the input vector
	unsigned int tx = threadIdx.x;
	 unsigned int idt = blockIdx.x*blockDim.x + threadIdx.x;

   //float  PartialSum[BLOCK_SIZE];

	 //PartialSum[tx] = (idt < len) ? input[idt] : 0;

 for (unsigned int stride= blockDim.x>>1; stride >0 ; stride>>=1){
	__syncthreads ();
	if (idt < len){
		input[idt]+=input[idt+stride];
	}
}

if (tx == 0) output[blockIdx.x] = input[idt];
}
__global__ void total4_notworking(float *input, float *output, int len) {
	//@@ Compute reduction for a segment of the input vector
				 unsigned int tx = threadIdx.x;
					unsigned int idt = blockIdx.x*blockDim.x + threadIdx.x;

				 float  PartialSum[BLOCK_SIZE];

					PartialSum[tx] = (idt < len) ? input[idt] : 0;


				for (unsigned int stride= blockDim.x>>1; stride >0 ; stride>>= 1){
				 __syncthreads ();
				 if (tx < stride){
					 PartialSum[tx]+=PartialSum[tx+stride];
				 }
			}
			if (tx == 0) output[blockIdx.x] = PartialSum[0];
}

__global__ void total5 (float *input, float *output, int len) {//Atomic
  __shared__ float sumation;
  int idt = blockDim.x*blockIdx.x + threadIdx.x;
  if (threadIdx.x == 0)
    sumation = 0;
  __syncthreads ();

  if (idt < len)
    atomicAdd (&sumation, input[idt]);

  __syncthreads ();

  if (threadIdx.x == 0)
    output[blockIdx.x] = sumation;
}


void parseInput(int argc, char **argv){
	if(argc < 2){
		printf("Not enough arguments\n");
		printf("Usage: reduction -i inputFile -o outputFile\n");
		exit(1);
	}
	int i=1;
	while(i<argc){
		if(!strcmp(argv[i],"-i")){
			++i;
			inputFile = argv[i];
		}
		else if(!strcmp(argv[i],"-o")){
			++i;
			outputFile = argv[i];
		}
		else{
			printf("Wrong input");
			exit(1);
		}
		i++;
	}
}
void getSize(int &size, char *file){
	FILE *fp;
	fp = fopen(file,"r");
	if(fp == NULL){
		perror("Error opening File\n");
		exit(1);
	}

	if(fscanf(fp,"%d",&size)==EOF){
		printf("Error reading file\n");
		exit(1);
	}
	fclose(fp);
}
void readFromFile(int &size,float *v, char *file){
	FILE *fp;
	fp = fopen(file,"r");
	if(fp == NULL){
		printf("Error opening File %s\n",file);
		exit(1);
	}

	if(fscanf(fp,"%d",&size)==EOF){
		printf("Error reading file\n");
		exit(1);
	}
	int i=0;
	float t;
	while(i < size){
		if(fscanf(fp,"%f",&t)==EOF){
			printf("Error reading file\n");
			exit(1);
		}
		v[i++]=t;
	}
	fclose(fp);


}

int main(int argc, char **argv) {
  int ii;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list
  float *solution;

  // Read arguments and input files
  parseInput(argc,argv);

  // Read input from data
  getSize(numInputElements,inputFile);
  hostInput = (float*) malloc(numInputElements*sizeof(float));

  readFromFile(numInputElements,hostInput,inputFile);

  int opsz;
  getSize(opsz,outputFile);
  solution = (float*) malloc(opsz*sizeof(float));

  readFromFile(opsz,solution,outputFile);

  //@@ You can change this, but assumes output element per block
  numOutputElements = numInputElements / (BLOCK_SIZE);
  // numOutputElements = 1;

  if (numInputElements % (BLOCK_SIZE)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  //@@ Allocate GPU memory here
  cudaMalloc ((void**)&deviceInput, numInputElements*(sizeof(float)));
  cudaMalloc ((void**)&deviceOutput, numOutputElements*(sizeof(float)));

  //@@ Copy memory to the GPU here
  cudaMemcpy (deviceInput, hostInput, numInputElements*(sizeof(float)), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 thread_num (BLOCK_SIZE, 1, 1);
  dim3 block_num  (numOutputElements, 1, 1);

  // Initialize timer
  cudaEvent_t start,stop;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //@@ Launch the GPU Kernel here, you may want multiple implementations to compare

  total5 <<<block_num, thread_num>>> (deviceInput, deviceOutput, numInputElements);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy (hostOutput, deviceOutput, numOutputElements*sizeof(float), cudaMemcpyDeviceToHost);


  /*
   * Reduce any remaining output on host
   */
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time,start, stop);

  //@@ Free the GPU memory here
  cudaFree (deviceInput);
  cudaFree (deviceOutput);

  if(solution[0] == hostOutput[0]){
	printf("The operation was successful, time = %2.6f\n",elapsed_time);
  }
  else{
	printf("The operation failed \n");
  }
  free(hostInput);
  free(hostOutput);

  return 0;
}
