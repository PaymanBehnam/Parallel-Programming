#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<random>

#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)
#define THREAD_DIM 1024

template<typename T>
void _cudaCheck(T e, const char* func, const char* call, const int line){
  if(e != cudaSuccess){
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// Number of elements worked by one thread
long nelem;

// TODO kernel code here: each threads work on nelem - elements in a pair of sz-long vector
__global__ void add_krnl(float *x, float *y, long sz, long nelem) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = (sz+nelem-1)/nelem;
	int j, ind;


	for (j = 0; j < nelem; j+=1) {
		ind = j*num_threads + id; // cyclic
	  // ind = j + id*nelem; // adjacent
		if ((ind < sz) & (id < num_threads)) {
			x[ind] = x[ind] + y[ind];
		}
	}
}

long func_add(float *x, float *y, long sz) {
  long i;
	float *d_x_array, *d_y_array;
	int num_threads, num_blocks;

  /* TODO: Make call to GPU kernel to compute results on GPU into d_x */
  /* STEP1: Write thread program to compute nelem elements per thread */
  /* STEP2: Invoke thread program for sz */
  /* STEP3: Using count6 as example, allocate GPU input/output data */
  /* STEP3: Using count6 as example, initialize GPU input/output data */
  cudaMalloc((void **) &d_x_array,sz*sizeof(float));
  cudaMalloc((void **) &d_y_array,sz*sizeof(float));
  cudaMemcpy(d_x_array, x, sz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_array, y, sz*sizeof(float),cudaMemcpyHostToDevice);

  // Timing using cudaEvent
  cudaEvent_t start, stop;
  float et;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));

	num_threads = (sz+nelem-1)/nelem;
	num_blocks = (sz+THREAD_DIM-1)/THREAD_DIM;

  // Time event start
  cudaCheck(cudaEventRecord(start));
  
  {
    // TODO Invoke the kernel code here
		add_krnl <<<num_blocks, THREAD_DIM>>> (d_x_array, d_y_array, sz, nelem);
		// add_krnl <<<1, num_threads, 0>>> (d_x_array, d_y_array, sz, nelem);
		// add_krnl <<<1, 10, 0>>> (d_x_array, d_y_array, sz, nelem);
  }

  cudaCheck(cudaGetLastError());

  // Time event end
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&et, start, stop));
  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));

  printf("\t%0.3f", et);

  // TODO Copy data back to d_x and free GPU memory
  float * d_x = (float *) malloc(sz * sizeof(float));

  cudaMemcpy(d_x, d_x_array, sz*sizeof(float),cudaMemcpyDeviceToHost);

  /* CPU Calculation */
  for (i = 0; i < sz; i++) {
		// printf ("%f+%f=%f\n", x[i], y[i], d_x[i]);
    x[i] += y[i];
	}

  /* Compare CPU and GPU output to see if it is within error tolerance */
  for (i = 0; i < sz; i++) {
    if (fabsf(d_x[i] - x[i]) > 1e-5) {
      free(d_x);
      return 0;
    }
  }

	cudaFree (d_x_array);
	cudaFree (d_y_array);

  free(d_x);
  return 1;
}


int main(int argc, char **argv) {
  float *a, *b;
  long j;
  long i;

  std::random_device rd;
  std::mt19937_64 mt(rd());
  std::uniform_real_distribution<float> u(0, 1);

  // Print title
  printf("sz");
  for (nelem = 1; nelem < 513; nelem *= 2)
    printf("\t%d", nelem);
  printf("\n");

  for (j = 10; j <= 1000000000; j *= 10) {
    a = (float *) malloc(sizeof(float) * j);
    b = (float *) malloc(sizeof(float) * j);

    /* Initialize with random number generator */
    for (i = 0; i < j; i++) {
      a[i] = u(mt);
      b[i] = u(mt);
    }

    printf("%d", j);

    for (nelem = 1; nelem < 513; nelem *= 2)
    // for (nelem = 2; nelem < 4; nelem *= 2)
      if (!func_add(a, b, j))
        printf("failed to add\n");

    printf("\n");

    free(a);
    free(b);
  }

  return 0;
}
