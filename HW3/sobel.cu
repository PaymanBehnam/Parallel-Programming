

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>


#define DEFAULT_THRESHOLD  4000

#define DEFAULT_FILENAME "BWstop-sign.ppm"

#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)
#define ThreadNumberPB 1024

template<typename T>
void _cudaCheck(T e, const char* func, const char* call, const int line){
  if(e != cudaSuccess){
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

//////////////////////
///        START_GPU
//////////////////////
__global__ void sobel_gpu(unsigned int *imgin, unsigned int *imgout, int width, int height) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int sum1, sum2, magnitude;
    int offset = y*width + x;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        sum1 = (-1* imgin[(y-1)*width + (x-1)]) + (-2*imgin[y*width+(x-1)]) + (-1*imgin[(y+1)*width+(x-1)]) +
             (imgin[(y-1)*width + (x+1)]) + ( 2*imgin[y*width+(x+1)]) + (imgin[(y+1)*width+(x+1)]);
        sum2 = (imgin[(y-1)*width + (x-1)]) + ( 2*imgin[(y-1)*width+x]) + (imgin[(y-1)*width+(x+1)]) +
             (-1* imgin[(y+1)*width + (x-1)]) + (-2*imgin[(y+1)*width+x]) + (-1*imgin[(y+1)*width+(x+1)]);
         magnitude =  (sum1*sum1) + (sum2*sum2);
        if (magnitude > DEFAULT_THRESHOLD )
          imgout[offset] = 255;
        else
          imgout[offset] = 0;
    }

}
//////////////////////
///        END_GPU
//////////////////////


unsigned int *read_ppm( char *filename, int & xsize, int & ysize, int & maxval ){

  if ( !filename || filename[0] == '\0') {
    fprintf(stderr, "read_ppm but no file name\n");
    return NULL;  // fail
  }

  fprintf(stderr, "read_ppm( %s )\n", filename);
  int fd = open( filename, O_RDONLY);
  if (fd == -1)
    {
      fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
      return NULL; // fail

    }

  char chars[1024];
  int num = read(fd, chars, 1000);

  if (chars[0] != 'P' || chars[1] != '6')
    {
      fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
      return NULL;
    }

  unsigned int width, height, maxvalue;


  char *ptr = chars+3; // P 6 newline
  if (*ptr == '#') // comment line!
    {
      ptr = 1 + strstr(ptr, "\n");
    }

  num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
  fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);
  xsize = width;
  ysize = height;
  maxval = maxvalue;

  unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
  if (!pic) {
    fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
    return NULL; // fail but return
  }

  // allocate buffer to read the rest of the file into
  int bufsize =  3 * width * height * sizeof(unsigned char);
  if (maxval > 255) bufsize *= 2;
  unsigned char *buf = (unsigned char *)malloc( bufsize );
  if (!buf) {
    fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
    return NULL; // fail but return
  }





  // TODO really read
  char duh[80];
  char *line = chars;

  // find the start of the pixel data.   no doubt stupid
  sprintf(duh, "%d\0", xsize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", ysize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", maxval);
  line = strstr(line, duh);


  fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
  line += strlen(duh) + 1;

  long offset = line - chars;
  lseek(fd, offset, SEEK_SET); // move to the correct offset
  long numread = read(fd, buf, bufsize);
  fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize);

  close(fd);


  int pixels = xsize * ysize;
  for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel



  return pic; // success
}












void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic)
{

  FILE *fp;

  fp = fopen(filename, "w");
  if (!fp)
    {
      fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
      exit(-1);
    }
  //int x,y;


  fprintf(fp, "P6\n");
  fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);

  int numpix = xsize * ysize;
  for (int i=0; i<numpix; i++) {
    unsigned char uc = (unsigned char) pic[i];
    fprintf(fp, "%c%c%c", uc, uc, uc);
  }
  fclose(fp);

}
//////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
main( int argc, char **argv )
{

  int thresh = DEFAULT_THRESHOLD;
  char *filename;
  filename = strdup( DEFAULT_FILENAME);

  if (argc > 1) {
    if (argc == 3)  { // filename AND threshold
      filename = strdup( argv[1]);
       thresh = atoi( argv[2] );
    }
    if (argc == 2) { // default file but specified threshhold

      thresh = atoi( argv[1] );
    }

    fprintf(stderr, "file %s    threshold %d\n", filename, thresh);
  }


  int xsize, ysize, maxval;
  unsigned int *pic = read_ppm( filename, xsize, ysize, maxval );


  int numbytes =  xsize * ysize * 3 * sizeof( int );
  int *result = (int *) malloc( numbytes );
  int *resultgpu = (int *) malloc( numbytes );
  if (!result) {
    fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
    exit(-1); // fail
  }

  int i, j, magnitude, sum1, sum2;
  int *out = result;

  for (int col=0; col<ysize; col++) {
    for (int row=0; row<xsize; row++) {
      *out++ = 0;
    }
  }

  for (i = 1;  i < ysize - 1; i++) {
    for (j = 1; j < xsize -1; j++) {

      int offset = i*xsize + j;

      sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ]
        + 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
        +     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];

      sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
            - pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];

      magnitude =  sum1*sum1 + sum2*sum2;

      if (magnitude > thresh)
        result[offset] = 255;
      else
        result[offset] = 0;
    }
  }

  write_ppm( "resultCPU.ppm", xsize, ysize, 255, result);

  fprintf(stderr, "sobel CPU done\n");





  /////////////
  unsigned int *d_pic, *d_result;

  cudaMalloc( (void**) &d_pic, numbytes);

  cudaMalloc( (void**) &d_result, numbytes);

  /** Transfer over the memory from host to device and memset the sobel array to 0s **/
  cudaMemcpy(d_pic, pic, numbytes/3, cudaMemcpyHostToDevice);
  //cudaMemset(gpu_sobel, 0, (origImg.width*origImg.height));


  // Timing using cudaEvent
  cudaEvent_t start, stop;
  float et;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));

	//num_threads = (sz+nelem-1)/nelem;
  dim3 numThreads(32, 32, 1);
  dim3 numBlocks(ceil(xsize/32), ceil(ysize/32), 1);


  // Time event start
  cudaCheck(cudaEventRecord(start));

  {
    // TODO Invoke the kernel code here
    sobel_gpu<<<numBlocks, numThreads>>>(d_pic, d_result, xsize, ysize);


  }

  cudaCheck(cudaGetLastError());

  // Time event end
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&et, start, stop));
  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));

  fprintf(stderr, "GPUTIME:");
  fprintf(stderr,"\t%0.3f\n", et);

  /** Copy data back to CPU from GPU **/
  cudaMemcpy(resultgpu, d_result, numbytes, cudaMemcpyDeviceToHost);
  cudaFree (d_pic);
  cudaFree (d_result);


  write_ppm( "resultgpu.ppm", xsize, ysize, 255, resultgpu);

  fprintf(stderr, "sobel gpu done\n");
/////////////////////////
  int index1, index2;
  for (index1 = 1; index1 < ysize-1; index1++) {
    for (index2 = 1; index2 < xsize-1; index2++) {
    if (fabsf(result[index1*xsize+index2] - resultgpu[index1*xsize+index2]) > 1e-5) {
      fprintf(stderr, "comparsion fails\n");
      return 0;
    }

  }
}
  fprintf(stderr, "comparsion passed\n");
  return 1;
}
