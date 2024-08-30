#include <iostream>

#include <nvtx3/nvToolsExt.h>

#define IDIVUP(a, b) (((a) + (b) - 1) / (b))

////////////////////////////////////////////////////////////////////////////////
///
///
///
///
////////////////////////////////////////////////////////////////////////////////
template< typename T, int WINDOW_SIZE > 
// __launch_bounds__(256, 2)
__global__ void windowAverage(T *pMatA, T *pMatC, uint32_t totalLength)
{
  int gridThreads = gridDim.x * blockDim.x;
  
  
  // grid stride loop to make sure we cover the entire data set
  for(int curIdx = threadIdx.x + blockIdx.x * blockDim.x; curIdx < totalLength; curIdx+=gridThreads )
  {
    float vals [WINDOW_SIZE];
    
    // load data into local vector
    for( int loadIdx = 0; loadIdx < WINDOW_SIZE; loadIdx++ )
    {
      int readIdx = curIdx + loadIdx - WINDOW_SIZE/2; 
      if((readIdx > 0) && (readIdx < totalLength))
      {
        vals[loadIdx] = pMatA[readIdx];
        
      }
      else
      {
        vals[loadIdx] = 0;
      }
    }
    
    //sum data into result
    for( int workIdx = 0; workIdx < WINDOW_SIZE; workIdx++)
    {
      pMatC[curIdx] += vals[workIdx];    
    }
    
  }
  
}


////////////////////////////////////////////////////////////////////////////////
///
///
///
///
////////////////////////////////////////////////////////////////////////////////
template< typename T, int WINDOW_SIZE, int BLOCK_SIZE > 
// __launch_bounds__(256, 2)
__global__ void windowAverageShared(T *pMatA, T *pMatC, uint32_t totalLength)
{
  int gridThreads = gridDim.x * blockDim.x;

  __shared__ float vals [BLOCK_SIZE+WINDOW_SIZE];
  
  // grid stride loop to make sure we cover the entire data set
  for(int curIdx = threadIdx.x + blockIdx.x * blockDim.x; curIdx < totalLength; curIdx+=gridThreads )
  {
    // load data into shared vector with a block-stride loop
    for( int sharedIdx = 0; sharedIdx < blockDim.x + WINDOW_SIZE; sharedIdx+=blockDim.x )
    {
      int readIdx = curIdx + sharedIdx - WINDOW_SIZE/2; //offset to center window on value
      
      if(readIdx > 0 && readIdx < totalLength)
      {
        vals[sharedIdx ] = pMatA[readIdx];
      }
      else
      {
        vals[sharedIdx] = 0;  
      }
      
    }
    
    __syncthreads();
    
    // float tempSum;
    
    //sum data into result
    for( int workIdx = 0; workIdx < WINDOW_SIZE; workIdx++)
    {
      pMatC[curIdx] += vals[workIdx + threadIdx.x];    
      // tempSum +=vals[workIdx + threadIdx.x];    
    }
    
    // pMatC[curIdx] = tempSum;
  }
  
}

////////////////////////////////////////////////////////////////////////////////
///
///
///
///
////////////////////////////////////////////////////////////////////////////////
template< typename T > 
void manual_cuda( int numSamples = 25)
{
  nvtxRangePushA("Manual CUDA Version");

  // initialize timing variables
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int warmupOffset = 5; //run a few times to warm up kernels  
  
  uint32_t matSizeX = 4096;
  uint32_t matSizeY = 4096;
  uint32_t dataSize =  matSizeX * matSizeY * sizeof(T);
  
  
  T *pMatA;
  T *pMatC;
  
  cudaMalloc(&pMatA, dataSize);
  cudaMalloc(&pMatC, dataSize);
  
  const int blockSize =  256; //hardcoded as largest block size
  
  //programatic
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int numBlocks = deviceProp.multiProcessorCount * 2; 
  
  
  for( int curSample = 0; curSample < numSamples; curSample++)
  {
    nvtxRangePushA("Iteration");
    if(curSample == warmupOffset )
    {
      //start of timing
      cudaEventRecord(start, 0);      
    }    
    
    // windowAverage<T, 100><<<numBlocks,blockSize>>>(pMatA, pMatC, matSizeX*matSizeY);
    windowAverageShared<T, 100, blockSize><<<numBlocks,blockSize>>>(pMatA, pMatC, matSizeX*matSizeY);
    
    nvtxRangePop();
  }
  
  //end of timing
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  
   //report average time
  float time_ms;
  cudaEventElapsedTime(&time_ms, start, stop);
  std::cout << "Average elapsed time per iteration is: " << time_ms * 1.0e3 / static_cast<double>(numSamples - warmupOffset)  << "us" << std::endl;
  cudaFree(pMatA);
  cudaFree(pMatC);
  nvtxRangePop();
}

////////////////////////////////////////////////////////////////////////////////
///
///
///
///
////////////////////////////////////////////////////////////////////////////////
int main()
{
  nvtxRangePushA("main");
  manual_cuda<float>();
  nvtxRangePop();
}

