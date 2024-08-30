#include <iostream>

#include <nvtx3/nvToolsExt.h>

#define IDIVUP(a, b) (((a) + (b) - 1) / (b))

////////////////////////////////////////////////////////////////////////////////
///
///
///
///
////////////////////////////////////////////////////////////////////////////////
template< typename T > 
__launch_bounds__(1024, 2)
__global__ void customElementMultiple(T *pMatA, T *pMatB, T *pMatC, uint32_t matSizeX, uint32_t matSizeY)
{
  int numElements = matSizeX * matSizeY;
  int gridThreads = gridDim.x * blockDim.x;
  
  for(int curIdx = threadIdx.x + blockIdx.x * blockDim.x; curIdx < numElements; curIdx+=gridThreads )
  {
    pMatC[curIdx] = pMatA[curIdx] * pMatB[curIdx];
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
  
  uint32_t matSizeX = 4096;
  uint32_t matSizeY = 4096;
  uint32_t dataSize =  matSizeX * matSizeY * sizeof(T);
  
  
  T *pMatA;
  T *pMatB;
  T *pMatC;
  
  cudaMalloc(&pMatA, dataSize);
  cudaMalloc(&pMatB, dataSize);
  cudaMalloc(&pMatC, dataSize);
  
  int blockSize =  1024; //hardcoded as largest block size
  
  // int numBlocks = IDIVUP(matSizeX, blockSize);
  // simple scale out
  // int numBlocks = IDIVUP(matSizeX, blockSize) * matSizeY;
  // scale out but not as much
  // int numBlocks = IDIVUP(matSizeX, blockSize) * matSizeY/100;
  
  // //programatic
  // cudaDeviceProp deviceProp;
  // cudaGetDeviceProperties(&deviceProp, 0);
  // int numBlocks = deviceProp.multiProcessorCount; 
  
  #pragma unroll
  for( int curSample = 0; curSample < numSamples; curSample++)
  {
    customElementMultiple<T><<<numBlocks,blockSize>>>(pMatA, pMatB, pMatC, matSizeX, matSizeY);
  }
  
  cudaDeviceSynchronize();
  
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

