#include <metal_stdlib>
using namespace metal;

//multiple filters can be applied at the same time using thread groups
kernel void conv2d(const device float* signal [[ buffer(0) ]],
                   const device float* filter [[ buffer(1) ]],
                   device float* result [[ buffer(2) ]],
                   const device uint &stride_r [[ buffer(3) ]],
                   const device uint &stride_c [[ buffer(4) ]],
                   const device uint &padding [[ buffer(5) ]],
                   const device uint &filter_r [[ buffer(6) ]],
                   const device uint &filter_c [[ buffer(7) ]],
                   const device uint &input_r [[ buffer(8) ]],
                   const device uint &input_c [[ buffer(9) ]],

                   const uint tgPos [[ threadgroup_position_in_grid ]],
                   const uint tPerTg [[ threads_per_threadgroup ]],
                   const uint tPos [[ thread_position_in_threadgroup ]]) {
  
  uint resultIndex = tgPos * tPerTg + tPos;
  
  uint offset = resultIndex; 
  
}

kernel void parsum(const device float* data [[ buffer(0) ]],
                   const device uint& dataLength [[ buffer(1) ]],
                   device float* sums [[ buffer(2) ]],
                   const device uint& elementsPerSum [[ buffer(3) ]],
                   
                   const uint tgPos [[ threadgroup_position_in_grid ]],
                   const uint tPerTg [[ threads_per_threadgroup ]],
                   const uint tPos [[ thread_position_in_threadgroup ]]) {
  

  //get index of results
  uint resultIndex = tgPos * tPerTg + tPos;
  
  //get the current index of data array.
  uint dataIndex = resultIndex * elementsPerSum; // Where the summation should begin
  uint endIndex = dataIndex + elementsPerSum < dataLength ? dataIndex + elementsPerSum : dataLength; // The index where summation should end
  
  float value = 0;
  
  for (; dataIndex < endIndex - 1; dataIndex += 2) {
    float first = data[dataIndex] * data[dataIndex + 1] ;
    value += first;
  }

  sums[resultIndex] = value;
}

kernel void activation(const device float* data [[ buffer(0) ]],
                       device float* sums [[ buffer(1) ]],
                       const device uint& activationType [[ buffer(2) ]],
                   
                       const uint tgPos [[ threadgroup_position_in_grid ]],
                       const uint tPerTg [[ threads_per_threadgroup ]],
                       const uint tPos [[ thread_position_in_threadgroup ]]) {
  
  //get index of results
  uint resultIndex = tgPos * tPerTg + tPos;
  
  float completeValue = data[resultIndex];

  if (activationType == 0) { //relu
    sums[resultIndex] = max((float)0, completeValue);
    
  } else if (activationType == 1) { //sigmoid
    sums[resultIndex] = 1.0 / (1.0 + exp(-completeValue));
    
  } else if (activationType == 2) { //leaky relu
    sums[resultIndex] = max((float)0.1 * completeValue, completeValue);
    
  } else if (activationType == 3) { //swish
    float sigmoid = 1.0 / (1.0 + exp(-completeValue));
    sums[resultIndex] = completeValue * sigmoid;
    
  } else if (activationType == 4) { //tanH
    float denom = 1.0 + exp(-2 * completeValue);
    sums[resultIndex] = (2.0 / denom) - 1.0;
    
  } else if (activationType == 5) { //none
    sums[resultIndex] = completeValue;
  }

}

