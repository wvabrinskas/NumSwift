#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// MARK: - Data Structures

struct NSC_Size {
  uint rows;
  uint columns;
};

struct NSC_IndexedValue {
  half value;
  uint index;
};

enum NSC_Padding {
  valid = 0,
  same = 1
  };
  
  // MARK: - Basic Array Operations
  
  // Legacy single-threaded sum (kept for small arrays)
  kernel void nsc_sum_kernel(device const half* input [[ buffer(0) ]],
                             device half* result [[ buffer(1) ]],
                             device const uint* size [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    half sum = 0.0h;
    for (uint i = 0; i < *size; i++) {
      sum += input[i];
    }
    *result = sum;
  }
  
  // Optimized parallel reduction sum
  kernel void nsc_parallel_sum_kernel(device const half* input [[ buffer(0) ]],
                                      device half* result [[ buffer(1) ]],
                                      device const uint* size [[ buffer(2) ]],
                                      threadgroup half* shared_data [[ threadgroup(0) ]],
                                      uint thread_id [[ thread_position_in_threadgroup ]],
                                      uint threadgroup_id [[ threadgroup_position_in_grid ]],
                                      uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                                      uint threadgroups_per_grid [[ threadgroups_per_grid ]]) {
    
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    uint total_threads = threads_per_threadgroup * threadgroups_per_grid;
    
    // Each thread sums multiple elements if array is larger than thread count
    half local_sum = 0.0h;
    for (uint i = global_id; i < *size; i += total_threads) {
      local_sum += input[i];
    }
    
    // Store local sum in shared memory
    shared_data[thread_id] = local_sum;
    
    // Synchronize threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride /= 2) {
      if (thread_id < stride) {
        shared_data[thread_id] += shared_data[thread_id + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread in each threadgroup writes result
    if (thread_id == 0) {
      result[threadgroup_id] = shared_data[0];
    }
  }
  
  kernel void nsc_sum_of_squares_kernel(device const half* input [[ buffer(0) ]],
                                        device half* result [[ buffer(1) ]],
                                        device const uint* size [[ buffer(2) ]],
                                        uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    half sum = 0.0h;
    for (uint i = 0; i < *size; i++) {
      sum += input[i] * input[i];
    }
    *result = sum;
  }
  
  // Legacy single-threaded max (kept for small arrays)
  kernel void nsc_max_kernel(device const half* input [[ buffer(0) ]],
                             device half* result [[ buffer(1) ]],
                             device const uint* size [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    half max_val = input[0];
    for (uint i = 1; i < *size; i++) {
      if (input[i] > max_val) {
        max_val = input[i];
      }
    }
    *result = max_val;
  }
  
  // Optimized parallel reduction max
  kernel void nsc_parallel_max_kernel(device const half* input [[ buffer(0) ]],
                                      device half* result [[ buffer(1) ]],
                                      device const uint* size [[ buffer(2) ]],
                                      threadgroup half* shared_data [[ threadgroup(0) ]],
                                      uint thread_id [[ thread_position_in_threadgroup ]],
                                      uint threadgroup_id [[ threadgroup_position_in_grid ]],
                                      uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                                      uint threadgroups_per_grid [[ threadgroups_per_grid ]]) {
    
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    uint total_threads = threads_per_threadgroup * threadgroups_per_grid;
    
    // Initialize with negative infinity for max operation
    half local_max = -INFINITY;
    
    // Each thread finds max of multiple elements if array is larger than thread count
    for (uint i = global_id; i < *size; i += total_threads) {
      local_max = max(local_max, input[i]);
    }
    
    // Store local max in shared memory
    shared_data[thread_id] = local_max;
    
    // Synchronize threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride /= 2) {
      if (thread_id < stride) {
        shared_data[thread_id] = max(shared_data[thread_id], shared_data[thread_id + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread in each threadgroup writes result
    if (thread_id == 0) {
      result[threadgroup_id] = shared_data[0];
    }
  }
  
  // Legacy single-threaded min (kept for small arrays)
  kernel void nsc_min_kernel(device const half* input [[ buffer(0) ]],
                             device half* result [[ buffer(1) ]],
                             device const uint* size [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    half min_val = input[0];
    for (uint i = 1; i < *size; i++) {
      if (input[i] < min_val) {
        min_val = input[i];
      }
    }
    *result = min_val;
  }
  
  // Optimized parallel reduction min
  kernel void nsc_parallel_min_kernel(device const half* input [[ buffer(0) ]],
                                      device half* result [[ buffer(1) ]],
                                      device const uint* size [[ buffer(2) ]],
                                      threadgroup half* shared_data [[ threadgroup(0) ]],
                                      uint thread_id [[ thread_position_in_threadgroup ]],
                                      uint threadgroup_id [[ threadgroup_position_in_grid ]],
                                      uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                                      uint threadgroups_per_grid [[ threadgroups_per_grid ]]) {
    
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    uint total_threads = threads_per_threadgroup * threadgroups_per_grid;
    
    // Initialize with positive infinity for min operation
    half local_min = INFINITY;
    
    // Each thread finds min of multiple elements if array is larger than thread count
    for (uint i = global_id; i < *size; i += total_threads) {
      local_min = min(local_min, input[i]);
    }
    
    // Store local min in shared memory
    shared_data[thread_id] = local_min;
    
    // Synchronize threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride /= 2) {
      if (thread_id < stride) {
        shared_data[thread_id] = min(shared_data[thread_id], shared_data[thread_id + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread in each threadgroup writes result
    if (thread_id == 0) {
      result[threadgroup_id] = shared_data[0];
    }
  }
  
  kernel void nsc_index_of_max_kernel(device const half* input [[ buffer(0) ]],
                                      device NSC_IndexedValue* result [[ buffer(1) ]],
                                      device const uint* size [[ buffer(2) ]],
                                      uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    half max_val = input[0];
    uint max_index = 0;
    for (uint i = 1; i < *size; i++) {
      if (input[i] > max_val) {
        max_val = input[i];
        max_index = i;
      }
    }
    result->value = max_val;
    result->index = max_index;
  }
  
  kernel void nsc_index_of_min_kernel(device const half* input [[ buffer(0) ]],
                                      device NSC_IndexedValue* result [[ buffer(1) ]],
                                      device const uint* size [[ buffer(2) ]],
                                      uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    half min_val = input[0];
    uint min_index = 0;
    for (uint i = 1; i < *size; i++) {
      if (input[i] < min_val) {
        min_val = input[i];
        min_index = i;
      }
    }
    result->value = min_val;
    result->index = min_index;
  }
  
  // MARK: - Arithmetic Operations
  
  kernel void nsc_add_scalar_kernel(device const half* lhs [[ buffer(0) ]],
                                    device const half* rhs [[ buffer(1) ]],
                                    device half* result [[ buffer(2) ]],
                                    uint id [[ thread_position_in_grid ]]) {
    result[id] = *lhs + rhs[id];
  }
  
  kernel void nsc_add_kernel(device const half* lhs [[ buffer(0) ]],
                             device const half* rhs [[ buffer(1) ]],
                             device half* result [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] + rhs[id];
  }
  
  kernel void nsc_sub_kernel(device const half* lhs [[ buffer(0) ]],
                             device const half* rhs [[ buffer(1) ]],
                             device half* result [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] - rhs[id];
  }
  
  kernel void nsc_mult_scalar_kernel(device const half* lhs [[ buffer(0) ]],
                                     device const half* rhs [[ buffer(1) ]],
                                     device half* result [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]]) {
    result[id] = *lhs * rhs[id];
  }
  
  kernel void nsc_mult_kernel(device const half* lhs [[ buffer(0) ]],
                              device const half* rhs [[ buffer(1) ]],
                              device half* result [[ buffer(2) ]],
                              uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] * rhs[id];
  }
  
  kernel void nsc_div_kernel(device const half* lhs [[ buffer(0) ]],
                             device const half* rhs [[ buffer(1) ]],
                             device half* result [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] / rhs[id];
  }
  
  kernel void nsc_div_scalar_array_kernel(device const half* lhs [[ buffer(0) ]],
                                          device const half* rhs [[ buffer(1) ]],
                                          device half* result [[ buffer(2) ]],
                                          uint id [[ thread_position_in_grid ]]) {
    result[id] = *lhs / rhs[id];
  }
  
  kernel void nsc_div_array_scalar_kernel(device const half* lhs [[ buffer(0) ]],
                                          device const half* rhs [[ buffer(1) ]],
                                          device half* result [[ buffer(2) ]],
                                          uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] / *rhs;
  }
  
  // MARK: - Matrix Operations
  
  // Legacy simple matrix multiplication (for small matrices)
  kernel void nsc_matmul_kernel(device const half* a [[ buffer(0) ]],
                                device const half* b [[ buffer(1) ]],
                                device half* result [[ buffer(2) ]],
                                device const NSC_Size* a_size [[ buffer(3) ]],
                                device const NSC_Size* b_size [[ buffer(4) ]],
                                uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row >= a_size->rows || col >= b_size->columns) return;
    
    half sum = 0.0h;
    for (uint k = 0; k < a_size->columns; k++) {
      half a_val = a[row * a_size->columns + k];
      half b_val = b[k * b_size->columns + col];
      sum += a_val * b_val;
    }
    
    result[row * b_size->columns + col] = sum;
  }
  
  // Optimized tiled matrix multiplication with shared memory
  kernel void nsc_tiled_matmul_kernel(device const half* a [[ buffer(0) ]],
                                      device const half* b [[ buffer(1) ]],
                                      device half* result [[ buffer(2) ]],
                                      device const NSC_Size* a_size [[ buffer(3) ]],
                                      device const NSC_Size* b_size [[ buffer(4) ]],
                                      threadgroup half* shared_a [[ threadgroup(0) ]],
                                      threadgroup half* shared_b [[ threadgroup(1) ]],
                                      uint2 thread_id [[ thread_position_in_threadgroup ]],
                                      uint2 threadgroup_id [[ threadgroup_position_in_grid ]],
                                      uint2 threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    
    const uint TILE_SIZE = 16; // Must match threadgroup size
    
    uint row = threadgroup_id.y * TILE_SIZE + thread_id.y;
    uint col = threadgroup_id.x * TILE_SIZE + thread_id.x;
    
    half sum = 0.0h;
    
    uint num_tiles = (a_size->columns + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint tile = 0; tile < num_tiles; tile++) {
      // Load tile of A into shared memory
      uint a_row = row;
      uint a_col = tile * TILE_SIZE + thread_id.x;
      if (a_row < a_size->rows && a_col < a_size->columns) {
        shared_a[thread_id.y * TILE_SIZE + thread_id.x] = a[a_row * a_size->columns + a_col];
      } else {
        shared_a[thread_id.y * TILE_SIZE + thread_id.x] = 0.0h;
      }
      
      // Load tile of B into shared memory
      uint b_row = tile * TILE_SIZE + thread_id.y;
      uint b_col = col;
      if (b_row < b_size->rows && b_col < b_size->columns) {
        shared_b[thread_id.y * TILE_SIZE + thread_id.x] = b[b_row * b_size->columns + b_col];
      } else {
        shared_b[thread_id.y * TILE_SIZE + thread_id.x] = 0.0h;
      }
      
      // Synchronize to ensure all threads have loaded their data
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Compute partial dot product for this tile
      for (uint k = 0; k < TILE_SIZE; k++) {
        sum += shared_a[thread_id.y * TILE_SIZE + k] * shared_b[k * TILE_SIZE + thread_id.x];
      }
      
      // Synchronize before loading next tile
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    if (row < a_size->rows && col < b_size->columns) {
      result[row * b_size->columns + col] = sum;
    }
  }
  
  // Batched matrix multiplication for neural networks
  kernel void nsc_batched_matmul_kernel(device const half* a [[ buffer(0) ]],
                                        device const half* b [[ buffer(1) ]],
                                        device half* result [[ buffer(2) ]],
                                        device const NSC_Size* a_size [[ buffer(3) ]],
                                        device const NSC_Size* b_size [[ buffer(4) ]],
                                        device const uint* batch_size [[ buffer(5) ]],
                                        uint3 id [[ thread_position_in_grid ]]) {
    uint batch = id.z;
    uint row = id.y;
    uint col = id.x;
    
    if (batch >= *batch_size || row >= a_size->rows || col >= b_size->columns) return;
    
    uint a_offset = batch * a_size->rows * a_size->columns;
    uint b_offset = batch * b_size->rows * b_size->columns;
    uint result_offset = batch * a_size->rows * b_size->columns;
    
    half sum = 0.0h;
    for (uint k = 0; k < a_size->columns; k++) {
      half a_val = a[a_offset + row * a_size->columns + k];
      half b_val = b[b_offset + k * b_size->columns + col];
      sum += a_val * b_val;
    }
    
    result[result_offset + row * b_size->columns + col] = sum;
  }
  
  kernel void nsc_transpose_2d_kernel(device const half* input [[ buffer(0) ]],
                                      device half* result [[ buffer(1) ]],
                                      device const NSC_Size* input_size [[ buffer(2) ]],
                                      uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row >= input_size->rows || col >= input_size->columns) return;
    
    // Transpose: result[col][row] = input[row][col]
    result[col * input_size->rows + row] = input[row * input_size->columns + col];
  }
  
  kernel void nsc_flatten2d_kernel(device const half* input [[ buffer(0) ]],
                                   device half* result [[ buffer(1) ]],
                                   device const NSC_Size* input_size [[ buffer(2) ]],
                                   uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row >= input_size->rows || col >= input_size->columns) return;
    
    uint flat_index = row * input_size->columns + col;
    result[flat_index] = input[row * input_size->columns + col];
  }
  
  // MARK: - Padding Operations
  
  kernel void nsc_specific_zero_pad_kernel(device const half* input [[ buffer(0) ]],
                                           device half* result [[ buffer(1) ]],
                                           device const NSC_Size* input_size [[ buffer(2) ]],
                                           device const int* padding_top [[ buffer(3) ]],
                                           device const int* padding_left [[ buffer(4) ]],
                                           uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row >= input_size->rows || col >= input_size->columns) return;
    
    uint padded_row = row + *padding_top;
    uint padded_col = col + *padding_left;
    uint padded_width = input_size->columns + *padding_left + *padding_left; // Assuming symmetric padding
    
    result[padded_row * padded_width + padded_col] = input[row * input_size->columns + col];
  }
  
  kernel void nsc_stride_pad_kernel(device const half* input [[ buffer(0) ]],
                                    device half* result [[ buffer(1) ]],
                                    device const NSC_Size* input_size [[ buffer(2) ]],
                                    device const NSC_Size* stride_size [[ buffer(3) ]],
                                    uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row >= input_size->rows || col >= input_size->columns) return;
    
    uint new_rows = input_size->rows + ((stride_size->rows - 1) * (input_size->rows - 1));
    uint new_cols = input_size->columns + ((stride_size->columns - 1) * (input_size->columns - 1));
    
    uint result_row = row * stride_size->rows;
    uint result_col = col * stride_size->columns;
    
    if (result_row < new_rows && result_col < new_cols) {
      result[result_row * new_cols + result_col] = input[row * input_size->columns + col];
    }
  }
  
  // MARK: - Convolution Operations
  
  // Legacy 2D convolution (for small inputs)
  kernel void nsc_conv2d_kernel(device const half* signal [[ buffer(0) ]],
                                device const half* filter [[ buffer(1) ]],
                                device half* result [[ buffer(2) ]],
                                device const NSC_Size* input_size [[ buffer(3) ]],
                                device const NSC_Size* filter_size [[ buffer(4) ]],
                                device const NSC_Size* stride [[ buffer(5) ]],
                                device const NSC_Size* result_size [[ buffer(6) ]],
                                device const int* padding_type [[ buffer(7) ]],
                                device const int* pad_top [[ buffer(8) ]],
                                device const int* pad_left [[ buffer(9) ]],
                                uint2 id [[ thread_position_in_grid ]]) {
    uint result_row = id.y;
    uint result_col = id.x;
    
    if (result_row >= result_size->rows || result_col >= result_size->columns) return;
    
    half sum = 0.0h;
    
    for (uint fr = 0; fr < filter_size->rows; fr++) {
      for (uint fc = 0; fc < filter_size->columns; fc++) {
        int signal_row = (int)(result_row * stride->rows + fr) - *pad_top;
        int signal_col = (int)(result_col * stride->columns + fc) - *pad_left;
        
        if (signal_row >= 0 && signal_row < (int)input_size->rows && 
            signal_col >= 0 && signal_col < (int)input_size->columns) {
          half signal_val = signal[signal_row * input_size->columns + signal_col];
          half filter_val = filter[fr * filter_size->columns + fc];
          sum += signal_val * filter_val;
        }
      }
    }
    
    result[result_row * result_size->columns + result_col] = sum;
  }
  
  // Optimized batched convolution for neural networks
  kernel void nsc_batched_conv2d_kernel(device const half* signal [[ buffer(0) ]],
                                        device const half* filter [[ buffer(1) ]],
                                        device half* result [[ buffer(2) ]],
                                        device const NSC_Size* input_size [[ buffer(3) ]],  // [height, width]
                                        device const NSC_Size* filter_size [[ buffer(4) ]], // [height, width]
                                        device const NSC_Size* stride [[ buffer(5) ]],
                                        device const NSC_Size* result_size [[ buffer(6) ]], // [height, width]
                                        device const int* padding_type [[ buffer(7) ]],
                                        device const int* pad_top [[ buffer(8) ]],
                                        device const int* pad_left [[ buffer(9) ]],
                                        device const uint* batch_size [[ buffer(10) ]],
                                        device const uint* in_channels [[ buffer(11) ]],
                                        device const uint* out_channels [[ buffer(12) ]],
                                        uint3 id [[ thread_position_in_grid ]]) {
    uint batch = id.z;
    uint result_row = id.y;
    uint result_col = id.x;
    
    if (batch >= *batch_size || result_row >= result_size->rows || result_col >= result_size->columns) return;
    
    uint input_spatial_size = input_size->rows * input_size->columns;
    uint result_spatial_size = result_size->rows * result_size->columns;
    uint filter_spatial_size = filter_size->rows * filter_size->columns;
    
    // For each output channel
    for (uint out_ch = 0; out_ch < *out_channels; out_ch++) {
      half sum = 0.0h;
      
      // For each input channel
      for (uint in_ch = 0; in_ch < *in_channels; in_ch++) {
        // Convolution computation
        for (uint fr = 0; fr < filter_size->rows; fr++) {
          for (uint fc = 0; fc < filter_size->columns; fc++) {
            int signal_row = (int)(result_row * stride->rows + fr) - *pad_top;
            int signal_col = (int)(result_col * stride->columns + fc) - *pad_left;
            
            if (signal_row >= 0 && signal_row < (int)input_size->rows && 
                signal_col >= 0 && signal_col < (int)input_size->columns) {
              
              uint signal_idx = batch * (*in_channels) * input_spatial_size + 
                               in_ch * input_spatial_size + 
                               signal_row * input_size->columns + signal_col;
              
              uint filter_idx = out_ch * (*in_channels) * filter_spatial_size + 
                               in_ch * filter_spatial_size + 
                               fr * filter_size->columns + fc;
              
              sum += signal[signal_idx] * filter[filter_idx];
            }
          }
        }
      }
      
      uint result_idx = batch * (*out_channels) * result_spatial_size + 
                       out_ch * result_spatial_size + 
                       result_row * result_size->columns + result_col;
      result[result_idx] = sum;
    }
  }
  
  // Optimized im2col-based convolution for better memory access patterns
  kernel void nsc_im2col_conv2d_kernel(device const half* signal [[ buffer(0) ]],
                                       device const half* filter [[ buffer(1) ]],
                                       device half* result [[ buffer(2) ]],
                                       device const NSC_Size* input_size [[ buffer(3) ]],
                                       device const NSC_Size* filter_size [[ buffer(4) ]],
                                       device const NSC_Size* stride [[ buffer(5) ]],
                                       device const NSC_Size* result_size [[ buffer(6) ]],
                                       device const int* pad_top [[ buffer(7) ]],
                                       device const int* pad_left [[ buffer(8) ]],
                                       threadgroup half* shared_data [[ threadgroup(0) ]],
                                       uint2 thread_id [[ thread_position_in_threadgroup ]],
                                       uint2 threadgroup_id [[ threadgroup_position_in_grid ]],
                                       uint2 threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    
    const uint TILE_SIZE = 16;
    uint result_row = threadgroup_id.y * TILE_SIZE + thread_id.y;
    uint result_col = threadgroup_id.x * TILE_SIZE + thread_id.x;
    
    if (result_row >= result_size->rows || result_col >= result_size->columns) return;
    
    half sum = 0.0h;
    
    // Unroll small filters for better performance
    if (filter_size->rows == 3 && filter_size->columns == 3) {
      // Optimized 3x3 convolution
      for (uint fr = 0; fr < 3; fr++) {
        for (uint fc = 0; fc < 3; fc++) {
          int signal_row = (int)(result_row * stride->rows + fr) - *pad_top;
          int signal_col = (int)(result_col * stride->columns + fc) - *pad_left;
          
          if (signal_row >= 0 && signal_row < (int)input_size->rows && 
              signal_col >= 0 && signal_col < (int)input_size->columns) {
            half signal_val = signal[signal_row * input_size->columns + signal_col];
            half filter_val = filter[fr * 3 + fc];
            sum += signal_val * filter_val;
          }
        }
      }
    } else {
      // General convolution
      for (uint fr = 0; fr < filter_size->rows; fr++) {
        for (uint fc = 0; fc < filter_size->columns; fc++) {
          int signal_row = (int)(result_row * stride->rows + fr) - *pad_top;
          int signal_col = (int)(result_col * stride->columns + fc) - *pad_left;
          
          if (signal_row >= 0 && signal_row < (int)input_size->rows && 
              signal_col >= 0 && signal_col < (int)input_size->columns) {
            half signal_val = signal[signal_row * input_size->columns + signal_col];
            half filter_val = filter[fr * filter_size->columns + fc];
            sum += signal_val * filter_val;
          }
        }
      }
    }
    
    result[result_row * result_size->columns + result_col] = sum;
  }
  
  kernel void nsc_conv1d_kernel(device const half* signal [[ buffer(0) ]],
                                device const half* filter [[ buffer(1) ]],
                                device half* result [[ buffer(2) ]],
                                device const NSC_Size* input_size [[ buffer(3) ]],
                                device const NSC_Size* filter_size [[ buffer(4) ]],
                                device const NSC_Size* stride [[ buffer(5) ]],
                                device const NSC_Size* result_size [[ buffer(6) ]],
                                uint id [[ thread_position_in_grid ]]) {
    uint result_idx = id;
    
    if (result_idx >= result_size->rows * result_size->columns) return;
    
    half sum = 0.0h;
    
    uint result_row = result_idx / result_size->columns;
    uint result_col = result_idx % result_size->columns;
    
    for (uint fr = 0; fr < filter_size->rows; fr++) {
      for (uint fc = 0; fc < filter_size->columns; fc++) {
        uint signal_row = result_row * stride->rows + fr;
        uint signal_col = result_col * stride->columns + fc;
        
        if (signal_row < input_size->rows && signal_col < input_size->columns) {
          half signal_val = signal[signal_row * input_size->columns + signal_col];
          half filter_val = filter[fr * filter_size->columns + fc];
          sum += signal_val * filter_val;
        }
      }
    }
    
    result[result_idx] = sum;
  }
  
  // MARK: - Transpose Convolution Operations
  
  kernel void nsc_transconv2d_kernel(device const half* signal [[ buffer(0) ]],
                                     device const half* filter [[ buffer(1) ]],
                                     device half* result [[ buffer(2) ]],
                                     device const NSC_Size* input_size [[ buffer(3) ]],
                                     device const NSC_Size* filter_size [[ buffer(4) ]],
                                     device const NSC_Size* stride [[ buffer(5) ]],
                                     device const NSC_Size* result_size [[ buffer(6) ]],
                                     uint2 id [[ thread_position_in_grid ]]) {
    uint input_row = id.y;
    uint input_col = id.x;
    
    if (input_row >= input_size->rows || input_col >= input_size->columns) return;
    
    half signal_val = signal[input_row * input_size->columns + input_col];
    
    uint i_prime = input_row * stride->rows;
    uint j_prime = input_col * stride->columns;
    
    for (uint fr = 0; fr < filter_size->rows; fr++) {
      for (uint fc = 0; fc < filter_size->columns; fc++) {
        uint result_row = i_prime + fr;
        uint result_col = j_prime + fc;
        
        if (result_row < result_size->rows && result_col < result_size->columns) {
          half filter_val = filter[fr * filter_size->columns + fc];
          uint result_idx = result_row * result_size->columns + result_col;
          
          
          // Metal doesn't support atomic operations on half precision
          // For transpose convolution, ensure result buffer is zero-initialized before kernel execution
          // Race conditions are generally acceptable for this operation type
          result[result_idx] += signal_val * filter_val;
        }
      }
    }
  }
  
  kernel void nsc_transconv1d_kernel(device const half* signal [[ buffer(0) ]],
                                     device const half* filter [[ buffer(1) ]],
                                     device half* result [[ buffer(2) ]],
                                     device const NSC_Size* input_size [[ buffer(3) ]],
                                     device const NSC_Size* filter_size [[ buffer(4) ]],
                                     device const NSC_Size* stride [[ buffer(5) ]],
                                     device const NSC_Size* result_size [[ buffer(6) ]],
                                     uint id [[ thread_position_in_grid ]]) {
    uint input_idx = id;
    
    if (input_idx >= input_size->rows * input_size->columns) return;
    
    half signal_val = signal[input_idx];
    
    uint input_row = input_idx / input_size->columns;
    uint input_col = input_idx % input_size->columns;
    
    uint i_prime = input_row * stride->rows;
    uint j_prime = input_col * stride->columns;
    
    for (uint fr = 0; fr < filter_size->rows; fr++) {
      for (uint fc = 0; fc < filter_size->columns; fc++) {
        uint result_row = i_prime + fr;
        uint result_col = j_prime + fc;
        
        if (result_row < result_size->rows && result_col < result_size->columns) {
          half filter_val = filter[fr * filter_size->columns + fc];
          uint result_idx = result_row * result_size->columns + result_col;
          
          // Metal doesn't support atomic operations on half precision
          // For transpose convolution, ensure result buffer is zero-initialized before kernel execution
          // Race conditions are generally acceptable for this operation type
          result[result_idx] += signal_val * filter_val;
        }
      }
    }
  }
  
  // MARK: - Perlin Noise
  
  inline half scaled_cosine(half i) {
    return 0.5h * (1.0h - cos(i * M_PI_H));
  }
  
  kernel void nsc_perlin_noise_kernel(device half* result [[ buffer(0) ]],
                                      device const half* coords [[ buffer(1) ]],
                                      device const half* amplitude [[ buffer(2) ]],
                                      device const int* octaves [[ buffer(3) ]],
                                      device const uint* size [[ buffer(4) ]],
                                      device const half* perlin_seed [[ buffer(5) ]],
                                      uint id [[ thread_position_in_grid ]]) {
    
    uint coord_idx = id * 3; // x, y, z coordinates
    if (coord_idx + 2 >= *size * 3) return;
    
    half x = coords[coord_idx];
    half y = coords[coord_idx + 1];
    half z = coords[coord_idx + 2];
    
    const int yWrapB = 4;
    const int yWrap = 1 << yWrapB;
    const int zWrapB = 8;
    const int zWrap = 1 << zWrapB;
    
    if (x < 0.0h) x = -x;
    if (y < 0.0h) y = -y;
    if (z < 0.0h) z = -z;
    
    int xi = (int)floor(x);
    int yi = (int)floor(y);
    int zi = (int)floor(z);
    
    half xf = x - (half)xi;
    half yf = y - (half)yi;
    half zf = z - (half)zi;
    
    half r = 0.0h;
    half ampl = 0.5h;
    
    for (int i = 0; i < *octaves; i++) {
      int of = xi + (yi << yWrapB) + (zi << zWrapB);
      
      half rxf = scaled_cosine(xf);
      half ryf = scaled_cosine(yf);
      
      half n1 = perlin_seed[of & *size];
      n1 += rxf * (perlin_seed[(of + 1) & *size] - n1);
      half n2 = perlin_seed[(of + yWrap) & *size];
      n2 += rxf * (perlin_seed[(of + yWrap + 1) & *size] - n2);
      n1 += ryf * (n2 - n1);
      
      of += zWrap;
      n2 = perlin_seed[of & *size];
      n2 += rxf * (perlin_seed[(of + 1) & *size] - n2);
      half n3 = perlin_seed[(of + yWrap) & *size];
      n3 += rxf * (perlin_seed[(of + yWrap + 1) & *size] - n3);
      n2 += ryf * (n3 - n2);
      
      n1 += scaled_cosine(zf) * (n2 - n1);
      
      r += n1 * ampl;
      ampl *= *amplitude;
      xi <<= 1;
      xf *= 2.0h;
      yi <<= 1;
      yf *= 2.0h;
      zi <<= 1;
      zf *= 2.0h;
      
      if (xf >= 1.0h) {
        xi += 1;
        xf -= 1.0h;
      }
      if (yf >= 1.0h) {
        yi += 1;
        yf -= 1.0h;
      }
      if (zf >= 1.0h) {
        zi += 1;
        zf -= 1.0h;
      }
    }
    
    result[id] = r;
  }
  
  // MARK: - Utility Functions
  
  kernel void nsc_zero_buffer_kernel(device half* buffer [[ buffer(0) ]],
                                     device const uint* size [[ buffer(1) ]],
                                     uint id [[ thread_position_in_grid ]]) {
    if (id >= *size) return;
    buffer[id] = 0.0h;
  }
  
  kernel void nsc_copy_buffer_kernel(device const half* source [[ buffer(0) ]],
                                     device half* destination [[ buffer(1) ]],
                                     device const uint* size [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]]) {
    if (id >= *size) return;
    destination[id] = source[id];
  }
  
  // MARK: - Float32 Variants (for compatibility with existing float code)
  
  // MARK: - Basic Array Operations (Float32)
  
  // Legacy single-threaded sum (kept for small arrays)
  kernel void nsc_sum_float_kernel(device const float* input [[ buffer(0) ]],
                                   device float* result [[ buffer(1) ]],
                                   device const uint* size [[ buffer(2) ]],
                                   uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < *size; i++) {
      sum += input[i];
    }
    *result = sum;
  }
  
  // Optimized parallel reduction sum (Float32)
  kernel void nsc_parallel_sum_float_kernel(device const float* input [[ buffer(0) ]],
                                            device float* result [[ buffer(1) ]],
                                            device const uint* size [[ buffer(2) ]],
                                            threadgroup float* shared_data [[ threadgroup(0) ]],
                                            uint thread_id [[ thread_position_in_threadgroup ]],
                                            uint threadgroup_id [[ threadgroup_position_in_grid ]],
                                            uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                                            uint threadgroups_per_grid [[ threadgroups_per_grid ]]) {
    
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    uint total_threads = threads_per_threadgroup * threadgroups_per_grid;
    
    // Each thread sums multiple elements if array is larger than thread count
    float local_sum = 0.0f;
    for (uint i = global_id; i < *size; i += total_threads) {
      local_sum += input[i];
    }
    
    // Store local sum in shared memory
    shared_data[thread_id] = local_sum;
    
    // Synchronize threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride /= 2) {
      if (thread_id < stride) {
        shared_data[thread_id] += shared_data[thread_id + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread in each threadgroup writes result
    if (thread_id == 0) {
      result[threadgroup_id] = shared_data[0];
    }
  }
  
  kernel void nsc_sum_of_squares_float_kernel(device const float* input [[ buffer(0) ]],
                                              device float* result [[ buffer(1) ]],
                                              device const uint* size [[ buffer(2) ]],
                                              uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < *size; i++) {
      sum += input[i] * input[i];
    }
    *result = sum;
  }
  
  // Legacy single-threaded max (kept for small arrays)
  kernel void nsc_max_float_kernel(device const float* input [[ buffer(0) ]],
                                   device float* result [[ buffer(1) ]],
                                   device const uint* size [[ buffer(2) ]],
                                   uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    float max_val = input[0];
    for (uint i = 1; i < *size; i++) {
      if (input[i] > max_val) {
        max_val = input[i];
      }
    }
    *result = max_val;
  }
  
  // Optimized parallel reduction max (Float32)
  kernel void nsc_parallel_max_float_kernel(device const float* input [[ buffer(0) ]],
                                            device float* result [[ buffer(1) ]],
                                            device const uint* size [[ buffer(2) ]],
                                            threadgroup float* shared_data [[ threadgroup(0) ]],
                                            uint thread_id [[ thread_position_in_threadgroup ]],
                                            uint threadgroup_id [[ threadgroup_position_in_grid ]],
                                            uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                                            uint threadgroups_per_grid [[ threadgroups_per_grid ]]) {
    
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    uint total_threads = threads_per_threadgroup * threadgroups_per_grid;
    
    // Initialize with negative infinity for max operation
    float local_max = -INFINITY;
    
    // Each thread finds max of multiple elements if array is larger than thread count
    for (uint i = global_id; i < *size; i += total_threads) {
      local_max = max(local_max, input[i]);
    }
    
    // Store local max in shared memory
    shared_data[thread_id] = local_max;
    
    // Synchronize threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride /= 2) {
      if (thread_id < stride) {
        shared_data[thread_id] = max(shared_data[thread_id], shared_data[thread_id + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread in each threadgroup writes result
    if (thread_id == 0) {
      result[threadgroup_id] = shared_data[0];
    }
  }
  
  // Legacy single-threaded min (kept for small arrays)
  kernel void nsc_min_float_kernel(device const float* input [[ buffer(0) ]],
                                   device float* result [[ buffer(1) ]],
                                   device const uint* size [[ buffer(2) ]],
                                   uint id [[ thread_position_in_grid ]]) {
    if (id != 0) return;
    
    float min_val = input[0];
    for (uint i = 1; i < *size; i++) {
      if (input[i] < min_val) {
        min_val = input[i];
      }
    }
    *result = min_val;
  }
  
  // Optimized parallel reduction min (Float32)
  kernel void nsc_parallel_min_float_kernel(device const float* input [[ buffer(0) ]],
                                            device float* result [[ buffer(1) ]],
                                            device const uint* size [[ buffer(2) ]],
                                            threadgroup float* shared_data [[ threadgroup(0) ]],
                                            uint thread_id [[ thread_position_in_threadgroup ]],
                                            uint threadgroup_id [[ threadgroup_position_in_grid ]],
                                            uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                                            uint threadgroups_per_grid [[ threadgroups_per_grid ]]) {
    
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    uint total_threads = threads_per_threadgroup * threadgroups_per_grid;
    
    // Initialize with positive infinity for min operation
    float local_min = INFINITY;
    
    // Each thread finds min of multiple elements if array is larger than thread count
    for (uint i = global_id; i < *size; i += total_threads) {
      local_min = min(local_min, input[i]);
    }
    
    // Store local min in shared memory
    shared_data[thread_id] = local_min;
    
    // Synchronize threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride /= 2) {
      if (thread_id < stride) {
        shared_data[thread_id] = min(shared_data[thread_id], shared_data[thread_id + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread in each threadgroup writes result
    if (thread_id == 0) {
      result[threadgroup_id] = shared_data[0];
    }
  }
  
  // MARK: - Arithmetic Operations (Float32)
  
  kernel void nsc_add_scalar_float_kernel(device const float* lhs [[ buffer(0) ]],
                                          device const float* rhs [[ buffer(1) ]],
                                          device float* result [[ buffer(2) ]],
                                          uint id [[ thread_position_in_grid ]]) {
    result[id] = *lhs + rhs[id];
  }
  
  kernel void nsc_add_float_kernel(device const float* lhs [[ buffer(0) ]],
                                   device const float* rhs [[ buffer(1) ]],
                                   device float* result [[ buffer(2) ]],
                                   uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] + rhs[id];
  }
  
  kernel void nsc_sub_float_kernel(device const float* lhs [[ buffer(0) ]],
                                   device const float* rhs [[ buffer(1) ]],
                                   device float* result [[ buffer(2) ]],
                                   uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] - rhs[id];
  }
  
  kernel void nsc_mult_scalar_float_kernel(device const float* lhs [[ buffer(0) ]],
                                           device const float* rhs [[ buffer(1) ]],
                                           device float* result [[ buffer(2) ]],
                                           uint id [[ thread_position_in_grid ]]) {
    result[id] = *lhs * rhs[id];
  }
  
  kernel void nsc_mult_float_kernel(device const float* lhs [[ buffer(0) ]],
                                    device const float* rhs [[ buffer(1) ]],
                                    device float* result [[ buffer(2) ]],
                                    uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] * rhs[id];
  }
  
  kernel void nsc_div_float_kernel(device const float* lhs [[ buffer(0) ]],
                                   device const float* rhs [[ buffer(1) ]],
                                   device float* result [[ buffer(2) ]],
                                   uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] / rhs[id];
  }
  
  kernel void nsc_div_scalar_array_float_kernel(device const float* lhs [[ buffer(0) ]],
                                                device const float* rhs [[ buffer(1) ]],
                                                device float* result [[ buffer(2) ]],
                                                uint id [[ thread_position_in_grid ]]) {
    result[id] = *lhs / rhs[id];
  }
  
  kernel void nsc_div_array_scalar_float_kernel(device const float* lhs [[ buffer(0) ]],
                                                device const float* rhs [[ buffer(1) ]],
                                                device float* result [[ buffer(2) ]],
                                                uint id [[ thread_position_in_grid ]]) {
    result[id] = lhs[id] / *rhs;
  }
  
  // Legacy simple matrix multiplication (for small matrices)
  kernel void nsc_matmul_float16_kernel(device const half* a [[ buffer(0) ]],
                                        device const half* b [[ buffer(1) ]],
                                        device half* result [[ buffer(2) ]],
                                        device const NSC_Size* a_size [[ buffer(3) ]],
                                        device const NSC_Size* b_size [[ buffer(4) ]],
                                        uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row >= a_size->rows || col >= b_size->columns) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < a_size->columns; k++) {
      float a_val = a[row * a_size->columns + k];
      float b_val = b[k * b_size->columns + col];
      sum += a_val * b_val;
    }
    
    result[row * b_size->columns + col] = sum;
  }
  
  // MARK: - Matrix Operations (Float32)
  
  // Legacy simple matrix multiplication (for small matrices)
  kernel void nsc_matmul_float_kernel(device const float* a [[ buffer(0) ]],
                                      device const float* b [[ buffer(1) ]],
                                      device float* result [[ buffer(2) ]],
                                      device const NSC_Size* a_size [[ buffer(3) ]],
                                      device const NSC_Size* b_size [[ buffer(4) ]],
                                      uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row >= a_size->rows || col >= b_size->columns) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < a_size->columns; k++) {
      float a_val = a[row * a_size->columns + k];
      float b_val = b[k * b_size->columns + col];
      sum += a_val * b_val;
    }
    
    result[row * b_size->columns + col] = sum;
  }
  
  // Optimized tiled matrix multiplication with shared memory (Float32)
  kernel void nsc_tiled_matmul_float_kernel(device const float* a [[ buffer(0) ]],
                                            device const float* b [[ buffer(1) ]],
                                            device float* result [[ buffer(2) ]],
                                            device const NSC_Size* a_size [[ buffer(3) ]],
                                            device const NSC_Size* b_size [[ buffer(4) ]],
                                            threadgroup float* shared_a [[ threadgroup(0) ]],
                                            threadgroup float* shared_b [[ threadgroup(1) ]],
                                            uint2 thread_id [[ thread_position_in_threadgroup ]],
                                            uint2 threadgroup_id [[ threadgroup_position_in_grid ]],
                                            uint2 threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    
    const uint TILE_SIZE = 16; // Must match threadgroup size
    
    uint row = threadgroup_id.y * TILE_SIZE + thread_id.y;
    uint col = threadgroup_id.x * TILE_SIZE + thread_id.x;
    
    float sum = 0.0f;
    
    uint num_tiles = (a_size->columns + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint tile = 0; tile < num_tiles; tile++) {
      // Load tile of A into shared memory
      uint a_row = row;
      uint a_col = tile * TILE_SIZE + thread_id.x;
      if (a_row < a_size->rows && a_col < a_size->columns) {
        shared_a[thread_id.y * TILE_SIZE + thread_id.x] = a[a_row * a_size->columns + a_col];
      } else {
        shared_a[thread_id.y * TILE_SIZE + thread_id.x] = 0.0f;
      }
      
      // Load tile of B into shared memory
      uint b_row = tile * TILE_SIZE + thread_id.y;
      uint b_col = col;
      if (b_row < b_size->rows && b_col < b_size->columns) {
        shared_b[thread_id.y * TILE_SIZE + thread_id.x] = b[b_row * b_size->columns + b_col];
      } else {
        shared_b[thread_id.y * TILE_SIZE + thread_id.x] = 0.0f;
      }
      
      // Synchronize to ensure all threads have loaded their data
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Compute partial dot product for this tile
      for (uint k = 0; k < TILE_SIZE; k++) {
        sum += shared_a[thread_id.y * TILE_SIZE + k] * shared_b[k * TILE_SIZE + thread_id.x];
      }
      
      // Synchronize before loading next tile
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    if (row < a_size->rows && col < b_size->columns) {
      result[row * b_size->columns + col] = sum;
    }
  }
  
  // Batched matrix multiplication for neural networks (Float32)
  kernel void nsc_batched_matmul_float_kernel(device const float* a [[ buffer(0) ]],
                                              device const float* b [[ buffer(1) ]],
                                              device float* result [[ buffer(2) ]],
                                              device const NSC_Size* a_size [[ buffer(3) ]],
                                              device const NSC_Size* b_size [[ buffer(4) ]],
                                              device const uint* batch_size [[ buffer(5) ]],
                                              uint3 id [[ thread_position_in_grid ]]) {
    uint batch = id.z;
    uint row = id.y;
    uint col = id.x;
    
    if (batch >= *batch_size || row >= a_size->rows || col >= b_size->columns) return;
    
    uint a_offset = batch * a_size->rows * a_size->columns;
    uint b_offset = batch * b_size->rows * b_size->columns;
    uint result_offset = batch * a_size->rows * b_size->columns;
    
    float sum = 0.0f;
    for (uint k = 0; k < a_size->columns; k++) {
      float a_val = a[a_offset + row * a_size->columns + k];
      float b_val = b[b_offset + k * b_size->columns + col];
      sum += a_val * b_val;
    }
    
    result[result_offset + row * b_size->columns + col] = sum;
  }
  
  // Legacy 2D convolution (Float32)
  kernel void nsc_conv2d_float_kernel(device const float* signal [[ buffer(0) ]],
                                      device const float* filter [[ buffer(1) ]],
                                      device float* result [[ buffer(2) ]],
                                      device const NSC_Size* input_size [[ buffer(3) ]],
                                      device const NSC_Size* filter_size [[ buffer(4) ]],
                                      device const NSC_Size* stride [[ buffer(5) ]],
                                      device const NSC_Size* result_size [[ buffer(6) ]],
                                      device const int* padding_type [[ buffer(7) ]],
                                      device const int* pad_top [[ buffer(8) ]],
                                      device const int* pad_left [[ buffer(9) ]],
                                      uint2 id [[ thread_position_in_grid ]]) {
    uint result_row = id.y;
    uint result_col = id.x;
    
    if (result_row >= result_size->rows || result_col >= result_size->columns) return;
    
    float sum = 0.0f;
    
    for (uint fr = 0; fr < filter_size->rows; fr++) {
      for (uint fc = 0; fc < filter_size->columns; fc++) {
        int signal_row = (int)(result_row * stride->rows + fr) - *pad_top;
        int signal_col = (int)(result_col * stride->columns + fc) - *pad_left;
        
        if (signal_row >= 0 && signal_row < (int)input_size->rows && 
            signal_col >= 0 && signal_col < (int)input_size->columns) {
          float signal_val = signal[signal_row * input_size->columns + signal_col];
          float filter_val = filter[fr * filter_size->columns + fc];
          sum += signal_val * filter_val;
        }
      }
    }
    
    result[result_row * result_size->columns + result_col] = sum;
  }
  
  // Optimized batched convolution for neural networks (Float32)
  kernel void nsc_batched_conv2d_float_kernel(device const float* signal [[ buffer(0) ]],
                                              device const float* filter [[ buffer(1) ]],
                                              device float* result [[ buffer(2) ]],
                                              device const NSC_Size* input_size [[ buffer(3) ]],  // [height, width]
                                              device const NSC_Size* filter_size [[ buffer(4) ]], // [height, width]
                                              device const NSC_Size* stride [[ buffer(5) ]],
                                              device const NSC_Size* result_size [[ buffer(6) ]], // [height, width]
                                              device const int* padding_type [[ buffer(7) ]],
                                              device const int* pad_top [[ buffer(8) ]],
                                              device const int* pad_left [[ buffer(9) ]],
                                              device const uint* batch_size [[ buffer(10) ]],
                                              device const uint* in_channels [[ buffer(11) ]],
                                              device const uint* out_channels [[ buffer(12) ]],
                                              uint3 id [[ thread_position_in_grid ]]) {
    uint batch = id.z;
    uint result_row = id.y;
    uint result_col = id.x;
    
    if (batch >= *batch_size || result_row >= result_size->rows || result_col >= result_size->columns) return;
    
    uint input_spatial_size = input_size->rows * input_size->columns;
    uint result_spatial_size = result_size->rows * result_size->columns;
    uint filter_spatial_size = filter_size->rows * filter_size->columns;
    
    // For each output channel
    for (uint out_ch = 0; out_ch < *out_channels; out_ch++) {
      float sum = 0.0f;
      
      // For each input channel
      for (uint in_ch = 0; in_ch < *in_channels; in_ch++) {
        // Convolution computation
        for (uint fr = 0; fr < filter_size->rows; fr++) {
          for (uint fc = 0; fc < filter_size->columns; fc++) {
            int signal_row = (int)(result_row * stride->rows + fr) - *pad_top;
            int signal_col = (int)(result_col * stride->columns + fc) - *pad_left;
            
            if (signal_row >= 0 && signal_row < (int)input_size->rows && 
                signal_col >= 0 && signal_col < (int)input_size->columns) {
              
              uint signal_idx = batch * (*in_channels) * input_spatial_size + 
                               in_ch * input_spatial_size + 
                               signal_row * input_size->columns + signal_col;
              
              uint filter_idx = out_ch * (*in_channels) * filter_spatial_size + 
                               in_ch * filter_spatial_size + 
                               fr * filter_size->columns + fc;
              
              sum += signal[signal_idx] * filter[filter_idx];
            }
          }
        }
      }
      
      uint result_idx = batch * (*out_channels) * result_spatial_size + 
                       out_ch * result_spatial_size + 
                       result_row * result_size->columns + result_col;
      result[result_idx] = sum;
    }
  }
  
  // Optimized im2col-based convolution for better memory access patterns (Float32)
  kernel void nsc_im2col_conv2d_float_kernel(device const float* signal [[ buffer(0) ]],
                                             device const float* filter [[ buffer(1) ]],
                                             device float* result [[ buffer(2) ]],
                                             device const NSC_Size* input_size [[ buffer(3) ]],
                                             device const NSC_Size* filter_size [[ buffer(4) ]],
                                             device const NSC_Size* stride [[ buffer(5) ]],
                                             device const NSC_Size* result_size [[ buffer(6) ]],
                                             device const int* pad_top [[ buffer(7) ]],
                                             device const int* pad_left [[ buffer(8) ]],
                                             threadgroup float* shared_data [[ threadgroup(0) ]],
                                             uint2 thread_id [[ thread_position_in_threadgroup ]],
                                             uint2 threadgroup_id [[ threadgroup_position_in_grid ]],
                                             uint2 threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    
    const uint TILE_SIZE = 16;
    uint result_row = threadgroup_id.y * TILE_SIZE + thread_id.y;
    uint result_col = threadgroup_id.x * TILE_SIZE + thread_id.x;
    
    if (result_row >= result_size->rows || result_col >= result_size->columns) return;
    
    float sum = 0.0f;
    
    // Unroll small filters for better performance
    if (filter_size->rows == 3 && filter_size->columns == 3) {
      // Optimized 3x3 convolution
      for (uint fr = 0; fr < 3; fr++) {
        for (uint fc = 0; fc < 3; fc++) {
          int signal_row = (int)(result_row * stride->rows + fr) - *pad_top;
          int signal_col = (int)(result_col * stride->columns + fc) - *pad_left;
          
          if (signal_row >= 0 && signal_row < (int)input_size->rows && 
              signal_col >= 0 && signal_col < (int)input_size->columns) {
            float signal_val = signal[signal_row * input_size->columns + signal_col];
            float filter_val = filter[fr * 3 + fc];
            sum += signal_val * filter_val;
          }
        }
      }
    } else {
      // General convolution
      for (uint fr = 0; fr < filter_size->rows; fr++) {
        for (uint fc = 0; fc < filter_size->columns; fc++) {
          int signal_row = (int)(result_row * stride->rows + fr) - *pad_top;
          int signal_col = (int)(result_col * stride->columns + fc) - *pad_left;
          
          if (signal_row >= 0 && signal_row < (int)input_size->rows && 
              signal_col >= 0 && signal_col < (int)input_size->columns) {
            float signal_val = signal[signal_row * input_size->columns + signal_col];
            float filter_val = filter[fr * filter_size->columns + fc];
            sum += signal_val * filter_val;
          }
        }
      }
    }
    
    result[result_row * result_size->columns + result_col] = sum;
  }
  
  // MARK: - Transposed Convolution Operations (Float32)
  
  kernel void nsc_transconv2d_float_kernel(device const float* signal [[ buffer(0) ]],
                                           device const float* filter [[ buffer(1) ]],
                                           device float* result [[ buffer(2) ]],
                                           device const NSC_Size* input_size [[ buffer(3) ]],
                                           device const NSC_Size* filter_size [[ buffer(4) ]],
                                           device const NSC_Size* stride [[ buffer(5) ]],
                                           device const NSC_Size* result_size [[ buffer(6) ]],
                                           uint2 id [[ thread_position_in_grid ]]) {
    uint input_row = id.y;
    uint input_col = id.x;
    
    if (input_row >= input_size->rows || input_col >= input_size->columns) return;
    
    float signal_val = signal[input_row * input_size->columns + input_col];
    
    uint i_prime = input_row * stride->rows;
    uint j_prime = input_col * stride->columns;
    
    for (uint fr = 0; fr < filter_size->rows; fr++) {
      for (uint fc = 0; fc < filter_size->columns; fc++) {
        uint result_row = i_prime + fr;
        uint result_col = j_prime + fc;
        
        if (result_row < result_size->rows && result_col < result_size->columns) {
          float filter_val = filter[fr * filter_size->columns + fc];
          uint result_idx = result_row * result_size->columns + result_col;
          
          
          // Metal doesn't support atomic operations on half precision
          // For transpose convolution, ensure result buffer is zero-initialized before kernel execution
          // Race conditions are generally acceptable for this operation type
          result[result_idx] += signal_val * filter_val;
        }
      }
    }
  }


// MARK: - Activation Functions

kernel void nsc_activation_kernel(const device float* data [[ buffer(0) ]],
                                  device float* results [[ buffer(1) ]],
                                  const device uint& activationType [[ buffer(2) ]],
                                  const device float& limit [[ buffer(3) ]],
                                  const uint tgPos [[ threadgroup_position_in_grid ]],
                                  const uint tPerTg [[ threads_per_threadgroup ]],
                                  const uint tPos [[ thread_position_in_threadgroup ]]) {
  
  uint resultIndex = tgPos * tPerTg + tPos;
  
  float completeValue = data[resultIndex];

  if (activationType == 0) { //relu
    results[resultIndex] = max((float)0, completeValue);
    
  } else if (activationType == 1) { //leaky relu
    if (completeValue < 0) {
      results[resultIndex] = limit * completeValue;
    } else {
      results[resultIndex] = completeValue;
    }
      
  } else if (activationType == 2) { //sigmoid
    results[resultIndex] = 1.0 / (1.0 + exp(-completeValue));

  } else if (activationType == 3) { //swish
    float sigmoid = 1.0 / (1.0 + exp(-completeValue));
    results[resultIndex] = completeValue * sigmoid;
    
  } else if (activationType == 4) { //tanH
    float denom = 1.0 + exp(-2 * completeValue);
    results[resultIndex] = (2.0 / denom) - 1.0;
    
  } else if (activationType == 5) { //none
    results[resultIndex] = completeValue;
    
  } else if (activationType == 6) { //selu
    const float alpha = 1.6732632423543772848170429916717;
    const float scale = 1.0507009873554804934193349852946;
    if (completeValue <= 0) {
      results[resultIndex] = scale * alpha * (exp(completeValue) - 1.0);
    } else {
      results[resultIndex] = scale * completeValue;
    }
    
  } else if (activationType == 7) { //gelu
    const float sqrt_2_pi = 0.7978845608028654; // sqrt(2/pi)
    const float a = 0.044715;
    float tanh_input = sqrt_2_pi * (completeValue + a * pow(completeValue, 3));
    results[resultIndex] = 0.5 * completeValue * (1.0 + tanh(tanh_input));
  }

}

kernel void nsc_derivate_kernel(const device float* data [[ buffer(0) ]],
                                device float* results [[ buffer(1) ]],
                                const device uint& activationType [[ buffer(2) ]],
                                const device float& limit [[ buffer(3) ]],
                                const uint tgPos [[ threadgroup_position_in_grid ]],
                                const uint tPerTg [[ threads_per_threadgroup ]],
                                const uint tPos [[ thread_position_in_threadgroup ]]) {

  uint resultIndex = tgPos * tPerTg + tPos;
  
  float completeValue = data[resultIndex];
  
  float value = completeValue;
  
  if (activationType == 0) { //relu
    if (completeValue >= 0) {
      value = 1;
    } else {
      value = 0;
    }
    
  } else if (activationType == 1) { //leaky relu
    if (completeValue > 0) {
      value = 1;
    } else {
      value = limit;
    }
    
  } else if (activationType == 2) { //sigmoid
    float sig = 1.0 / (1.0 + exp(-completeValue));
    value = sig * (1 - sig);
    
  } else if (activationType == 3) { //swish
    value = (exp(-completeValue) * (completeValue + 1) + 1) / pow((1 + exp(-completeValue)), 2);
    
  } else if (activationType == 4) { //tanH
    float denom = 1.0 + exp(-2 * completeValue);
    float tanActivate = (2.0 / denom) - 1.0;
    value = 1 - (pow(tanActivate, 2));
    
  } else if (activationType == 5) { //none
    results[resultIndex] = 1;
    
  } else if (activationType == 6) { //selu
    const float alpha = 1.6732632423543772848170429916717;
    const float scale = 1.0507009873554804934193349852946;
    if (completeValue <= 0) {
      value = scale * alpha * exp(completeValue);
    } else {
      value = scale;
    }
    
  } else if (activationType == 7) { //gelu
    const float sqrt_2_pi = 0.7978845608028654; // sqrt(2/pi)
    const float a = 0.044715;
    float x_cubed = pow(completeValue, 3);
    float tanh_input = sqrt_2_pi * (completeValue + a * x_cubed);
    float tanh_val = tanh(tanh_input);
    float sech_val = 1.0 - tanh_val * tanh_val; // sech^2(x) = 1 - tanh^2(x)
    value = 0.5 * (1.0 + tanh_val) + 0.5 * completeValue * sech_val * sqrt_2_pi * (1.0 + 3.0 * a * completeValue * completeValue);
  }
  
  results[resultIndex] = value;
}

kernel void nsc_activation_half_kernel(const device half* data [[ buffer(0) ]],
                                       device half* results [[ buffer(1) ]],
                                       const device uint& activationType [[ buffer(2) ]],
                                       const device half& limit [[ buffer(3) ]],
                                       const uint tgPos [[ threadgroup_position_in_grid ]],
                                       const uint tPerTg [[ threads_per_threadgroup ]],
                                       const uint tPos [[ thread_position_in_threadgroup ]]) {
  
  uint resultIndex = tgPos * tPerTg + tPos;
  
  half completeValue = data[resultIndex];

  if (activationType == 0) { //relu
    results[resultIndex] = max((half)0, completeValue);
    
  } else if (activationType == 1) { //leaky relu
    if (completeValue < 0) {
      results[resultIndex] = limit * completeValue;
    } else {
      results[resultIndex] = completeValue;
    }
      
  } else if (activationType == 2) { //sigmoid
    results[resultIndex] = 1.0h / (1.0h + exp(-completeValue));

  } else if (activationType == 3) { //swish
    half sigmoid = 1.0h / (1.0h + exp(-completeValue));
    results[resultIndex] = completeValue * sigmoid;
    
  } else if (activationType == 4) { //tanH
    half denom = 1.0h + exp(-2 * completeValue);
    results[resultIndex] = (2.0h / denom) - 1.0h;
    
  } else if (activationType == 5) { //none
    results[resultIndex] = completeValue;
    
  } else if (activationType == 6) { //selu
    const half alpha = 1.6732632423543772848170429916717h;
    const half scale = 1.0507009873554804934193349852946h;
    if (completeValue <= 0) {
      results[resultIndex] = scale * alpha * (exp(completeValue) - 1.0h);
    } else {
      results[resultIndex] = scale * completeValue;
    }
    
  } else if (activationType == 7) { //gelu
    const half sqrt_2_pi = 0.7978845608028654h; // sqrt(2/pi)
    const half a = 0.044715h;
    half tanh_input = sqrt_2_pi * (completeValue + a * pow(completeValue, 3));
    results[resultIndex] = 0.5h * completeValue * (1.0h + tanh(tanh_input));
  }

}

kernel void nsc_derivate_half_kernel(const device half* data [[ buffer(0) ]],
                                     device half* results [[ buffer(1) ]],
                                     const device uint& activationType [[ buffer(2) ]],
                                     const device half& limit [[ buffer(3) ]],
                                     const uint tgPos [[ threadgroup_position_in_grid ]],
                                     const uint tPerTg [[ threads_per_threadgroup ]],
                                     const uint tPos [[ thread_position_in_threadgroup ]]) {

  uint resultIndex = tgPos * tPerTg + tPos;
  
  half completeValue = data[resultIndex];
  
  half value = completeValue;
  
  if (activationType == 0) { //relu
    if (completeValue >= 0) {
      value = 1;
    } else {
      value = 0;
    }
    
  } else if (activationType == 1) { //leaky relu
    if (completeValue > 0) {
      value = 1;
    } else {
      value = limit;
    }
    
  } else if (activationType == 2) { //sigmoid
    half sig = 1.0h / (1.0h + exp(-completeValue));
    value = sig * (1 - sig);
    
  } else if (activationType == 3) { //swish
    value = (exp(-completeValue) * (completeValue + 1) + 1) / pow((1 + exp(-completeValue)), 2);
    
  } else if (activationType == 4) { //tanH
    half denom = 1.0h + exp(-2 * completeValue);
    half tanActivate = (2.0h / denom) - 1.0h;
    value = 1 - (pow(tanActivate, 2));
    
  } else if (activationType == 5) { //none
    results[resultIndex] = 1;
    
  } else if (activationType == 6) { //selu
    const half alpha = 1.6732632423543772848170429916717h;
    const half scale = 1.0507009873554804934193349852946h;
    if (completeValue <= 0) {
      value = scale * alpha * exp(completeValue);
    } else {
      value = scale;
    }
    
  } else if (activationType == 7) { //gelu
    const half sqrt_2_pi = 0.7978845608028654h; // sqrt(2/pi)
    const half a = 0.044715h;
    half x_cubed = pow(completeValue, 3);
    half tanh_input = sqrt_2_pi * (completeValue + a * x_cubed);
    half tanh_val = tanh(tanh_input);
    half sech_val = 1.0h - tanh_val * tanh_val; // sech^2(x) = 1 - tanh^2(x)
    value = 0.5h * (1.0h + tanh_val) + 0.5h * completeValue * sech_val * sqrt_2_pi * (1.0h + 3.0h * a * completeValue * completeValue);
  }
  
  results[resultIndex] = value;
}
