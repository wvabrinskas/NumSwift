#include <metal_stdlib>
using namespace metal;

// MARK: - Data Structures

struct NSC_Size {
    int rows;
    int columns;
};

struct NSC_IndexedValue {
    half value;
    int index;
};

enum NSC_Padding {
    valid = 0,
    same = 1
};

// MARK: - Basic Array Operations

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

kernel void nsc_conv2d_kernel(device const half* signal [[ buffer(0) ]],
                             device const half* filter [[ buffer(1) ]],
                             device half* result [[ buffer(2) ]],
                             device const NSC_Size* input_size [[ buffer(3) ]],
                             device const NSC_Size* filter_size [[ buffer(4) ]],
                             device const NSC_Size* stride [[ buffer(5) ]],
                             device const NSC_Size* result_size [[ buffer(6) ]],
                             uint2 id [[ thread_position_in_grid ]]) {
    uint result_row = id.y;
    uint result_col = id.x;
    
    if (result_row >= result_size->rows || result_col >= result_size->columns) return;
    
    half sum = 0.0h;
    
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
                
                // Atomic add since multiple threads might write to the same location
                atomic_fetch_add_explicit((device atomic_uint*)&result[result_idx], 
                                         as_type<uint>(signal_val * filter_val), 
                                         memory_order_relaxed);
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
                
                // Atomic add since multiple threads might write to the same location
                atomic_fetch_add_explicit((device atomic_uint*)&result[result_idx], 
                                         as_type<uint>(signal_val * filter_val), 
                                         memory_order_relaxed);
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
                                   device const int* size [[ buffer(4) ]],
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

kernel void nsc_conv2d_float_kernel(device const float* signal [[ buffer(0) ]],
                                   device const float* filter [[ buffer(1) ]],
                                   device float* result [[ buffer(2) ]],
                                   device const NSC_Size* input_size [[ buffer(3) ]],
                                   device const NSC_Size* filter_size [[ buffer(4) ]],
                                   device const NSC_Size* stride [[ buffer(5) ]],
                                   device const NSC_Size* result_size [[ buffer(6) ]],
                                   uint2 id [[ thread_position_in_grid ]]) {
    uint result_row = id.y;
    uint result_col = id.x;
    
    if (result_row >= result_size->rows || result_col >= result_size->columns) return;
    
    float sum = 0.0f;
    
    for (uint fr = 0; fr < filter_size->rows; fr++) {
        for (uint fc = 0; fc < filter_size->columns; fc++) {
            uint signal_row = result_row * stride->rows + fr;
            uint signal_col = result_col * stride->columns + fc;
            
            if (signal_row < input_size->rows && signal_col < input_size->columns) {
                float signal_val = signal[signal_row * input_size->columns + signal_col];
                float filter_val = filter[fr * filter_size->columns + fc];
                sum += signal_val * filter_val;
            }
        }
    }
    
    result[result_row * result_size->columns + result_col] = sum;
}