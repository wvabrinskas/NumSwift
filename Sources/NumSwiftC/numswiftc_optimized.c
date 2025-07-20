#include "include/numswiftc.h"
#include <time.h>
#include <string.h>
#include <stdlib.h>

// Platform-specific SIMD includes
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

// Memory alignment for SIMD
#define ALIGN_BYTES 32
#define ALIGN(x) __attribute__((aligned(x)))

// SIMD-optimized matrix multiplication that matches nsc_matmul exactly
extern void nsc_matmul_optimized(NSC_Size a_size,
                                 NSC_Size b_size,
                                 float *const *a,
                                 float *const *b,
                                 float **result) {
  
  int rowFirst = a_size.rows;
  int columnFirst = a_size.columns;
  int columnSecond = b_size.columns;
  
  // Use same loop structure as original but with SIMD optimization
  for(int i = 0; i < rowFirst; ++i) {
    for(int j = 0; j < columnSecond; ++j) {
      
      // Vectorize the inner k loop
      int k = 0;
      float sum = 0.0f;
      
#ifdef __AVX2__
      // Process 8 elements at a time with AVX2
      __m256 sum_vec = _mm256_setzero_ps();
      for (; k <= columnFirst - 8; k += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i][k]);
        __m256 b_vec = _mm256_set_ps(b[k+7][j], b[k+6][j], b[k+5][j], b[k+4][j],
                                     b[k+3][j], b[k+2][j], b[k+1][j], b[k][j]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
      }
      
      // Horizontal sum
      __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
      __m128 sum_low = _mm256_castps256_ps128(sum_vec);
      __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
      __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
      __m128 sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));
      sum = _mm_cvtss_f32(sum_32);
      
#elif defined(__ARM_NEON__)
      // Process 4 elements at a time with NEON
      float32x4_t sum_vec = vdupq_n_f32(0.0f);
      for (; k <= columnFirst - 4; k += 4) {
        float32x4_t a_vec = vld1q_f32(&a[i][k]);
        float32x4_t b_vec = {b[k][j], b[k+1][j], b[k+2][j], b[k+3][j]};
        sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
      }
      
      // Horizontal sum
      sum = vaddvq_f32(sum_vec);
#endif
      
      // Handle remaining elements (scalar)
      for(; k < columnFirst; ++k) {
        sum += a[i][k] * b[k][j];
      }
      
      result[i][j] += sum;
    }
  }
}

// Optimized convolution that matches nsc_conv2d exactly
extern void nsc_conv2d_optimized(float *const *signal,
                                 float *const *filter,
                                 float **result,
                                 NSC_Size stride,
                                 NSC_Padding padding,
                                 NSC_Size filter_size,
                                 NSC_Size input_size) {
  
  // Get padding values using the same calculation as original
  int paddingLeft, paddingRight, paddingBottom, paddingTop;
  int *pad_l_ptr = &paddingLeft;
  int *pad_r_ptr = &paddingRight;
  int *pad_b_ptr = &paddingBottom;
  int *pad_t_ptr = &paddingTop;

  nsc_padding_calculation(stride, padding, filter_size, input_size,
                          pad_t_ptr, pad_b_ptr, pad_l_ptr, pad_r_ptr);
  
  // Use same variable names as original for consistency
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  int strideR = stride.rows;
  int strideC = stride.columns;
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  int rf = filterRows;
  int cf = filterColumns;
  int rd = inputRows + paddingTop + paddingBottom;
  int cd = inputColumns + paddingLeft + paddingRight;
  
  int max_r = rd - rf + 1;
  int max_c = cd - cf + 1;
  
  // Create working signal if padding is needed (same as original)
  float **working_signal = NULL;
  if (padding == same) {
    working_signal = (float**)malloc(rd * sizeof(float*));
    if (!working_signal) return;
    
    for (int i = 0; i < rd; i++) {
      working_signal[i] = (float*)calloc(cd, sizeof(float));
      if (!working_signal[i]) {
        // Cleanup on failure
        for (int j = 0; j < i; j++) free(working_signal[j]);
        free(working_signal);
        return;
      }
    }
    
    // Copy input data to center of working signal
    for (int r = 0; r < inputRows; r++) {
      for (int c = 0; c < inputColumns; c++) {
        working_signal[r + paddingTop][c + paddingLeft] = signal[r][c];
      }
    }
  }
  
  // Main convolution loop - exactly like original but with SIMD optimization
  int result_index_r = 0;
  for (int r = 0; r < max_r; r += strideR) {
    int result_index_c = 0;
    for (int c = 0; c < max_c; c += strideC) {
      float sum = 0.0f;
      
      // Vectorized filter application when possible
      int filter_elements = rf * cf;
      
#ifdef __AVX2__
      // Try to vectorize the inner filter loops
      __m256 sum_vec = _mm256_setzero_ps();
      int vec_count = 0;
      
      // Process filter elements in chunks of 8
      float signal_vals[8], filter_vals[8];
      
      for (int fr = 0; fr < filterRows; fr++) {
        for (int fc = 0; fc < filterColumns; fc++) {
          int current_data_row = r + fr;
          int current_data_col = c + fc;
          
          float s_data = 0.0f;
          if (padding == same) {
            s_data = working_signal[current_data_row][current_data_col];
          } else {
            s_data = signal[current_data_row][current_data_col];
          }
          
          float f_data = filter[fr][fc];
          
          // Accumulate for vectorization
          signal_vals[vec_count] = s_data;
          filter_vals[vec_count] = f_data;
          vec_count++;
          
          // Process when we have 8 elements or at the end
          if (vec_count == 8 || (fr == filterRows-1 && fc == filterColumns-1)) {
            // Pad remaining elements with zeros if needed
            while (vec_count < 8) {
              signal_vals[vec_count] = 0.0f;
              filter_vals[vec_count] = 0.0f;
              vec_count++;
            }
            
            __m256 s_vec = _mm256_loadu_ps(signal_vals);
            __m256 f_vec = _mm256_loadu_ps(filter_vals);
            sum_vec = _mm256_fmadd_ps(s_vec, f_vec, sum_vec);
            vec_count = 0;
          }
        }
      }
      
      // Horizontal sum of vector
      __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
      __m128 sum_low = _mm256_castps256_ps128(sum_vec);
      __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
      __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
      __m128 sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));
      sum = _mm_cvtss_f32(sum_32);
      
#elif defined(__ARM_NEON__)
      // NEON optimization
      float32x4_t sum_vec = vdupq_n_f32(0.0f);
      int vec_count = 0;
      float signal_vals[4], filter_vals[4];
      
      for (int fr = 0; fr < filterRows; fr++) {
        for (int fc = 0; fc < filterColumns; fc++) {
          int current_data_row = r + fr;
          int current_data_col = c + fc;
          
          float s_data = 0.0f;
          if (padding == same) {
            s_data = working_signal[current_data_row][current_data_col];
          } else {
            s_data = signal[current_data_row][current_data_col];
          }
          
          float f_data = filter[fr][fc];
          
          signal_vals[vec_count] = s_data;
          filter_vals[vec_count] = f_data;
          vec_count++;
          
          if (vec_count == 4 || (fr == filterRows-1 && fc == filterColumns-1)) {
            while (vec_count < 4) {
              signal_vals[vec_count] = 0.0f;
              filter_vals[vec_count] = 0.0f;
              vec_count++;
            }
            
            float32x4_t s_vec = vld1q_f32(signal_vals);
            float32x4_t f_vec = vld1q_f32(filter_vals);
            sum_vec = vfmaq_f32(sum_vec, s_vec, f_vec);
            vec_count = 0;
          }
        }
      }
      
      sum = vaddvq_f32(sum_vec);
      
#else
      // Scalar version with manual unrolling - same as original
      for (int fr = 0; fr < filterRows; fr++) {
        for (int fc = 0; fc < filterColumns; fc++) {
          int current_data_row = r + fr;
          int current_data_col = c + fc;
          
          float s_data = 0.0f;
          if (padding == same) {
            s_data = working_signal[current_data_row][current_data_col];
          } else {
            s_data = signal[current_data_row][current_data_col];
          }
          
          float f_data = filter[fr][fc];
          sum += s_data * f_data;
        }
      }
#endif
      
      result[result_index_r][result_index_c] = sum;
      result_index_c++;
    }
    result_index_r++;
  }
  
  // Cleanup working signal if allocated
  if (working_signal) {
    for (int i = 0; i < rd; i++) {
      free(working_signal[i]);
    }
    free(working_signal);
  }
}

// Optimized Float16 matrix multiplication that matches nsc_matmul_16 exactly
extern void nsc_matmul_16_optimized(NSC_Size a_size,
                                    NSC_Size b_size,
                                    __fp16 *const *a,
                                    __fp16 *const *b,
                                    __fp16 **result) {
  
  int rowFirst = a_size.rows;
  int columnFirst = a_size.columns;
  int columnSecond = b_size.columns;
  
  // Use same loop structure as original but with SIMD optimization
  for(int i = 0; i < rowFirst; ++i) {
    for(int j = 0; j < columnSecond; ++j) {
      
      // Vectorize the inner k loop
      int k = 0;
      __fp16 sum = 0.0;
      
#ifdef __ARM_NEON__
      // Process 8 elements at a time with NEON Float16
      float16x8_t sum_vec = vdupq_n_f16(0.0);
      for (; k <= columnFirst - 8; k += 8) {
        float16x8_t a_vec = vld1q_f16(&a[i][k]);
        float16x8_t b_vec = {b[k][j], b[k+1][j], b[k+2][j], b[k+3][j],
                             b[k+4][j], b[k+5][j], b[k+6][j], b[k+7][j]};
        sum_vec = vfmaq_f16(sum_vec, a_vec, b_vec);
      }
      
      // Horizontal sum of Float16 vector
      float16x4_t sum_low = vget_low_f16(sum_vec);
      float16x4_t sum_high = vget_high_f16(sum_vec);
      float16x4_t sum_total = vadd_f16(sum_low, sum_high);
      
      // Further reduce to scalar
      sum += vget_lane_f16(sum_total, 0) + vget_lane_f16(sum_total, 1) +
             vget_lane_f16(sum_total, 2) + vget_lane_f16(sum_total, 3);
#endif
      
      // Handle remaining elements (scalar, same as original)
      for(; k < columnFirst; ++k) {
        sum += a[i][k] * b[k][j];
      }
      
      result[i][j] += sum;
    }
  }
}

// Optimized transpose that matches nsc_transpose_2d exactly  
extern void nsc_transpose_2d_optimized(float *const *input,
                                       float **result,
                                       NSC_Size input_size) {
  
  int rows = input_size.rows;
  int cols = input_size.columns;
  
  // Use same loop structure as original but with SIMD optimization where beneficial
  for (int i = 0; i < rows; i++) {
    int j = 0;
    
#ifdef __AVX2__
    // Process 8 elements at a time with AVX2 (load from input row, scatter to result)
    for (; j <= cols - 8; j += 8) {
      __m256 values = _mm256_loadu_ps(&input[i][j]);
      
      // Extract and store individual elements (transpose requires scatter)
      result[j][i] = _mm256_cvtss_f32(values);
      result[j+1][i] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, values, 1));
      result[j+2][i] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, values, 2));
      result[j+3][i] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, values, 3));
      
      __m128 high = _mm256_extractf128_ps(values, 1);
      result[j+4][i] = _mm_cvtss_f32(high);
      result[j+5][i] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 1));
      result[j+6][i] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 2));
      result[j+7][i] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 3));
    }
    
#elif defined(__ARM_NEON__)
    // Process 4 elements at a time with NEON
    for (; j <= cols - 4; j += 4) {
      float32x4_t values = vld1q_f32(&input[i][j]);
      
      // Extract and store individual elements
      result[j][i] = vgetq_lane_f32(values, 0);
      result[j+1][i] = vgetq_lane_f32(values, 1);
      result[j+2][i] = vgetq_lane_f32(values, 2);
      result[j+3][i] = vgetq_lane_f32(values, 3);
    }
#endif
    
    // Handle remaining elements (same as original)
    for (; j < cols; j++) {
      result[j][i] = input[i][j];
    }
  }
}

// Optimized zero padding that matches nsc_specific_zero_pad_2d exactly
extern void nsc_specific_zero_pad_2d_optimized(float *const *input,
                                               float **result,
                                               NSC_Size input_size,
                                               int paddingTop,
                                               int paddingBottom,
                                               int paddingLeft,
                                               int paddingRight) {
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int padded_row_total = inputRows + paddingLeft + paddingRight;
  int padded_col_total = inputColumns + paddingTop + paddingBottom;

  if (result == NULL || input == NULL)
    return;
  
  // Use same loop structure as original but with SIMD optimization for copying
  for (int r = 0; r < inputRows; r++) {
    int padded_r = r + paddingTop;
    int c = 0;
    
#ifdef __AVX2__
    // Process 8 elements at a time with AVX2
    for (; c <= inputColumns - 8; c += 8) {
      int padded_c = c + paddingLeft;
      __m256 values = _mm256_loadu_ps(&input[r][c]);
      _mm256_storeu_ps(&result[padded_r][padded_c], values);
    }
    
#elif defined(__ARM_NEON__)
    // Process 4 elements at a time with NEON
    for (; c <= inputColumns - 4; c += 4) {
      int padded_c = c + paddingLeft;
      float32x4_t values = vld1q_f32(&input[r][c]);
      vst1q_f32(&result[padded_r][padded_c], values);
    }
#endif
    
    // Handle remaining elements (same as original)
    for (; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      result[padded_r][padded_c] = input[r][c];
    }
  }
}

// Fast element-wise operations
extern void nsc_add_optimized(float *const *a,
                              float *const *b,
                              float **result,
                              NSC_Size size) {
  
  const int rows = size.rows;
  const int cols = size.columns;
  
  for (int i = 0; i < rows; i++) {
    int j = 0;
    
#ifdef __AVX2__
    // Process 8 floats at a time
    for (; j <= cols - 8; j += 8) {
      __m256 a_vec = _mm256_loadu_ps(&a[i][j]);
      __m256 b_vec = _mm256_loadu_ps(&b[i][j]);
      __m256 result_vec = _mm256_add_ps(a_vec, b_vec);
      _mm256_storeu_ps(&result[i][j], result_vec);
    }
#elif defined(__ARM_NEON__)
    // Process 4 floats at a time
    for (; j <= cols - 4; j += 4) {
      float32x4_t a_vec = vld1q_f32(&a[i][j]);
      float32x4_t b_vec = vld1q_f32(&b[i][j]);
      float32x4_t result_vec = vaddq_f32(a_vec, b_vec);
      vst1q_f32(&result[i][j], result_vec);
    }
#endif
    
    // Handle remaining elements
    for (; j < cols; j++) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }
}

// Optimized Float16 convolution
extern void nsc_conv2d_f16_optimized(__fp16 *const *signal,
                                     __fp16 *const *filter,
                                     __fp16 **result,
                                     NSC_Size stride,
                                     NSC_Padding padding,
                                     NSC_Size filter_size,
                                     NSC_Size input_size) {
  
  const int input_h = input_size.rows;
  const int input_w = input_size.columns;
  const int filter_h = filter_size.rows;
  const int filter_w = filter_size.columns;
  const int stride_h = stride.rows;
  const int stride_w = stride.columns;
  
  // Calculate output dimensions
  int pad_h = 0, pad_w = 0;
  if (padding == same) {
    pad_h = filter_h / 2;
    pad_w = filter_w / 2;
  }
  
  const int output_h = (input_h + 2 * pad_h - filter_h) / stride_h + 1;
  const int output_w = (input_w + 2 * pad_w - filter_w) / stride_w + 1;
  
  // Direct convolution with SIMD optimization for Float16
  for (int out_y = 0; out_y < output_h; out_y++) {
    for (int out_x = 0; out_x < output_w; out_x++) {
      
#ifdef __ARM_NEON__
      // Use NEON for Float16 accumulation
      float16x8_t sum_vec = vdupq_n_f16(0.0);
      int filter_elements = filter_h * filter_w;
      int simd_end = (filter_elements / 8) * 8;
      
      // Vectorized filter application
      int vec_idx = 0;
      for (int fy = 0; fy < filter_h && vec_idx < simd_end; fy++) {
        for (int fx = 0; fx < filter_w && vec_idx < simd_end; fx += 8) {
          int remaining = filter_w - fx;
          if (remaining >= 8) {
            // Load 8 filter values
            float16x8_t filter_vec = vld1q_f16(&filter[fy][fx]);
            
            // Load corresponding input values
            __fp16 input_vals[8];
            for (int i = 0; i < 8; i++) {
              int in_y = out_y * stride_h + fy - pad_h;
              int in_x = out_x * stride_w + (fx + i) - pad_w;
              
              if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                input_vals[i] = signal[in_y][in_x];
              } else {
                input_vals[i] = 0.0;
              }
            }
            
            float16x8_t input_vec = vld1q_f16(input_vals);
            sum_vec = vfmaq_f16(sum_vec, filter_vec, input_vec);
            vec_idx += 8;
          }
        }
      }
      
      // Horizontal sum of vector
      __fp16 sum;
      
      // Manual horizontal sum - more compatible approach
      // Convert to float32 first for better compatibility across ARM versions
      float32x4_t sum_low_f32 = vcvt_f32_f16(vget_low_f16(sum_vec));
      float32x4_t sum_high_f32 = vcvt_f32_f16(vget_high_f16(sum_vec));
      float32x4_t sum_f32 = vaddq_f32(sum_low_f32, sum_high_f32);
      
      // Horizontal sum in float32
      float32x2_t sum_2 = vpadd_f32(vget_low_f32(sum_f32), vget_high_f32(sum_f32));
      float32x2_t sum_1 = vpadd_f32(sum_2, sum_2);
      
      // Convert back to fp16
      sum = (__fp16)vget_lane_f32(sum_1, 0);
      
      
      // Handle remaining elements
      for (int fy = 0; fy < filter_h; fy++) {
        for (int fx = 0; fx < filter_w; fx++) {
          if (fy * filter_w + fx >= simd_end) {
            int in_y = out_y * stride_h + fy - pad_h;
            int in_x = out_x * stride_w + fx - pad_w;
            
            if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
              sum += filter[fy][fx] * signal[in_y][in_x];
            }
          }
        }
      }
      
#else
      // Scalar fallback with loop unrolling
      __fp16 sum = 0.0;
      
      // Unroll inner loops for better performance
      for (int fy = 0; fy < filter_h; fy++) {
        for (int fx = 0; fx < filter_w; fx += 4) {
          // Process 4 elements at a time
          for (int unroll = 0; unroll < 4 && (fx + unroll) < filter_w; unroll++) {
            int in_y = out_y * stride_h + fy - pad_h;
            int in_x = out_x * stride_w + (fx + unroll) - pad_w;
            
            if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
              sum += filter[fy][fx + unroll] * signal[in_y][in_x];
            }
          }
        }
      }
#endif
      
      result[out_y][out_x] = sum;
    }
  }
}

// Additional optimized operations
extern void nsc_multiply_optimized(float *const *a,
                                   float *const *b,
                                   float **result,
                                   NSC_Size size) {
  
  const int rows = size.rows;
  const int cols = size.columns;
  
  for (int i = 0; i < rows; i++) {
    int j = 0;
    
#ifdef __AVX2__
    for (; j <= cols - 8; j += 8) {
      __m256 a_vec = _mm256_loadu_ps(&a[i][j]);
      __m256 b_vec = _mm256_loadu_ps(&b[i][j]);
      __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
      _mm256_storeu_ps(&result[i][j], result_vec);
    }
#elif defined(__ARM_NEON__)
    for (; j <= cols - 4; j += 4) {
      float32x4_t a_vec = vld1q_f32(&a[i][j]);
      float32x4_t b_vec = vld1q_f32(&b[i][j]);
      float32x4_t result_vec = vmulq_f32(a_vec, b_vec);
      vst1q_f32(&result[i][j], result_vec);
    }
#endif
    
    for (; j < cols; j++) {
      result[i][j] = a[i][j] * b[i][j];
    }
  }
}

extern void nsc_subtract_optimized(float *const *a,
                                   float *const *b,
                                   float **result,
                                   NSC_Size size) {
  
  const int rows = size.rows;
  const int cols = size.columns;
  
  for (int i = 0; i < rows; i++) {
    int j = 0;
    
#ifdef __AVX2__
    for (; j <= cols - 8; j += 8) {
      __m256 a_vec = _mm256_loadu_ps(&a[i][j]);
      __m256 b_vec = _mm256_loadu_ps(&b[i][j]);
      __m256 result_vec = _mm256_sub_ps(a_vec, b_vec);
      _mm256_storeu_ps(&result[i][j], result_vec);
    }
#elif defined(__ARM_NEON__)
    for (; j <= cols - 4; j += 4) {
      float32x4_t a_vec = vld1q_f32(&a[i][j]);
      float32x4_t b_vec = vld1q_f32(&b[i][j]);
      float32x4_t result_vec = vsubq_f32(a_vec, b_vec);
      vst1q_f32(&result[i][j], result_vec);
    }
#endif
    
    for (; j < cols; j++) {
      result[i][j] = a[i][j] - b[i][j];
    }
  }
}

// Memory management helpers
extern float** nsc_alloc_aligned_2d_float(int rows, int cols) {
  float **array = (float**)aligned_alloc(ALIGN_BYTES, rows * sizeof(float*));
  if (!array) return NULL;
  
  for (int i = 0; i < rows; i++) {
    array[i] = (float*)aligned_alloc(ALIGN_BYTES, cols * sizeof(float));
    if (!array[i]) {
      // Cleanup on failure
      for (int j = 0; j < i; j++) {
        free(array[j]);
      }
      free(array);
      return NULL;
    }
  }
  return array;
}

extern __fp16** nsc_alloc_aligned_2d_f16(int rows, int cols) {
  __fp16 **array = (__fp16**)aligned_alloc(ALIGN_BYTES, rows * sizeof(__fp16*));
  if (!array) return NULL;
  
  for (int i = 0; i < rows; i++) {
    array[i] = (__fp16*)aligned_alloc(ALIGN_BYTES, cols * sizeof(__fp16));
    if (!array[i]) {
      for (int j = 0; j < i; j++) {
        free(array[j]);
      }
      free(array);
      return NULL;
    }
  }
  return array;
}

extern void nsc_free_2d_float(float **array, int rows) {
  if (!array) return;
  for (int i = 0; i < rows; i++) {
    free(array[i]);
  }
  free(array);
}

extern void nsc_free_2d_f16(__fp16 **array, int rows) {
  if (!array) return;
  for (int i = 0; i < rows; i++) {
    free(array[i]);
  }
  free(array);
}

// Optimized Float16 transpose that matches nsc_transpose_2d_16 exactly
extern void nsc_transpose_2d_16_optimized(__fp16 *const *input,
                                          __fp16 **result,
                                          NSC_Size size) {
  
  int rows = size.rows;
  int cols = size.columns;
  
  // Use same loop structure as original but with NEON optimization
  for (int i = 0; i < rows; i++) {
    int j = 0;
    
#ifdef __ARM_NEON__
    // Process 8 Float16 elements at a time with NEON
    for (; j <= cols - 8; j += 8) {
      float16x8_t values = vld1q_f16(&input[i][j]);
      
      // Extract and store individual elements (transpose requires scatter)
      result[j][i] = vgetq_lane_f16(values, 0);
      result[j+1][i] = vgetq_lane_f16(values, 1);
      result[j+2][i] = vgetq_lane_f16(values, 2);
      result[j+3][i] = vgetq_lane_f16(values, 3);
      result[j+4][i] = vgetq_lane_f16(values, 4);
      result[j+5][i] = vgetq_lane_f16(values, 5);
      result[j+6][i] = vgetq_lane_f16(values, 6);
      result[j+7][i] = vgetq_lane_f16(values, 7);
    }
#endif
    
    // Handle remaining elements (same as original)
    for (; j < cols; j++) {
      result[j][i] = input[i][j];
    }
  }
}


// Optimized Float16 zero padding
extern void nsc_specific_zero_pad_2d_f16_optimized(__fp16 *const *input,
                                                   __fp16 **result,
                                                   NSC_Size input_size,
                                                   int paddingTop,
                                                   int paddingBottom,
                                                   int paddingLeft,
                                                   int paddingRight) {
    
    const int inputRows = input_size.rows;
    const int inputColumns = input_size.columns;
    const int outputRows = inputRows + paddingTop + paddingBottom;
    const int outputCols = inputColumns + paddingLeft + paddingRight;
    
    // First, zero out the entire result with SIMD memset where possible
    for (int r = 0; r < outputRows; r++) {
#ifdef __ARM_NEON__
        // Use NEON to zero out rows efficiently
        int c = 0;
        for (; c <= outputCols - 8; c += 8) {
            float16x8_t zero_vec = vdupq_n_f16(0.0);
            vst1q_f16(&result[r][c], zero_vec);
        }
        // Handle remaining elements
        for (; c < outputCols; c++) {
            result[r][c] = 0.0;
        }
#else
        // Fallback to regular memset
        memset(result[r], 0, outputCols * sizeof(__fp16));
#endif
    }
    
    // Copy input data to center region with optimized copying
    for (int r = 0; r < inputRows; r++) {
        int dest_row = r + paddingTop;
        int c = 0;
        
#ifdef __ARM_NEON__
        // Use NEON for vectorized copy
        for (; c <= inputColumns - 8; c += 8) {
            float16x8_t input_vec = vld1q_f16(&input[r][c]);
            vst1q_f16(&result[dest_row][paddingLeft + c], input_vec);
        }
        // Handle remaining elements
        for (; c < inputColumns; c++) {
            result[dest_row][paddingLeft + c] = input[r][c];
        }
#else
        // Fallback to memcpy
        memcpy(&result[dest_row][paddingLeft],
               input[r],
               inputColumns * sizeof(__fp16));
#endif
    }
}


// Optimized Float16 transposed convolution
extern void nsc_transConv2d_f16_optimized(__fp16 *const *signal,
                                          __fp16 *const *filter,
                                          __fp16 **result,
                                          NSC_Size stride,
                                          NSC_Padding padding,
                                          NSC_Size filter_size,
                                          NSC_Size input_size) {
    
    const int input_h = input_size.rows;
    const int input_w = input_size.columns;
    const int filter_h = filter_size.rows;
    const int filter_w = filter_size.columns;
    const int stride_h = stride.rows;
    const int stride_w = stride.columns;
    
    // Calculate output dimensions
    int pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0;
    
    if (padding == same) {
        pad_left = (int)floor((double)(filter_h - stride_h) / 2.0);
        pad_right = filter_h - stride_h - pad_left;
        pad_top = (int)floor((double)(filter_w - stride_w) / 2.0);
        pad_bottom = filter_w - stride_w - pad_top;
    }
    
    const int output_h_full = (input_h - 1) * stride_h + filter_h;
    const int output_w_full = (input_w - 1) * stride_w + filter_w;
    const int output_h = output_h_full - (pad_top + pad_bottom);
    const int output_w = output_w_full - (pad_left + pad_right);
    
    // Initialize result to zero
    for (int r = 0; r < output_h; r++) {
#ifdef __ARM_NEON__
        int c = 0;
        for (; c <= output_w - 8; c += 8) {
            float16x8_t zero_vec = vdupq_n_f16(0.0);
            vst1q_f16(&result[r][c], zero_vec);
        }
        for (; c < output_w; c++) {
            result[r][c] = 0.0;
        }
#else
        memset(result[r], 0, output_w * sizeof(__fp16));
#endif
    }
    
    // Perform transposed convolution
    // For each input pixel, apply the filter to the output
    for (int in_y = 0; in_y < input_h; in_y++) {
        for (int in_x = 0; in_x < input_w; in_x++) {
            __fp16 input_val = signal[in_y][in_x];
            
            if (input_val == 0.0) continue; // Skip zero inputs for efficiency
            
            // Apply filter centered at (in_y * stride_h, in_x * stride_w)
            for (int fy = 0; fy < filter_h; fy++) {
                for (int fx = 0; fx < filter_w; fx++) {
                    int out_y = in_y * stride_h + fy - pad_top;
                    int out_x = in_x * stride_w + fx - pad_left;
                    
                    // Check bounds
                    if (out_y >= 0 && out_y < output_h && out_x >= 0 && out_x < output_w) {
                        
#ifdef __ARM_NEON__
                        // Use NEON for scalar multiplication and accumulation
                        __fp16 filter_val = filter[fy][fx];
                        __fp16 contribution = input_val * filter_val;
                        result[out_y][out_x] += contribution;
#else
                        result[out_y][out_x] += input_val * filter[fy][fx];
#endif
                    }
                }
            }
        }
    }
}


// Optimized Float32 transposed convolution
extern void nsc_transConv2d_optimized(float *const *signal,
                                      float *const *filter,
                                      float **result,
                                      NSC_Size stride,
                                      NSC_Padding padding,
                                      NSC_Size filter_size,
                                      NSC_Size input_size) {
    
    const int input_h = input_size.rows;
    const int input_w = input_size.columns;
    const int filter_h = filter_size.rows;
    const int filter_w = filter_size.columns;
    const int stride_h = stride.rows;
    const int stride_w = stride.columns;
    
    // Calculate output dimensions
    int pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0;
    
    if (padding == same) {
        pad_left = (int)floor((double)(filter_h - stride_h) / 2.0);
        pad_right = filter_h - stride_h - pad_left;
        pad_top = (int)floor((double)(filter_w - stride_w) / 2.0);
        pad_bottom = filter_w - stride_w - pad_top;
    }
    
    const int output_h_full = (input_h - 1) * stride_h + filter_h;
    const int output_w_full = (input_w - 1) * stride_w + filter_w;
    const int output_h = output_h_full - (pad_top + pad_bottom);
    const int output_w = output_w_full - (pad_left + pad_right);
    
    // Initialize result to zero with SIMD optimization
    for (int r = 0; r < output_h; r++) {
#ifdef __AVX2__
        int c = 0;
        for (; c <= output_w - 8; c += 8) {
            __m256 zero_vec = _mm256_setzero_ps();
            _mm256_storeu_ps(&result[r][c], zero_vec);
        }
        for (; c < output_w; c++) {
            result[r][c] = 0.0f;
        }
#elif defined(__ARM_NEON__)
        int c = 0;
        for (; c <= output_w - 4; c += 4) {
            float32x4_t zero_vec = vdupq_n_f32(0.0f);
            vst1q_f32(&result[r][c], zero_vec);
        }
        for (; c < output_w; c++) {
            result[r][c] = 0.0f;
        }
#else
        memset(result[r], 0, output_w * sizeof(float));
#endif
    }
    
    // Perform transposed convolution
    // For each input pixel, apply the filter to the output
    for (int in_y = 0; in_y < input_h; in_y++) {
        for (int in_x = 0; in_x < input_w; in_x++) {
            float input_val = signal[in_y][in_x];
            
            if (input_val == 0.0f) continue; // Skip zero inputs for efficiency
            
            // Apply filter centered at (in_y * stride_h, in_x * stride_w)
            for (int fy = 0; fy < filter_h; fy++) {
                int fx = 0;
                int out_y = in_y * stride_h + fy - pad_top;
                
                if (out_y < 0 || out_y >= output_h) continue; // Skip if out of bounds
                
#ifdef __AVX2__
                // Process 8 filter elements at a time
                __m256 input_vec = _mm256_set1_ps(input_val);
                
                for (; fx <= filter_w - 8; fx += 8) {
                    int out_x_base = in_x * stride_w + fx - pad_left;
                    
                    // Check if all 8 outputs are in bounds
                    if (out_x_base >= 0 && out_x_base + 7 < output_w) {
                        __m256 filter_vec = _mm256_loadu_ps(&filter[fy][fx]);
                        __m256 contribution = _mm256_mul_ps(input_vec, filter_vec);
                        __m256 result_vec = _mm256_loadu_ps(&result[out_y][out_x_base]);
                        result_vec = _mm256_add_ps(result_vec, contribution);
                        _mm256_storeu_ps(&result[out_y][out_x_base], result_vec);
                    } else {
                        // Handle boundary cases element by element
                        for (int i = 0; i < 8 && (fx + i) < filter_w; i++) {
                            int out_x = out_x_base + i;
                            if (out_x >= 0 && out_x < output_w) {
                                result[out_y][out_x] += input_val * filter[fy][fx + i];
                            }
                        }
                    }
                }
                
#elif defined(__ARM_NEON__)
                // Process 4 filter elements at a time
                float32x4_t input_vec = vdupq_n_f32(input_val);
                
                for (; fx <= filter_w - 4; fx += 4) {
                    int out_x_base = in_x * stride_w + fx - pad_left;
                    
                    // Check if all 4 outputs are in bounds
                    if (out_x_base >= 0 && out_x_base + 3 < output_w) {
                        float32x4_t filter_vec = vld1q_f32(&filter[fy][fx]);
                        float32x4_t contribution = vmulq_f32(input_vec, filter_vec);
                        float32x4_t result_vec = vld1q_f32(&result[out_y][out_x_base]);
                        result_vec = vaddq_f32(result_vec, contribution);
                        vst1q_f32(&result[out_y][out_x_base], result_vec);
                    } else {
                        // Handle boundary cases element by element
                        for (int i = 0; i < 4 && (fx + i) < filter_w; i++) {
                            int out_x = out_x_base + i;
                            if (out_x >= 0 && out_x < output_w) {
                                result[out_y][out_x] += input_val * filter[fy][fx + i];
                            }
                        }
                    }
                }
#endif
                
                // Handle remaining elements
                for (; fx < filter_w; fx++) {
                    int out_x = in_x * stride_w + fx - pad_left;
                    
                    if (out_x >= 0 && out_x < output_w) {
                        result[out_y][out_x] += input_val * filter[fy][fx];
                    }
                }
            }
        }
    }
}
