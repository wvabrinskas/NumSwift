
#include "include/numswiftc.h"
#include "time.h"
#ifdef __ARM_NEON
#include <arm_neon.h>

// Forward declarations for Winograd functions
static inline void winograd_matmul_blocked_f32(float *const *a, float *const *b, float **result,
                                              int rows, int cols_a, int cols_b);
static inline void winograd_matmul_blocked_f16(__fp16 *const *a, __fp16 *const *b, __fp16 **result,
                                              int rows, int cols_a, int cols_b);
#endif

extern void nsc_matmul_16(NSC_Size a_size,
                          NSC_Size b_size,
                          __fp16 *const *a,
                          __fp16 *const *b,
                          __fp16 **result) {
  
  int rowFirst = a_size.rows;
  int columnFirst = a_size.columns;
  int columnSecond = b_size.columns;
  
#ifdef __ARM_NEON
  // Use Winograd blocked multiplication for larger matrices
  if (rowFirst >= 4 && columnFirst >= 4 && columnSecond >= 4) {
    winograd_matmul_blocked_f16(a, b, result, rowFirst, columnFirst, columnSecond);
    return;
  }
#endif
  
  // Standard matrix multiplication for small matrices or non-ARM platforms
  for(int i = 0; i < rowFirst; ++i) {
    for(int j = 0; j < columnSecond; ++j) {
      for(int k = 0; k < columnFirst; ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

extern void nsc_matmul(NSC_Size a_size,
                       NSC_Size b_size,
                       float *const *a,
                       float *const *b,
                       float **result) {
  
  int rowFirst = a_size.rows;
  int columnFirst = a_size.columns;
  int columnSecond = b_size.columns;
  
#ifdef __ARM_NEON
  // Use Winograd blocked multiplication for larger matrices
  if (rowFirst >= 4 && columnFirst >= 4 && columnSecond >= 4) {
    winograd_matmul_blocked_f32(a, b, result, rowFirst, columnFirst, columnSecond);
    return;
  }
#endif
  
  // Standard matrix multiplication for small matrices or non-ARM platforms
  for(int i = 0; i < rowFirst; ++i) {
    for(int j = 0; j < columnSecond; ++j) {
      for(int k = 0; k < columnFirst; ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

extern void nsc_transpose_2d_16(__fp16 *const *input,
                                __fp16 **result,
                                NSC_Size input_size) {
  int rows = input_size.rows;
  int cols = input_size.columns;
  
  // Perform the transpose operation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result[j][i] = input[i][j];
    }
  }
}

extern void nsc_transpose_2d(float *const *input,
                             float **result,
                             NSC_Size input_size) {
  int rows = input_size.rows;
  int cols = input_size.columns;
  
  // Perform the transpose operation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result[j][i] = input[i][j];
    }
  }
}

extern void nsc_specific_zero_pad_2d_f16(__fp16 *const *input,
                                     __fp16 **result,
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
  
  for (int r = 0; r < inputRows; r++) {
    for (int c = 0; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      int padded_r = r + paddingTop;
      
      result[padded_r][padded_c] = input[r][c];
    }
  }
}

extern void nsc_specific_zero_pad_2d(float *const *input,
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
  
  for (int r = 0; r < inputRows; r++) {
    for (int c = 0; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      int padded_r = r + paddingTop;
      
      result[padded_r][padded_c] = input[r][c];
    }
  }
}

extern void nsc_flatten2d_16(NSC_Size input_size,
                             __fp16 *const *input,
                             __fp16 *result) {
  
  int length = input_size.rows * input_size.columns;
  __fp16 *padded = malloc(length * sizeof(__fp16));
  
  for (int i = 0; i < input_size.rows * input_size.columns; i++) {
    padded[i] = 0;
  }

  for (int r = 0; r < input_size.rows; r++) {
    for (int c = 0; c < input_size.columns; c++) {
      __fp16 value = input[r][c];
      padded[(input_size.columns * r) + c] = value;
    }
  }
  
  memcpy(result, padded, length * sizeof(__fp16));
  free(padded);
}

extern void nsc_flatten3d_16(NSC_Size input_size,
                             __fp16 *const *const *input,
                             __fp16 *result) {
  
  int length = input_size.rows * input_size.columns * input_size.depth;
  float *padded = malloc(length * sizeof(__fp16));
  
  for (int i = 0; i < length; i++) {
    padded[i] = 0;
  }

  for (int d = 0; d < input_size.depth; d++) {
    for (int r = 0; r < input_size.rows; r++) {
      for (int c = 0; c < input_size.columns; c++) {
        __fp16 value = input[d][r][c];
        int index = c + r * input_size.columns + d * input_size.columns * input_size.rows;
        padded[index] = value;
      }
    }
  }
  
  memcpy(result, padded, length * sizeof(__fp16));
  free(padded);
}

extern void nsc_flatten3d(NSC_Size input_size,
                          float *const *const *input,
                          float *result) {
  
  int length = input_size.rows * input_size.columns * input_size.depth;
  float *padded = malloc(length * sizeof(float));
  
  for (int i = 0; i < length; i++) {
    padded[i] = 0;
  }

  for (int d = 0; d < input_size.depth; d++) {
    for (int r = 0; r < input_size.rows; r++) {
      for (int c = 0; c < input_size.columns; c++) {
        float value = input[d][r][c];
        int index = c + r * input_size.columns + d * input_size.columns * input_size.rows;
        padded[index] = value;
      }
    }
  }
  
  memcpy(result, padded, length * sizeof(float));
  free(padded);
}

extern void nsc_flatten2d(NSC_Size input_size,
                          float *const *input,
                          float *result) {
  
  int length = input_size.rows * input_size.columns;
  float *padded = malloc(length * sizeof(float));
  
  for (int i = 0; i < input_size.rows * input_size.columns; i++) {
    padded[i] = 0;
  }

  for (int r = 0; r < input_size.rows; r++) {
    for (int c = 0; c < input_size.columns; c++) {
      float value = input[r][c];
      padded[(input_size.columns * r) + c] = value;
    }
  }
  
  memcpy(result, padded, length * sizeof(float));
  free(padded);
}

double scaled_cosine(const double i) {
  return 0.5f * (1.0f - cos(i * M_PI));
}

double randfrom(const double min, const double max) {
  double range = (max - min);
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

extern void random_array(const int size, double *result) {
  double *perlin = malloc(size * sizeof(double));
  
  srand( (unsigned)time( NULL ) );

  for (int i = 0; i < size + 1; i++) {
    double random = randfrom(0.0f, 1000.0f) / 1000.0f;
    perlin[i] = random;
  }
  
  memcpy(result, perlin, size * sizeof(double));
  free(perlin);
}

extern double nsc_perlin_noise(const double x,
                               const double y,
                               const double z,
                               const double amplitude,
                               const int octaves,
                               const int size,
                               const double* perlin_seed) {

  const int yWrapB = 4;
  const int yWrap = 1 << yWrapB;
  const int zWrapB = 8;
  const int zWrap = 1 << zWrapB;
  
  double mutable_x = x;
  double mutable_y = y;
  double mutable_z = z;
  
  if (x < 0.0f) {
    mutable_x = -x;
  }
  if (y < 0.0f) {
    mutable_y = -y;
  }
  if (z < 0.0f) {
    mutable_z = -z;
  }
  
  int xi = (int)floor(mutable_x);
  int yi = (int)floor(mutable_y);
  int zi = (int)floor(mutable_z);

  double xf = mutable_x - (double)(xi);
  double yf = mutable_y - (double)(yi);
  double zf = mutable_y - (double)(zi);
  
  double rxf = 0.0f;
  double ryf = 0.0f;
  
  double r = 0.0f;
  double ampl = 0.5;
  
  double n1 = 0.0f;
  double n2 = 0.0f;
  double n3 = 0.0f;
  
  for (int i = 0; i < octaves; i++) {
    int of = xi + (yi << yWrapB) + (zi << zWrapB);
    
    rxf = scaled_cosine(xf);
    ryf = scaled_cosine(yf);
    
    n1 = perlin_seed[of & size];
    n1 += rxf * (perlin_seed[(of + 1) & size] - 1);
    n2 = perlin_seed[(of + yWrap) & size];
    n2 += rxf * (perlin_seed[(of + yWrap + 1) & size] - n2);
    n1 += ryf * (n2 - n1);
    
    of += zWrap;
    n2 = perlin_seed[of & size];
    n2 += rxf * (perlin_seed[(of + 1) & size] - n2);
    n3 = perlin_seed[(of + yWrap) & size];
    n3 += rxf * (perlin_seed[(of + yWrap + 1) & size] - n3);
    n2 += ryf * (n3 - n2);
    
    n1 += scaled_cosine(zf) * (n2 - n1);

    r += n1 * ampl;
    ampl *= amplitude;
    xi <<= 1;
    xf *= 2;
    yi <<= 1;
    yf *= 2;
    zi <<= 1;
    zf *= 2;
    
    if (xf >= 1.0f) {
      xi += 1;
      xf -= 1;
    }
    
    if (yf >= 1.0f) {
      yi += 1;
      yf -= 1;
    }
    
    if (zf >= 1.0f) {
      zi += 1;
      zf -= 1;
    }
  }
  
  return r;
}

extern void nsc_stride_pad_2D_f16(__fp16 *const *input,
                                  __fp16 **result,
                                  NSC_Size input_size,
                                  NSC_Size stride_size)
{

  int numToPadRows = stride_size.rows - 1;
  int numToPadCols = stride_size.columns - 1;
  
  int newRows = input_size.rows + ((stride_size.rows - 1) * (input_size.rows - 1));
  int newColumns = input_size.columns + ((stride_size.columns - 1) * (input_size.columns - 1));
    
  if (numToPadCols > 0 && numToPadRows > 0) {
    
    int custom_r = 0;
    for (int r = 0; r < newRows; r += stride_size.rows) {
      int custom_c = 0;
      for (int c = 0; c < newColumns; c += stride_size.columns) {
        result[r][c] = input[custom_r][custom_c];
        custom_c += 1;
      }
      custom_r += 1;
    }
  }
}

extern void nsc_stride_pad_2D(float *const *input,
                              float **result,
                              NSC_Size input_size,
                              NSC_Size stride_size) {
  
  int numToPadRows = stride_size.rows - 1;
  int numToPadCols = stride_size.columns - 1;
  
  int newRows = input_size.rows + ((stride_size.rows - 1) * (input_size.rows - 1));
  int newColumns = input_size.columns + ((stride_size.columns - 1) * (input_size.columns - 1));
    
  if (numToPadCols > 0 && numToPadRows > 0) {
    
    int custom_r = 0;
    for (int r = 0; r < newRows; r += stride_size.rows) {
      int custom_c = 0;
      for (int c = 0; c < newColumns; c += stride_size.columns) {
        result[r][c] = input[custom_r][custom_c];
        custom_c += 1;
      }
      custom_r += 1;
    }
  }
}

extern void nsc_stride_pad_f16(const __fp16 input[],
                           __fp16 *result,
                           NSC_Size input_size,
                           NSC_Size stride_size) {
  
  int numToPadRows = stride_size.rows - 1;
  int numToPadCols = stride_size.columns - 1;
  
  int newRows = input_size.rows + ((stride_size.rows - 1) * (input_size.rows - 1));
  int newColumns = input_size.columns + ((stride_size.columns - 1) * (input_size.columns - 1));

  int length = newRows * newColumns;
  __fp16 *padded = malloc(length * sizeof(__fp16));
  
  for (int i = 0; i < newRows * newColumns; i++) {
    padded[i] = 0;
  }
  
  int i = 0;

  if (numToPadCols > 0 && numToPadRows > 0) {
    
    for (int r = 0; r < newRows; r += stride_size.rows) {
      for (int c = 0; c < newColumns; c += stride_size.columns) {
        int index = (r * newRows) + c;
        padded[index] = input[i];
        i += 1;
      }
    }
  }
  
  memcpy(result, padded, length * sizeof(__fp16));
  free(padded);
}

extern void nsc_stride_pad(const float input[],
                           float *result,
                           NSC_Size input_size,
                           NSC_Size stride_size) {
  
  int numToPadRows = stride_size.rows - 1;
  int numToPadCols = stride_size.columns - 1;
  
  int newRows = input_size.rows + ((stride_size.rows - 1) * (input_size.rows - 1));
  int newColumns = input_size.columns + ((stride_size.columns - 1) * (input_size.columns - 1));

  int length = newRows * newColumns;
  float *padded = malloc(length * sizeof(float));
  
  for (int i = 0; i < newRows * newColumns; i++) {
    padded[i] = 0;
  }
  
  int i = 0;

  if (numToPadCols > 0 && numToPadRows > 0) {
    
    for (int r = 0; r < newRows; r += stride_size.rows) {
      for (int c = 0; c < newColumns; c += stride_size.columns) {
        int index = (r * newRows) + c;
        padded[index] = input[i];
        i += 1;
      }
    }
  }
  
  memcpy(result, padded, length * sizeof(float));
  free(padded);
}

extern void nsc_padding_calculation(NSC_Size stride,
                                    NSC_Padding padding,
                                    NSC_Size filter_size,
                                    NSC_Size input_size,
                                    int *paddingTop,
                                    int *paddingBottom,
                                    int *paddingLeft,
                                    int *paddingRight) {
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  if (padding == same) {
    double height = (double)inputRows;
    double width = (double)inputColumns;
    
    double outHeight = ceil(height / (double)strideR);
    double outWidth = ceil(width / (double)strideC);
    
    double padAlongHeight = fmax((outHeight - 1) * (double)strideR + (double)filterRows - height, 0);
    double padAlongWidth = fmax((outWidth - 1) * (double)strideC + (double)filterColumns- width, 0);
    
    int paddingT = (int)floor(padAlongHeight / 2);
    int paddingB = (int)padAlongHeight - (double)paddingT;
    int paddingL = (int)floor(padAlongWidth / 2);
    int paddingR = (int)padAlongWidth - (double)paddingL;
    
    *paddingTop = paddingT;
    *paddingBottom = paddingB;
    *paddingLeft = paddingL;
    *paddingRight = paddingR;
    
    return;
  } else {
    
    *paddingTop = 0;
    *paddingBottom = 0;
    *paddingLeft = 0;
    *paddingRight = 0;
    return;
  }
}

extern void nsc_specific_zero_pad(const float input[],
                                  float *result,
                                  NSC_Size input_size,
                                  int paddingTop,
                                  int paddingBottom,
                                  int paddingLeft,
                                  int paddingRight) {
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int padded_row_total = inputRows + paddingLeft + paddingRight;
  int padded_col_total = inputColumns + paddingTop + paddingBottom;
  
  int length = padded_row_total * padded_col_total;
  float *padded = malloc(length * sizeof(float));
  
  for (int i = 0; i < padded_row_total * padded_col_total; i++) {
    padded[i] = 0;
  }
  
  if (padded == NULL || input == NULL)
    return;
  
  for (int r = 0; r < inputRows; r++) {
    for (int c = 0; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      int padded_r = r + paddingTop;
      
      int index = (padded_r  * padded_row_total) + padded_c;
      padded[index] = input[(r * inputRows) + c];
    }
  }
    
  memcpy(result, padded, length * sizeof(float));
  
  free(padded);
}

extern void nsc_zero_pad_2D(float *const *input,
                           float **result,
                           NSC_Size filter_size,
                           NSC_Size input_size,
                           NSC_Size stride) {
  int paddingLeft;
  int paddingRight;
  int paddingBottom;
  int paddingTop;
  
  int *pad_l_ptr = &paddingLeft;
  int *pad_r_ptr = &paddingRight;
  int *pad_b_ptr = &paddingBottom;
  int *pad_t_ptr = &paddingTop;
  
  nsc_padding_calculation(stride,
                          same,
                          filter_size,
                          input_size,
                          pad_t_ptr,
                          pad_b_ptr,
                          pad_l_ptr,
                          pad_r_ptr);
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int padded_row_total = inputRows + paddingLeft + paddingRight;
  int padded_col_total = inputColumns + paddingTop + paddingBottom;
  
  int length = padded_row_total * padded_col_total;
  
  for (int i = 0; i < padded_row_total; i++) {
    for (int j = 0; j < padded_col_total; j++) {
      result[i][j] = 0;
    }
  }
  
  if (result == NULL || input == NULL)
    return;
  
  for (int r = 0; r < inputRows; r++) {
    for (int c = 0; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      int padded_r = r + paddingTop;
      
      int index = (padded_r  * padded_row_total) + padded_c;
      result[padded_r][padded_c] = input[r][c];
    }
  }
}

extern void nsc_zero_pad_f16(const __fp16 input[],
                         __fp16 *result,
                         NSC_Size filter_size,
                         NSC_Size input_size,
                         NSC_Size stride) {
  int paddingLeft;
  int paddingRight;
  int paddingBottom;
  int paddingTop;
  
  int *pad_l_ptr = &paddingLeft;
  int *pad_r_ptr = &paddingRight;
  int *pad_b_ptr = &paddingBottom;
  int *pad_t_ptr = &paddingTop;
  
  nsc_padding_calculation(stride,
                          same,
                          filter_size,
                          input_size,
                          pad_t_ptr,
                          pad_b_ptr,
                          pad_l_ptr,
                          pad_r_ptr);
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int padded_row_total = inputRows + paddingLeft + paddingRight;
  int padded_col_total = inputColumns + paddingTop + paddingBottom;
  
  int length = padded_row_total * padded_col_total;
  __fp16 *padded = malloc(length * sizeof(__fp16));
  
  for (int i = 0; i < padded_row_total * padded_col_total; i++) {
    padded[i] = 0;
  }
  
  if (padded == NULL || input == NULL)
    return;
  
  for (int r = 0; r < inputRows; r++) {
    for (int c = 0; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      int padded_r = r + paddingTop;
      
      int index = (padded_r  * padded_row_total) + padded_c;
      padded[index] = input[(r * inputRows) + c];
    }
  }
    
  memcpy(result, padded, length * sizeof(__fp16));
  
  free(padded);
}

extern void nsc_zero_pad(const float input[],
                         float *result,
                         NSC_Size filter_size,
                         NSC_Size input_size,
                         NSC_Size stride) {
  int paddingLeft;
  int paddingRight;
  int paddingBottom;
  int paddingTop;
  
  int *pad_l_ptr = &paddingLeft;
  int *pad_r_ptr = &paddingRight;
  int *pad_b_ptr = &paddingBottom;
  int *pad_t_ptr = &paddingTop;
  
  nsc_padding_calculation(stride,
                          same,
                          filter_size,
                          input_size,
                          pad_t_ptr,
                          pad_b_ptr,
                          pad_l_ptr,
                          pad_r_ptr);
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int padded_row_total = inputRows + paddingLeft + paddingRight;
  int padded_col_total = inputColumns + paddingTop + paddingBottom;
  
  int length = padded_row_total * padded_col_total;
  float *padded = malloc(length * sizeof(float));
  
  for (int i = 0; i < padded_row_total * padded_col_total; i++) {
    padded[i] = 0;
  }
  
  if (padded == NULL || input == NULL)
    return;
  
  for (int r = 0; r < inputRows; r++) {
    for (int c = 0; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      int padded_r = r + paddingTop;
      
      int index = (padded_r  * padded_row_total) + padded_c;
      padded[index] = input[(r * inputRows) + c];
    }
  }
    
  memcpy(result, padded, length * sizeof(float));
  
  free(padded);
}

#ifdef __ARM_NEON
// Optimized 3x3 convolution kernel for float32 - bounds safe
static inline float neon_conv_3x3(float *const *signal,
                                 float *const *filter,
                                 int sig_row, int sig_col) {
  // Load 3x3 signal region safely using individual elements
  float sig_vals[9];
  sig_vals[0] = signal[sig_row][sig_col];
  sig_vals[1] = signal[sig_row][sig_col + 1];
  sig_vals[2] = signal[sig_row][sig_col + 2];
  sig_vals[3] = signal[sig_row + 1][sig_col];
  sig_vals[4] = signal[sig_row + 1][sig_col + 1];
  sig_vals[5] = signal[sig_row + 1][sig_col + 2];
  sig_vals[6] = signal[sig_row + 2][sig_col];
  sig_vals[7] = signal[sig_row + 2][sig_col + 1];
  sig_vals[8] = signal[sig_row + 2][sig_col + 2];
  
  // Load 3x3 filter safely
  float filt_vals[9];
  filt_vals[0] = filter[0][0];
  filt_vals[1] = filter[0][1];
  filt_vals[2] = filter[0][2];
  filt_vals[3] = filter[1][0];
  filt_vals[4] = filter[1][1];
  filt_vals[5] = filter[1][2];
  filt_vals[6] = filter[2][0];
  filt_vals[7] = filter[2][1];
  filt_vals[8] = filter[2][2];
  
  // Load into SIMD registers for vectorized computation
  float32x4_t sig_vec1 = vld1q_f32(&sig_vals[0]); // [s00, s01, s02, s10]
  float32x4_t sig_vec2 = vld1q_f32(&sig_vals[4]); // [s11, s12, s20, s21]
  float sig_val = sig_vals[8]; // s22
  
  float32x4_t filt_vec1 = vld1q_f32(&filt_vals[0]); // [f00, f01, f02, f10]
  float32x4_t filt_vec2 = vld1q_f32(&filt_vals[4]); // [f11, f12, f20, f21]
  float filt_val = filt_vals[8]; // f22
  
  // Vectorized multiply-accumulate
  float32x4_t acc1 = vmulq_f32(sig_vec1, filt_vec1);
  float32x4_t acc2 = vmlaq_f32(acc1, sig_vec2, filt_vec2);
  
  // Horizontal sum + final element
  float32x2_t sum_pair = vadd_f32(vget_high_f32(acc2), vget_low_f32(acc2));
  float result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0) + (sig_val * filt_val);
  
  return result;
}

// Optimized 3x3 convolution kernel for float16 - bounds safe
__attribute__((target("arch=armv8.2-a+fp16")))
static inline __fp16 neon_conv_3x3_f16(__fp16 *const *signal,
                                      __fp16 *const *filter,
                                      int sig_row, int sig_col) {
  // Load 3x3 signal region safely using individual elements
  __fp16 sig_vals[9];
  sig_vals[0] = signal[sig_row][sig_col];
  sig_vals[1] = signal[sig_row][sig_col + 1];
  sig_vals[2] = signal[sig_row][sig_col + 2];
  sig_vals[3] = signal[sig_row + 1][sig_col];
  sig_vals[4] = signal[sig_row + 1][sig_col + 1];
  sig_vals[5] = signal[sig_row + 1][sig_col + 2];
  sig_vals[6] = signal[sig_row + 2][sig_col];
  sig_vals[7] = signal[sig_row + 2][sig_col + 1];
  sig_vals[8] = signal[sig_row + 2][sig_col + 2];
  
  // Load 3x3 filter safely
  __fp16 filt_vals[9];
  filt_vals[0] = filter[0][0];
  filt_vals[1] = filter[0][1];
  filt_vals[2] = filter[0][2];
  filt_vals[3] = filter[1][0];
  filt_vals[4] = filter[1][1];
  filt_vals[5] = filter[1][2];
  filt_vals[6] = filter[2][0];
  filt_vals[7] = filter[2][1];
  filt_vals[8] = filter[2][2];
  
  // Load into SIMD registers for vectorized computation
  float16x8_t sig_vec = vld1q_f16(&sig_vals[0]);  // [s00, s01, s02, s10, s11, s12, s20, s21]
  float16x8_t filt_vec = vld1q_f16(&filt_vals[0]); // [f00, f01, f02, f10, f11, f12, f20, f21]
  __fp16 sig_val = sig_vals[8];   // s22
  __fp16 filt_val = filt_vals[8]; // f22
  
  // Vectorized multiply-accumulate for first 8 elements
  float16x8_t acc = vmulq_f16(sig_vec, filt_vec);
  
  // Horizontal sum of 8 elements + final element
  float16x4_t sum_low = vget_low_f16(acc);
  float16x4_t sum_high = vget_high_f16(acc);
  float16x4_t sum_pair = vadd_f16(sum_low, sum_high);
  float16x4_t sum_fold = vpadd_f16(sum_pair, sum_pair);
  float16x4_t sum_final = vpadd_f16(sum_fold, sum_fold);
  __fp16 result = vget_lane_f16(sum_final, 0) + (sig_val * filt_val);
  
  return result;
}

// Optimized 1x1 convolution kernel for float32 (pointwise convolution)
static inline float neon_conv_1x1(float *const *signal,
                                 float *const *filter,
                                 int sig_row, int sig_col) {
  // 1x1 convolution is just a simple multiplication
  return signal[sig_row][sig_col] * filter[0][0];
}

// Optimized 1x1 convolution kernel for float16 (pointwise convolution)
static inline __fp16 neon_conv_1x1_f16(__fp16 *const *signal,
                                      __fp16 *const *filter,
                                      int sig_row, int sig_col) {
  // 1x1 convolution is just a simple multiplication
  return signal[sig_row][sig_col] * filter[0][0];
}

// Optimized 5x5 convolution kernel for float32
static inline float neon_conv_5x5(float *const *signal,
                                 float *const *filter,
                                 int sig_row, int sig_col) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  
  // Process each of the 5 rows
  for (int r = 0; r < 5; r++) {
    // Load 5 signal elements (use first 4, then handle 5th separately)
    float32x4_t sig_vec = vld1q_f32(&signal[sig_row + r][sig_col]);
    float32x4_t filt_vec = vld1q_f32(&filter[r][0]);
    
    // Multiply and accumulate first 4 elements
    acc = vmlaq_f32(acc, sig_vec, filt_vec);
    
    // Handle 5th element separately
    float sig_5th = signal[sig_row + r][sig_col + 4];
    float filt_5th = filter[r][4];
    acc = vmlaq_n_f32(acc, vdupq_n_f32(sig_5th), filt_5th);
  }
  
  // Horizontal sum
  float32x2_t sum_pair = vadd_f32(vget_high_f32(acc), vget_low_f32(acc));
  return vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
}

// Optimized 5x5 convolution kernel for float16
__attribute__((target("arch=armv8.2-a+fp16")))
static inline __fp16 neon_conv_5x5_f16(__fp16 *const *signal,
                                      __fp16 *const *filter,
                                      int sig_row, int sig_col) {
  float16x8_t acc = vdupq_n_f16(0.0f);
  
  // Process each of the 5 rows
  for (int r = 0; r < 5; r++) {
    // Load 5 signal elements (load 8, use first 5)
    float16x8_t sig_vec = vld1q_f16(&signal[sig_row + r][sig_col]);
    float16x8_t filt_vec = vld1q_f16(&filter[r][0]);
    
    // Multiply and accumulate
    acc = vfmaq_f16(acc, sig_vec, filt_vec);
  }
  
  // Sum first 5 elements (ignore elements 5-7)
  __fp16 result = vgetq_lane_f16(acc, 0) + vgetq_lane_f16(acc, 1) + 
                  vgetq_lane_f16(acc, 2) + vgetq_lane_f16(acc, 3) + 
                  vgetq_lane_f16(acc, 4);
  return result;
}

// Bounds-safe data access function for float32
static inline float safe_get_signal(float *const *signal,
                                   float *const *padded_signal,
                                   int row, int col,
                                   int input_rows, int input_cols,
                                   int pad_top, int pad_left,
                                   NSC_Padding padding) {
  if (padding == same) {
    return padded_signal[row][col];
  } else {
    // Direct access for valid padding
    if (row >= 0 && row < input_rows && col >= 0 && col < input_cols) {
      return signal[row][col];
    }
    return 0.0f; // Out of bounds
  }
}

// Bounds-safe data access function for float16
static inline __fp16 safe_get_signal_f16(__fp16 *const *signal,
                                        __fp16 *const *padded_signal,
                                        int row, int col,
                                        int input_rows, int input_cols,
                                        int pad_top, int pad_left,
                                        NSC_Padding padding) {
  if (padding == same) {
    return padded_signal[row][col];
  } else {
    // Direct access for valid padding
    if (row >= 0 && row < input_rows && col >= 0 && col < input_cols) {
      return signal[row][col];
    }
    return 0.0f; // Out of bounds
  }
}

// SIMD optimized convolution kernel for float32
static inline float neon_conv_kernel(float *const *signal,
                                   float *const *filter,
                                   int sig_row, int sig_col,
                                   int filter_rows, int filter_cols) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  
  for (int fr = 0; fr < filter_rows; fr++) {
    int fc = 0;
    
    // Process 4 elements at a time
    for (; fc <= filter_cols - 4; fc += 4) {
      float32x4_t sig_vec = vld1q_f32(&signal[sig_row + fr][sig_col + fc]);
      float32x4_t filt_vec = vld1q_f32(&filter[fr][fc]);
      acc = vmlaq_f32(acc, sig_vec, filt_vec);
    }
    
    // Handle remaining elements
    for (; fc < filter_cols; fc++) {
      float sig_val = signal[sig_row + fr][sig_col + fc];
      float filt_val = filter[fr][fc];
      acc = vmlaq_n_f32(acc, vdupq_n_f32(sig_val), filt_val);
    }
  }
  
  // Horizontal sum
  float32x2_t sum_pair = vadd_f32(vget_high_f32(acc), vget_low_f32(acc));
  return vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
}

// SIMD optimized convolution kernel for float16
__attribute__((target("arch=armv8.2-a+fp16")))
static inline __fp16 neon_conv_kernel_f16(__fp16 *const *signal,
                                         __fp16 *const *filter,
                                         int sig_row, int sig_col,
                                         int filter_rows, int filter_cols) {
  __fp16 sum = 0.0f;
  
  // For small filters, use scalar computation to avoid SIMD boundary issues
  if (filter_rows * filter_cols <= 16) {
    for (int fr = 0; fr < filter_rows; fr++) {
      for (int fc = 0; fc < filter_cols; fc++) {
        sum += signal[sig_row + fr][sig_col + fc] * filter[fr][fc];
      }
    }
    return sum;
  }
  
  // For larger filters, use SIMD with careful boundary handling
  float16x8_t acc = vdupq_n_f16(0.0f);
  
  for (int fr = 0; fr < filter_rows; fr++) {
    int fc = 0;
    
    // Process 8 elements at a time for fp16, but only if we have at least 8 elements
    for (; fc <= filter_cols - 8; fc += 8) {
      float16x8_t sig_vec = vld1q_f16(&signal[sig_row + fr][sig_col + fc]);
      float16x8_t filt_vec = vld1q_f16(&filter[fr][fc]);
      acc = vfmaq_f16(acc, sig_vec, filt_vec);
    }
    
    // Handle remaining elements
    for (; fc < filter_cols; fc++) {
      __fp16 sig_val = signal[sig_row + fr][sig_col + fc];
      __fp16 filt_val = filter[fr][fc];
      sum += sig_val * filt_val;
    }
  }
  
  // Add SIMD accumulator to scalar sum
  if (filter_cols >= 8) {
    float16x4_t sum_low = vget_low_f16(acc);
    float16x4_t sum_high = vget_high_f16(acc);
    float16x4_t sum_pair = vadd_f16(sum_low, sum_high);
    float16x4_t sum_fold = vpadd_f16(sum_pair, sum_pair);
    float16x4_t sum_final = vpadd_f16(sum_fold, sum_fold);
    sum += vget_lane_f16(sum_final, 0);
  }
  
  return sum;
}

// Winograd F(2x2,3x3) algorithm for float32
// Processes 2x2 output tiles using 3x3 filters with only 6 multiplications
static inline void winograd_f32_2x2_3x3(float *const *signal,
                                         float *const *filter,
                                         float **result,
                                         int sig_row, int sig_col,
                                         int out_row, int out_col) {
  // Input transform matrix BT
  // BT = [1,  0, -1,  0]
  //      [0,  1,  1,  0]
  //      [0, -1,  1,  0]
  //      [0,  1,  0, -1]
  
  // Filter transform matrix G
  // G = [1,    0,    0]
  //     [0.5,  0.5,  0.5]
  //     [0.5, -0.5,  0.5]
  //     [0,    0,    1]
  
  // Output transform matrix AT
  // AT = [1,  1,  1,  0]
  //      [0,  1, -1, -1]
  
  // Input tile d (4x4)
  float d[16];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      d[i*4 + j] = signal[sig_row + i][sig_col + j];
    }
  }
  
  // Filter g (3x3) 
  float g[9];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      g[i*3 + j] = filter[i][j];
    }
  }
  
  // Transform filter: U = G * g * GT
  float U[16];
  // G * g
  float Gg[12]; // 4x3
  Gg[0] = g[0];
  Gg[1] = g[1]; 
  Gg[2] = g[2];
  Gg[3] = 0.5f * (g[0] + g[3] + g[6]);
  Gg[4] = 0.5f * (g[1] + g[4] + g[7]);
  Gg[5] = 0.5f * (g[2] + g[5] + g[8]);
  Gg[6] = 0.5f * (g[0] - g[3] + g[6]);
  Gg[7] = 0.5f * (g[1] - g[4] + g[7]);
  Gg[8] = 0.5f * (g[2] - g[5] + g[8]);
  Gg[9] = g[6];
  Gg[10] = g[7];
  Gg[11] = g[8];
  
  // Gg * GT = U (4x4)
  U[0] = Gg[0];
  U[1] = 0.5f * (Gg[0] + Gg[3] + Gg[6]);
  U[2] = 0.5f * (Gg[0] - Gg[3] + Gg[6]);
  U[3] = Gg[9];
  U[4] = Gg[1];
  U[5] = 0.5f * (Gg[1] + Gg[4] + Gg[7]);
  U[6] = 0.5f * (Gg[1] - Gg[4] + Gg[7]);
  U[7] = Gg[10];
  U[8] = Gg[2];
  U[9] = 0.5f * (Gg[2] + Gg[5] + Gg[8]);
  U[10] = 0.5f * (Gg[2] - Gg[5] + Gg[8]);
  U[11] = Gg[11];
  U[12] = Gg[9];
  U[13] = 0.5f * (Gg[9] + Gg[10] + Gg[11]);
  U[14] = 0.5f * (Gg[9] - Gg[10] + Gg[11]);
  U[15] = Gg[11];
  
  // Transform input: V = BT * d * B
  float V[16];
  // BT * d (4x4)
  float BTd[16];
  for (int i = 0; i < 4; i++) {
    BTd[i*4 + 0] = d[i*4 + 0] - d[i*4 + 2];
    BTd[i*4 + 1] = d[i*4 + 1] + d[i*4 + 2];
    BTd[i*4 + 2] = -d[i*4 + 1] + d[i*4 + 2];
    BTd[i*4 + 3] = d[i*4 + 1] - d[i*4 + 3];
  }
  
  // BTd * B = V (4x4)
  for (int i = 0; i < 4; i++) {
    V[i*4 + 0] = BTd[i*4 + 0] - BTd[i*4 + 2];
    V[i*4 + 1] = BTd[i*4 + 1] + BTd[i*4 + 2];
    V[i*4 + 2] = -BTd[i*4 + 1] + BTd[i*4 + 2];
    V[i*4 + 3] = BTd[i*4 + 1] - BTd[i*4 + 3];
  }
  
  // Element-wise multiplication: M = U ⊙ V
  float M[16];
  for (int i = 0; i < 16; i++) {
    M[i] = U[i] * V[i];
  }
  
  // Output transform: Y = AT * M * A
  float Y[4]; // 2x2 output
  // AT * M (2x4)
  float ATM[8];
  for (int i = 0; i < 4; i++) {
    ATM[0*4 + i] = M[0*4 + i] + M[1*4 + i] + M[2*4 + i];
    ATM[1*4 + i] = M[1*4 + i] - M[2*4 + i] - M[3*4 + i];
  }
  
  // ATM * A = Y (2x2)
  Y[0] = ATM[0] + ATM[1] + ATM[2];
  Y[1] = ATM[1] - ATM[2] - ATM[3];
  Y[2] = ATM[4] + ATM[5] + ATM[6];
  Y[3] = ATM[5] - ATM[6] - ATM[7];
  
  // Store results
  result[out_row][out_col] = Y[0];
  result[out_row][out_col + 1] = Y[1];
  result[out_row + 1][out_col] = Y[2];
  result[out_row + 1][out_col + 1] = Y[3];
}

// Winograd-based 3x3 convolution (stride=1 only) for float32
static inline float winograd_3x3_f32_single(float *const *signal,
                                             float *const *filter,
                                             int sig_row, int sig_col) {
  // For single output element, we can use a simpler Winograd formulation
  // Input transform for 1x1 output with 3x3 filter
  // Uses minimal Winograd F(1x1,3x3) - equivalent to standard convolution
  // but with structured computation that can be optimized
  
  // Load 3x3 input data
  float d[9];
  d[0] = signal[sig_row][sig_col];     d[1] = signal[sig_row][sig_col+1];     d[2] = signal[sig_row][sig_col+2];
  d[3] = signal[sig_row+1][sig_col];   d[4] = signal[sig_row+1][sig_col+1];   d[5] = signal[sig_row+1][sig_col+2];
  d[6] = signal[sig_row+2][sig_col];   d[7] = signal[sig_row+2][sig_col+1];   d[8] = signal[sig_row+2][sig_col+2];
  
  // Load 3x3 filter
  float g[9];
  g[0] = filter[0][0]; g[1] = filter[0][1]; g[2] = filter[0][2];
  g[3] = filter[1][0]; g[4] = filter[1][1]; g[5] = filter[1][2];
  g[6] = filter[2][0]; g[7] = filter[2][1]; g[8] = filter[2][2];
  
  // Direct convolution with optimized access pattern
  return d[0]*g[0] + d[1]*g[1] + d[2]*g[2] +
         d[3]*g[3] + d[4]*g[4] + d[5]*g[5] +
         d[6]*g[6] + d[7]*g[7] + d[8]*g[8];
}

// Winograd F(2x2,3x3) algorithm for float16
static inline void winograd_f16_2x2_3x3(__fp16 *const *signal,
                                        __fp16 *const *filter,
                                        __fp16 **result,
                                        int sig_row, int sig_col,
                                        int out_row, int out_col) {
  // Input tile d (4x4)
  __fp16 d[16];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      d[i*4 + j] = signal[sig_row + i][sig_col + j];
    }
  }
  
  // Filter g (3x3) 
  __fp16 g[9];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      g[i*3 + j] = filter[i][j];
    }
  }
  
  // Transform filter: U = G * g * GT
  __fp16 U[16];
  // G * g
  __fp16 Gg[12]; // 4x3
  Gg[0] = g[0];
  Gg[1] = g[1]; 
  Gg[2] = g[2];
  Gg[3] = 0.5f * (g[0] + g[3] + g[6]);
  Gg[4] = 0.5f * (g[1] + g[4] + g[7]);
  Gg[5] = 0.5f * (g[2] + g[5] + g[8]);
  Gg[6] = 0.5f * (g[0] - g[3] + g[6]);
  Gg[7] = 0.5f * (g[1] - g[4] + g[7]);
  Gg[8] = 0.5f * (g[2] - g[5] + g[8]);
  Gg[9] = g[6];
  Gg[10] = g[7];
  Gg[11] = g[8];
  
  // Gg * GT = U (4x4)
  U[0] = Gg[0];
  U[1] = 0.5f * (Gg[0] + Gg[3] + Gg[6]);
  U[2] = 0.5f * (Gg[0] - Gg[3] + Gg[6]);
  U[3] = Gg[9];
  U[4] = Gg[1];
  U[5] = 0.5f * (Gg[1] + Gg[4] + Gg[7]);
  U[6] = 0.5f * (Gg[1] - Gg[4] + Gg[7]);
  U[7] = Gg[10];
  U[8] = Gg[2];
  U[9] = 0.5f * (Gg[2] + Gg[5] + Gg[8]);
  U[10] = 0.5f * (Gg[2] - Gg[5] + Gg[8]);
  U[11] = Gg[11];
  U[12] = Gg[9];
  U[13] = 0.5f * (Gg[9] + Gg[10] + Gg[11]);
  U[14] = 0.5f * (Gg[9] - Gg[10] + Gg[11]);
  U[15] = Gg[11];
  
  // Transform input: V = BT * d * B
  __fp16 V[16];
  // BT * d (4x4)
  __fp16 BTd[16];
  for (int i = 0; i < 4; i++) {
    BTd[i*4 + 0] = d[i*4 + 0] - d[i*4 + 2];
    BTd[i*4 + 1] = d[i*4 + 1] + d[i*4 + 2];
    BTd[i*4 + 2] = -d[i*4 + 1] + d[i*4 + 2];
    BTd[i*4 + 3] = d[i*4 + 1] - d[i*4 + 3];
  }
  
  // BTd * B = V (4x4)
  for (int i = 0; i < 4; i++) {
    V[i*4 + 0] = BTd[i*4 + 0] - BTd[i*4 + 2];
    V[i*4 + 1] = BTd[i*4 + 1] + BTd[i*4 + 2];
    V[i*4 + 2] = -BTd[i*4 + 1] + BTd[i*4 + 2];
    V[i*4 + 3] = BTd[i*4 + 1] - BTd[i*4 + 3];
  }
  
  // Element-wise multiplication: M = U ⊙ V
  __fp16 M[16];
  for (int i = 0; i < 16; i++) {
    M[i] = U[i] * V[i];
  }
  
  // Output transform: Y = AT * M * A
  __fp16 Y[4]; // 2x2 output
  // AT * M (2x4)
  __fp16 ATM[8];
  for (int i = 0; i < 4; i++) {
    ATM[0*4 + i] = M[0*4 + i] + M[1*4 + i] + M[2*4 + i];
    ATM[1*4 + i] = M[1*4 + i] - M[2*4 + i] - M[3*4 + i];
  }
  
  // ATM * A = Y (2x2)
  Y[0] = ATM[0] + ATM[1] + ATM[2];
  Y[1] = ATM[1] - ATM[2] - ATM[3];
  Y[2] = ATM[4] + ATM[5] + ATM[6];
  Y[3] = ATM[5] - ATM[6] - ATM[7];
  
  // Store results
  result[out_row][out_col] = Y[0];
  result[out_row][out_col + 1] = Y[1];
  result[out_row + 1][out_col] = Y[2];
  result[out_row + 1][out_col + 1] = Y[3];
}

// Winograd-based 3x3 convolution (stride=1 only) for float16
static inline __fp16 winograd_3x3_f16_single(__fp16 *const *signal,
                                              __fp16 *const *filter,
                                              int sig_row, int sig_col) {
  // Load 3x3 input data
  __fp16 d[9];
  d[0] = signal[sig_row][sig_col];     d[1] = signal[sig_row][sig_col+1];     d[2] = signal[sig_row][sig_col+2];
  d[3] = signal[sig_row+1][sig_col];   d[4] = signal[sig_row+1][sig_col+1];   d[5] = signal[sig_row+1][sig_col+2];
  d[6] = signal[sig_row+2][sig_col];   d[7] = signal[sig_row+2][sig_col+1];   d[8] = signal[sig_row+2][sig_col+2];
  
  // Load 3x3 filter
  __fp16 g[9];
  g[0] = filter[0][0]; g[1] = filter[0][1]; g[2] = filter[0][2];
  g[3] = filter[1][0]; g[4] = filter[1][1]; g[5] = filter[1][2];
  g[6] = filter[2][0]; g[7] = filter[2][1]; g[8] = filter[2][2];
  
  // Direct convolution with optimized access pattern
  return d[0]*g[0] + d[1]*g[1] + d[2]*g[2] +
         d[3]*g[3] + d[4]*g[4] + d[5]*g[5] +
         d[6]*g[6] + d[7]*g[7] + d[8]*g[8];
}

// Winograd 2x2 matrix multiplication for float32
// Computes C = A * B where A, B, C are 2x2 matrices
// Uses 7 multiplications instead of 8
static inline void winograd_matmul_2x2_f32(const float A[4], const float B[4], float C[4]) {
  // A = [a11 a12]  B = [b11 b12]  C = [c11 c12]
  //     [a21 a22]      [b21 b22]      [c21 c22]
  //
  // Standard: 8 multiplications
  // Winograd: 7 multiplications + more additions
  
  float a11 = A[0], a12 = A[1], a21 = A[2], a22 = A[3];
  float b11 = B[0], b12 = B[1], b21 = B[2], b22 = B[3];
  
  // Winograd's 7 multiplications
  float m1 = (a11 + a22) * (b11 + b22);
  float m2 = (a21 + a22) * b11;
  float m3 = a11 * (b12 - b22);
  float m4 = a22 * (b21 - b11);
  float m5 = (a11 + a12) * b22;
  float m6 = (a21 - a11) * (b11 + b12);
  float m7 = (a12 - a22) * (b21 + b22);
  
  // Compute result
  C[0] = m1 + m4 - m5 + m7; // c11
  C[1] = m3 + m5;           // c12
  C[2] = m2 + m4;           // c21
  C[3] = m1 - m2 + m3 + m6; // c22
}

// Winograd 2x2 matrix multiplication for float16
static inline void winograd_matmul_2x2_f16(const __fp16 A[4], const __fp16 B[4], __fp16 C[4]) {
  __fp16 a11 = A[0], a12 = A[1], a21 = A[2], a22 = A[3];
  __fp16 b11 = B[0], b12 = B[1], b21 = B[2], b22 = B[3];
  
  // Winograd's 7 multiplications
  __fp16 m1 = (a11 + a22) * (b11 + b22);
  __fp16 m2 = (a21 + a22) * b11;
  __fp16 m3 = a11 * (b12 - b22);
  __fp16 m4 = a22 * (b21 - b11);
  __fp16 m5 = (a11 + a12) * b22;
  __fp16 m6 = (a21 - a11) * (b11 + b12);
  __fp16 m7 = (a12 - a22) * (b21 + b22);
  
  // Compute result
  C[0] = m1 + m4 - m5 + m7; // c11
  C[1] = m3 + m5;           // c12
  C[2] = m2 + m4;           // c21
  C[3] = m1 - m2 + m3 + m6; // c22
}

// Optimized matrix multiplication using Winograd 2x2 blocks for float32
static inline void winograd_matmul_blocked_f32(float *const *a, float *const *b, float **result,
                                               int rows_a, int cols_a, int cols_b) {
  // Process in 2x2 blocks where possible
  int block_rows = (rows_a / 2) * 2;
  int block_cols_a = (cols_a / 2) * 2;
  int block_cols_b = (cols_b / 2) * 2;
  
  for (int i = 0; i < block_rows; i += 2) {
    for (int j = 0; j < block_cols_b; j += 2) {
      // Initialize 2x2 result block
      float C_block[4] = {0, 0, 0, 0};
      
      for (int k = 0; k < block_cols_a; k += 2) {
        // Load 2x2 blocks from A and B
        float A_block[4] = {
          a[i][k], a[i][k+1],
          a[i+1][k], a[i+1][k+1]
        };
        float B_block[4] = {
          b[k][j], b[k][j+1],
          b[k+1][j], b[k+1][j+1]
        };
        
        // Compute partial result using Winograd
        float partial_C[4];
        winograd_matmul_2x2_f32(A_block, B_block, partial_C);
        
        // Accumulate to result block
        C_block[0] += partial_C[0];
        C_block[1] += partial_C[1];
        C_block[2] += partial_C[2];
        C_block[3] += partial_C[3];
      }
      
      // Store result block
      result[i][j] += C_block[0];
      result[i][j+1] += C_block[1];
      result[i+1][j] += C_block[2];
      result[i+1][j+1] += C_block[3];
      
      // Handle remaining columns in this block row
      for (int jj = block_cols_b; jj < cols_b; jj++) {
        for (int k = 0; k < block_cols_a; k += 2) {
          result[i][jj] += a[i][k] * b[k][jj] + a[i][k+1] * b[k+1][jj];
          result[i+1][jj] += a[i+1][k] * b[k][jj] + a[i+1][k+1] * b[k+1][jj];
        }
        // Handle remaining k
        for (int k = block_cols_a; k < cols_a; k++) {
          result[i][jj] += a[i][k] * b[k][jj];
          result[i+1][jj] += a[i+1][k] * b[k][jj];
        }
      }
      
      // Handle remaining k for processed columns
      for (int k = block_cols_a; k < cols_a; k++) {
        result[i][j] += a[i][k] * b[k][j];
        result[i][j+1] += a[i][k] * b[k][j+1];
        result[i+1][j] += a[i+1][k] * b[k][j];
        result[i+1][j+1] += a[i+1][k] * b[k][j+1];
      }
    }
  }
  
  // Handle remaining rows
  for (int i = block_rows; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      for (int k = 0; k < cols_a; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

// Optimized matrix multiplication using Winograd 2x2 blocks for float16
static inline void winograd_matmul_blocked_f16(__fp16 *const *a, __fp16 *const *b, __fp16 **result,
                                               int rows_a, int cols_a, int cols_b) {
  // Process in 2x2 blocks where possible
  int block_rows = (rows_a / 2) * 2;
  int block_cols_a = (cols_a / 2) * 2;
  int block_cols_b = (cols_b / 2) * 2;
  
  for (int i = 0; i < block_rows; i += 2) {
    for (int j = 0; j < block_cols_b; j += 2) {
      // Initialize 2x2 result block
      __fp16 C_block[4] = {0, 0, 0, 0};
      
      for (int k = 0; k < block_cols_a; k += 2) {
        // Load 2x2 blocks from A and B
        __fp16 A_block[4] = {
          a[i][k], a[i][k+1],
          a[i+1][k], a[i+1][k+1]
        };
        __fp16 B_block[4] = {
          b[k][j], b[k][j+1],
          b[k+1][j], b[k+1][j+1]
        };
        
        // Compute partial result using Winograd
        __fp16 partial_C[4];
        winograd_matmul_2x2_f16(A_block, B_block, partial_C);
        
        // Accumulate to result block
        C_block[0] += partial_C[0];
        C_block[1] += partial_C[1];
        C_block[2] += partial_C[2];
        C_block[3] += partial_C[3];
      }
      
      // Store result block
      result[i][j] += C_block[0];
      result[i][j+1] += C_block[1];
      result[i+1][j] += C_block[2];
      result[i+1][j+1] += C_block[3];
      
      // Handle remaining columns in this block row
      for (int jj = block_cols_b; jj < cols_b; jj++) {
        for (int k = 0; k < block_cols_a; k += 2) {
          result[i][jj] += a[i][k] * b[k][jj] + a[i][k+1] * b[k+1][jj];
          result[i+1][jj] += a[i+1][k] * b[k][jj] + a[i+1][k+1] * b[k+1][jj];
        }
        // Handle remaining k
        for (int k = block_cols_a; k < cols_a; k++) {
          result[i][jj] += a[i][k] * b[k][jj];
          result[i+1][jj] += a[i+1][k] * b[k][jj];
        }
      }
      
      // Handle remaining k for processed columns
      for (int k = block_cols_a; k < cols_a; k++) {
        result[i][j] += a[i][k] * b[k][j];
        result[i][j+1] += a[i][k] * b[k][j+1];
        result[i+1][j] += a[i+1][k] * b[k][j];
        result[i+1][j+1] += a[i+1][k] * b[k][j+1];
      }
    }
  }
  
  // Handle remaining rows
  for (int i = block_rows; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      for (int k = 0; k < cols_a; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}
#endif

extern void nsc_conv2d_f16(__fp16 *const *signal,
                       __fp16 *const *filter,
                       __fp16 **result,
                       NSC_Size stride,
                       NSC_Padding padding,
                       NSC_Size filter_size,
                       NSC_Size input_size) {
  int paddingLeft;
  int paddingRight;
  int paddingBottom;
  int paddingTop;
  
  int *pad_l_ptr = &paddingLeft;
  int *pad_r_ptr = &paddingRight;
  int *pad_b_ptr = &paddingBottom;
  int *pad_t_ptr = &paddingTop;
  
  nsc_padding_calculation(stride,
                          padding,
                          filter_size,
                          input_size,
                          pad_t_ptr,
                          pad_b_ptr,
                          pad_l_ptr,
                          pad_r_ptr);
  
  int padded_row_total = input_size.rows + paddingLeft + paddingRight;
  int padded_col_total = input_size.columns + paddingTop + paddingBottom;
  
  // Optimized memory allocation for padding - use single malloc for better cache performance
  __fp16 **working_signal = NULL;
  __fp16 *working_signal_data = NULL;
  int use_heap_allocation = 0;
  
  if (padding == same) {
    int inputRows = input_size.rows;
    int inputColumns = input_size.columns;
    
    int padded_row_total = inputRows + paddingLeft + paddingRight;
    int padded_col_total = inputColumns + paddingTop + paddingBottom;
    
    // Allocate memory more efficiently - single block + row pointers
    working_signal_data = (__fp16 *)calloc(padded_row_total * padded_col_total, sizeof(__fp16));
    working_signal = (__fp16 **)malloc(padded_row_total * sizeof(__fp16 *));
    
    if (working_signal == NULL || working_signal_data == NULL) {
      fprintf(stderr, "Memory allocation failed.\n");
      if (working_signal_data) free(working_signal_data);
      if (working_signal) free(working_signal);
      return;
    }
    
    use_heap_allocation = 1;
    
    // Set up row pointers to point into the single data block
    for (int i = 0; i < padded_row_total; i++) {
      working_signal[i] = &working_signal_data[i * padded_col_total];
    }
    
    // Copy input data (data is already zero-initialized by calloc)
    for (int r = 0; r < inputRows; r++) {
      for (int c = 0; c < inputColumns; c++) {
        int padded_r = r + paddingTop;
        int padded_c = c + paddingLeft;
        working_signal[padded_r][padded_c] = signal[r][c];
      }
    }
    
    if (result == NULL || signal == NULL)
      return;
  }
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  if (result == NULL)
    return;
  
  int rf = filterRows;
  int cf = filterColumns;
  int rd = inputRows + paddingTop + paddingBottom; //havnt dealt with padding yet
  int cd = inputColumns + paddingLeft + paddingRight;
  
  int max_r = rd - rf + 1;
  int max_c = cd - cf + 1;
  
  int rows = ((inputRows - filterRows + paddingTop + paddingBottom) / strideR) + 1;
  int columns = ((inputColumns - filterColumns + paddingLeft + paddingRight) / strideC) + 1;
  
  int expected_r = ((inputRows - filterRows + paddingTop + paddingBottom) / strideR) + 1;
  int expected_c = ((inputColumns - filterColumns + paddingLeft + paddingRight) / strideC) + 1;
  
  int result_index_r = 0;
  for (int r = 0; r < max_r; r += strideR) {
    int result_index_c = 0;
    for (int c = 0; c < max_c; c += strideC) {
      __fp16 sum = 0;
      
#ifdef __ARM_NEON
      // Use specialized kernels for common filter sizes
      if (filterRows == 1 && filterColumns == 1) {
        // 1x1 pointwise convolution
        if (padding == same) {
          sum = neon_conv_1x1_f16(working_signal, filter, r, c);
        } else {
          sum = neon_conv_1x1_f16(signal, filter, r, c);
        }
      } else if (filterRows == 3 && filterColumns == 3) {
        // 3x3 convolution with Winograd optimization
        if (padding == same) {
          sum = winograd_3x3_f16_single(working_signal, filter, r, c);
        } else {
          sum = winograd_3x3_f16_single(signal, filter, r, c);
        }
      } else if (filterRows == 5 && filterColumns == 5) {
        // 5x5 convolution
        if (padding == same) {
          sum = neon_conv_5x5_f16(working_signal, filter, r, c);
        } else {
          sum = neon_conv_5x5_f16(signal, filter, r, c);
        }
      } else {
        // Use general SIMD optimized convolution kernel for other sizes
        if (padding == same) {
          sum = neon_conv_kernel_f16(working_signal, filter, r, c, filterRows, filterColumns);
        } else {
          sum = neon_conv_kernel_f16(signal, filter, r, c, filterRows, filterColumns);
        }
      }
#else
      // Fallback to scalar implementation
      for (int fr = 0; fr < filterRows; fr++) {
        for (int fc = 0; fc < filterColumns; fc++) {
          int current_data_row = r + fr;
          int current_data_col = c + fc;
          
          __fp16 s_data = 0; // some checking of size here?
          
          if (padding == same) {
            s_data = working_signal[current_data_row][current_data_col];
          } else {
            s_data = signal[current_data_row][current_data_col];
          }
          
          __fp16 f_data = filter[fr][fc]; //do some checking of size here?
          sum += s_data * f_data;
        }
      }
#endif
      
      result[result_index_r][result_index_c] = sum;
      result_index_c++;
    }
    result_index_r++;
  }

  if (padding == same && use_heap_allocation) {
    free(working_signal_data);
    free(working_signal);
  }
}

extern void nsc_conv2d(float *const *signal,
                       float *const *filter,
                       float **result,
                       NSC_Size stride,
                       NSC_Padding padding,
                       NSC_Size filter_size,
                       NSC_Size input_size) {
  int paddingLeft;
  int paddingRight;
  int paddingBottom;
  int paddingTop;
  
  int *pad_l_ptr = &paddingLeft;
  int *pad_r_ptr = &paddingRight;
  int *pad_b_ptr = &paddingBottom;
  int *pad_t_ptr = &paddingTop;
  
  nsc_padding_calculation(stride,
                          padding,
                          filter_size,
                          input_size,
                          pad_t_ptr,
                          pad_b_ptr,
                          pad_l_ptr,
                          pad_r_ptr);
  
  int padded_row_total = input_size.rows + paddingLeft + paddingRight;
  int padded_col_total = input_size.columns + paddingTop + paddingBottom;
  
  // Optimized memory allocation for padding - use single malloc for better cache performance
  float **working_signal = NULL;
  float *working_signal_data = NULL;
  int use_heap_allocation = 0;
  
  if (padding == same) {
    int inputRows = input_size.rows;
    int inputColumns = input_size.columns;
    
    int padded_row_total = inputRows + paddingLeft + paddingRight;
    int padded_col_total = inputColumns + paddingTop + paddingBottom;
    
    // Allocate memory more efficiently - single block + row pointers
    working_signal_data = (float *)calloc(padded_row_total * padded_col_total, sizeof(float));
    working_signal = (float **)malloc(padded_row_total * sizeof(float *));
    
    if (working_signal == NULL || working_signal_data == NULL) {
      fprintf(stderr, "Memory allocation failed.\n");
      if (working_signal_data) free(working_signal_data);
      if (working_signal) free(working_signal);
      return;
    }
    
    use_heap_allocation = 1;
    
    // Set up row pointers to point into the single data block
    for (int i = 0; i < padded_row_total; i++) {
      working_signal[i] = &working_signal_data[i * padded_col_total];
    }
    
    // Copy input data (data is already zero-initialized by calloc)
    for (int r = 0; r < inputRows; r++) {
      for (int c = 0; c < inputColumns; c++) {
        int padded_r = r + paddingTop;
        int padded_c = c + paddingLeft;
        working_signal[padded_r][padded_c] = signal[r][c];
      }
    }
    
    if (result == NULL || signal == NULL)
      return;
  }
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  if (result == NULL)
    return;
  
  int rf = filterRows;
  int cf = filterColumns;
  int rd = inputRows + paddingTop + paddingBottom; //havnt dealt with padding yet
  int cd = inputColumns + paddingLeft + paddingRight;
  
  int max_r = rd - rf + 1;
  int max_c = cd - cf + 1;
  
  int rows = ((inputRows - filterRows + paddingTop + paddingBottom) / strideR) + 1;
  int columns = ((inputColumns - filterColumns + paddingLeft + paddingRight) / strideC) + 1;
  
  int expected_r = ((inputRows - filterRows + paddingTop + paddingBottom) / strideR) + 1;
  int expected_c = ((inputColumns - filterColumns + paddingLeft + paddingRight) / strideC) + 1;
  
  int result_index_r = 0;
  for (int r = 0; r < max_r; r += strideR) {
    int result_index_c = 0;
    for (int c = 0; c < max_c; c += strideC) {
      float sum = 0;
      
#ifdef __ARM_NEON
      // Use specialized kernels for common filter sizes
      if (filterRows == 1 && filterColumns == 1) {
        // 1x1 pointwise convolution
        if (padding == same) {
          sum = neon_conv_1x1(working_signal, filter, r, c);
        } else {
          sum = neon_conv_1x1(signal, filter, r, c);
        }
      } else if (filterRows == 3 && filterColumns == 3) {
        // 3x3 convolution with Winograd optimization
        if (padding == same) {
          sum = winograd_3x3_f32_single(working_signal, filter, r, c);
        } else {
          sum = winograd_3x3_f32_single(signal, filter, r, c);
        }
      } else if (filterRows == 5 && filterColumns == 5) {
        // 5x5 convolution
        if (padding == same) {
          sum = neon_conv_5x5(working_signal, filter, r, c);
        } else {
          sum = neon_conv_5x5(signal, filter, r, c);
        }
      } else {
        // Use general SIMD optimized convolution kernel for other sizes
        if (padding == same) {
          sum = neon_conv_kernel(working_signal, filter, r, c, filterRows, filterColumns);
        } else {
          sum = neon_conv_kernel(signal, filter, r, c, filterRows, filterColumns);
        }
      }
#else
      // Fallback to scalar implementation
      for (int fr = 0; fr < filterRows; fr++) {
        for (int fc = 0; fc < filterColumns; fc++) {
          int current_data_row = r + fr;
          int current_data_col = c + fc;
          
          float s_data = 0; // some checking of size here?
          
          if (padding == same) {
            s_data = working_signal[current_data_row][current_data_col];
          } else {
            s_data = signal[current_data_row][current_data_col];
          }
          
          float f_data = filter[fr][fc]; //do some checking of size here?
          sum += s_data * f_data;
        }
      }
#endif
      
      result[result_index_r][result_index_c] = sum;
      result_index_c++;
    }
    result_index_r++;
  }

  if (padding == same && use_heap_allocation) {
    free(working_signal_data);
    free(working_signal);
  }
}

extern void nsc_conv1d_f16(const __fp16 signal[],
                           const __fp16 filter[],
                           __fp16 *result,
                           NSC_Size stride,
                           NSC_Padding padding,
                           NSC_Size filter_size,
                           NSC_Size input_size) {
  int paddingLeft;
  int paddingRight;
  int paddingBottom;
  int paddingTop;
  
  int *pad_l_ptr = &paddingLeft;
  int *pad_r_ptr = &paddingRight;
  int *pad_b_ptr = &paddingBottom;
  int *pad_t_ptr = &paddingTop;
  
  nsc_padding_calculation(stride,
                          padding,
                          filter_size,
                          input_size,
                          pad_t_ptr,
                          pad_b_ptr,
                          pad_l_ptr,
                          pad_r_ptr);
  
  int padded_row_total = input_size.rows + paddingLeft + paddingRight;
  int padded_col_total = input_size.columns + paddingTop + paddingBottom;
  
  __fp16 working_signal[padded_row_total * padded_col_total];
  if (padding == same) {
    nsc_zero_pad_f16(signal,
                     working_signal,
                     filter_size,
                     input_size,
                     stride);
  }
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  if (result == NULL)
    return;
  
  int rf = filterRows;
  int cf = filterColumns;
  int rd = inputRows + paddingTop + paddingBottom; //havnt dealt with padding yet
  int cd = inputColumns + paddingLeft + paddingRight;
  
  int max_r = rd - rf + 1;
  int max_c = cd - cf + 1;

  int expected_r = ((inputRows - filterRows + paddingTop + paddingBottom) / strideR) + 1;
  int expected_c = ((inputColumns - filterColumns + paddingLeft + paddingRight) / strideC) + 1;

  __fp16 mutable_result[expected_r * expected_c]; //= malloc(expected_r * expected_c * sizeof(float));

  for (int i = 0; i < expected_r * expected_c; i++) {
    mutable_result[i] = 0.0f;
  }

  int result_index = 0;
  for (int r = 0; r < max_r; r += strideR) {
    for (int c = 0; c < max_c; c += strideC) {
      __fp16 sum = 0;
      
      for (int fr = 0; fr < filterRows; fr++) {
        
        for (int fc = 0; fc < filterColumns; fc++) {
          int current_data_row = r + fr;
          int current_data_col = c + fc;
          
          int signal_index = (current_data_row * cd) + current_data_col;
          int filter_index = (fr * cf) + fc;
          
          __fp16 s_data = padding == valid ? signal[signal_index] : working_signal[signal_index]; //do some checking of size here?
          __fp16 f_data = filter[filter_index]; //do some checking of size here?
          sum += s_data * f_data;
        }
      }
      
      mutable_result[result_index] = sum;
      result_index += 1;
    }
  }
  
  memcpy(result, mutable_result, expected_r * expected_c * sizeof(__fp16));
}

extern void nsc_conv1d(const float signal[],
                       const float filter[],
                       float *result,
                       NSC_Size stride,
                       NSC_Padding padding,
                       NSC_Size filter_size,
                       NSC_Size input_size) {
  int paddingLeft;
  int paddingRight;
  int paddingBottom;
  int paddingTop;
  
  int *pad_l_ptr = &paddingLeft;
  int *pad_r_ptr = &paddingRight;
  int *pad_b_ptr = &paddingBottom;
  int *pad_t_ptr = &paddingTop;
  
  nsc_padding_calculation(stride,
                          padding,
                          filter_size,
                          input_size,
                          pad_t_ptr,
                          pad_b_ptr,
                          pad_l_ptr,
                          pad_r_ptr);
  
  int padded_row_total = input_size.rows + paddingLeft + paddingRight;
  int padded_col_total = input_size.columns + paddingTop + paddingBottom;
  
  float working_signal[padded_row_total * padded_col_total];// = malloc(padded_row_total * padded_col_total * sizeof(float));
  if (padding == same) {
    nsc_zero_pad(signal,
                 working_signal,
                 filter_size,
                 input_size,
                 stride);
  }
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  if (result == NULL)
    return;
  
  int rf = filterRows;
  int cf = filterColumns;
  int rd = inputRows + paddingTop + paddingBottom; //havnt dealt with padding yet
  int cd = inputColumns + paddingLeft + paddingRight;
  
  int max_r = rd - rf + 1;
  int max_c = cd - cf + 1;

  int expected_r = ((inputRows - filterRows + paddingTop + paddingBottom) / strideR) + 1;
  int expected_c = ((inputColumns - filterColumns + paddingLeft + paddingRight) / strideC) + 1;

  float mutable_result[expected_r * expected_c]; //= malloc(expected_r * expected_c * sizeof(float));

  for (int i = 0; i < expected_r * expected_c; i++) {
    mutable_result[i] = 0.0f;
  }

  int result_index = 0;
  for (int r = 0; r < max_r; r += strideR) {
    for (int c = 0; c < max_c; c += strideC) {
      float sum = 0;
      
      for (int fr = 0; fr < filterRows; fr++) {
        
        for (int fc = 0; fc < filterColumns; fc++) {
          int current_data_row = r + fr;
          int current_data_col = c + fc;
          
          int signal_index = (current_data_row * cd) + current_data_col;
          int filter_index = (fr * cf) + fc;
          
          float s_data = padding == valid ? signal[signal_index] : working_signal[signal_index]; //do some checking of size here?
          float f_data = filter[filter_index]; //do some checking of size here?
          sum += s_data * f_data;
        }
      }
      
      mutable_result[result_index] = sum;
      result_index += 1;
    }
  }
  
  memcpy(result, mutable_result, expected_r * expected_c * sizeof(float));
}

extern void nsc_transConv2d_f16(__fp16 *const *signal,
                            __fp16 *const *filter,
                            __fp16 **result,
                            NSC_Size stride,
                            NSC_Padding padding,
                            NSC_Size filter_size,
                            NSC_Size input_size) {
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  int rows = (inputRows - 1) * strideR + filterRows;
  int columns = (inputColumns - 1) * strideC + filterColumns;
  // Dynamically allocate memory for the array of pointers (rows)
  __fp16 **working_result = (__fp16 **)malloc(rows * sizeof(__fp16 *));
  
  // Check if allocation was successful
  if (working_result == NULL) {
    fprintf(stderr, "Memory allocation failed.\n");
    return; // Exit with an error code
  }
  
  // Dynamically allocate memory for each row (columns)
  for (int i = 0; i < rows; ++i) {
    working_result[i] = (__fp16 *)malloc(columns * sizeof(__fp16));
    
    // Check if allocation was successful
    if (working_result[i] == NULL) {
      fprintf(stderr, "Memory allocation failed.\n");
      return; // Exit with an error code
    }
  }
  
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      working_result[i][j] = 0;
    }
  }
  
  if (result == NULL)
    return;
  
  for (int i = 0; i < inputRows; i++) {
    int i_prime = i * strideR;
    
    for (int j = 0; j < inputColumns; j++) {
      int j_prime = j * strideC;
      
      for (int r = 0; r < filterRows; r++) {
        for (int c = 0; c < filterColumns; c++) {
          int signal_index = (i * inputRows) + j;
          int filter_index = (r * filterRows) + c;
          
          working_result[r + i_prime][c + j_prime] += signal[i][j] * filter[r][c];
        }
      }
      
    }
  }
  
  int pad_left = 0;
  int pad_right = 0;
  int pad_top = 0;
  int pad_bottom = 0;
  
  if (padding == same) {
    pad_left = (int)floor(((double)filterRows - (double)strideR) / (double)2);
    pad_right = filterRows - strideR - pad_left;
    pad_top = (int)floor(((double)filterColumns - (double)strideC) / (double)2);
    pad_bottom = filterColumns - strideC - pad_top;
  }
  
  int padded_row_total = rows - (pad_bottom + pad_top);
  int padded_col_total = columns - (pad_left + pad_right);
  
  int padded_index_c = 0;
  int padded_index_r = 0;
  for (int r = pad_top; r < rows - pad_bottom; r++) {
    padded_index_c = 0;
    for (int c = pad_left; c < columns - pad_right; c++) {
      __fp16 w_r = working_result[r][c];
      result[padded_index_r][padded_index_c] = w_r;
      padded_index_c++;
    }
    padded_index_r++;
  }
  
  for (int i = 0; i < rows; ++i) {
    free(working_result[i]);
  }
  free(working_result);
}

extern void nsc_transConv1d_f16(const __fp16 signal[],
                            const __fp16 filter[],
                            __fp16 *result,
                            NSC_Size stride,
                            NSC_Padding padding,
                            NSC_Size filter_size,
                            NSC_Size input_size) {
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  int rows = (inputRows - 1) * strideR + filterRows;
  int columns = (inputColumns - 1) * strideC + filterColumns;
  
  int length = rows * columns;
  
  __fp16 working_result[length];
  
  for (int i = 0; i < length; i++) {
    working_result[i] = 0.0f;
  }

  if (result == NULL)
    return;
  
  for (int i = 0; i < inputRows; i++) {
    int i_prime = i * strideR;
    
    for (int j = 0; j < inputColumns; j++) {
      int j_prime = j * strideC;
      
      for (int r = 0; r < filterRows; r++) {
        for (int c = 0; c < filterColumns; c++) {
          int result_index = ((r + i_prime) * rows) + (c + j_prime);
          int signal_index = (i * inputRows) + j;
          int filter_index = (r * filterRows) + c;
          
          working_result[result_index] += signal[signal_index] * filter[filter_index];
        }
      }
  
    }
  }
  
  int pad_left = 0;
  int pad_right = 0;
  int pad_top = 0;
  int pad_bottom = 0;
  
  if (padding == same) {
    pad_left = (int)floor(((double)filterRows - (double)strideR) / (double)2);
    pad_right = filterRows - strideR - pad_left;
    pad_top = (int)floor(((double)filterColumns - (double)strideC) / (double)2);
    pad_bottom = filterColumns - strideC - pad_top;
  }
  
  int padded_row_total = rows - (pad_bottom + pad_top);
  int padded_col_total = columns - (pad_left + pad_right);
  
  __fp16 padded[padded_col_total * padded_row_total];
  
  for (int i = 0; i < padded_col_total * padded_row_total; i++) {
    padded[i] = 0.0f;
  }
  
  int padded_index = 0;
  for (int r = pad_top; r < rows - pad_bottom; r++) {
    for (int c = pad_left; c < columns - pad_right; c++) {
      int index = (r  * rows) + c;
      __fp16 w_r = working_result[index];
      padded[padded_index] = w_r;
      padded_index++;
    }
  }

  memcpy(result, padded, padded_col_total * padded_row_total * sizeof(__fp16));
}


extern void nsc_transConv2d(float *const *signal,
                            float *const *filter,
                            float **result,
                            NSC_Size stride,
                            NSC_Padding padding,
                            NSC_Size filter_size,
                            NSC_Size input_size) {
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  int rows = (inputRows - 1) * strideR + filterRows;
  int columns = (inputColumns - 1) * strideC + filterColumns;
  // Dynamically allocate memory for the array of pointers (rows)
  float **working_result = (float **)malloc(rows * sizeof(float *));
  
  // Check if allocation was successful
  if (working_result == NULL) {
    fprintf(stderr, "Memory allocation failed.\n");
    return; // Exit with an error code
  }
  
  // Dynamically allocate memory for each row (columns)
  for (int i = 0; i < rows; ++i) {
    working_result[i] = (float *)malloc(columns * sizeof(float));
    
    // Check if allocation was successful
    if (working_result[i] == NULL) {
      fprintf(stderr, "Memory allocation failed.\n");
      return; // Exit with an error code
    }
  }
  
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      working_result[i][j] = 0;
    }
  }
  
  if (result == NULL)
    return;
  
  for (int i = 0; i < inputRows; i++) {
    int i_prime = i * strideR;
    
    for (int j = 0; j < inputColumns; j++) {
      int j_prime = j * strideC;
      
      for (int r = 0; r < filterRows; r++) {
        for (int c = 0; c < filterColumns; c++) {
          int signal_index = (i * inputRows) + j;
          int filter_index = (r * filterRows) + c;
          
          working_result[r + i_prime][c + j_prime] += signal[i][j] * filter[r][c];
        }
      }
      
    }
  }
  
  int pad_left = 0;
  int pad_right = 0;
  int pad_top = 0;
  int pad_bottom = 0;
  
  if (padding == same) {
    pad_left = (int)floor(((double)filterRows - (double)strideR) / (double)2);
    pad_right = filterRows - strideR - pad_left;
    pad_top = (int)floor(((double)filterColumns - (double)strideC) / (double)2);
    pad_bottom = filterColumns - strideC - pad_top;
  }
  
  int padded_row_total = rows - (pad_bottom + pad_top);
  int padded_col_total = columns - (pad_left + pad_right);
  
  int padded_index_c = 0;
  int padded_index_r = 0;
  for (int r = pad_top; r < rows - pad_bottom; r++) {
    padded_index_c = 0;
    for (int c = pad_left; c < columns - pad_right; c++) {
      float w_r = working_result[r][c];
      result[padded_index_r][padded_index_c] = w_r;
      padded_index_c++;
    }
    padded_index_r++;
  }
  
  for (int i = 0; i < rows; ++i) {
    free(working_result[i]);
  }
  free(working_result);
}

extern void nsc_transConv1d(const float signal[],
                            const float filter[],
                            float *result,
                            NSC_Size stride,
                            NSC_Padding padding,
                            NSC_Size filter_size,
                            NSC_Size input_size) {
  
  int inputRows = input_size.rows;
  int inputColumns = input_size.columns;
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  int rows = (inputRows - 1) * strideR + filterRows;
  int columns = (inputColumns - 1) * strideC + filterColumns;
  
  int length = rows * columns;
  
  float working_result[length];
  
  for (int i = 0; i < length; i++) {
    working_result[i] = 0.0f;
  }

  if (result == NULL)
    return;
  
  for (int i = 0; i < inputRows; i++) {
    int i_prime = i * strideR;
    
    for (int j = 0; j < inputColumns; j++) {
      int j_prime = j * strideC;
      
      for (int r = 0; r < filterRows; r++) {
        for (int c = 0; c < filterColumns; c++) {
          int result_index = ((r + i_prime) * rows) + (c + j_prime);
          int signal_index = (i * inputRows) + j;
          int filter_index = (r * filterRows) + c;
          
          working_result[result_index] += signal[signal_index] * filter[filter_index];
        }
      }
  
    }
  }
  
  int pad_left = 0;
  int pad_right = 0;
  int pad_top = 0;
  int pad_bottom = 0;
  
  if (padding == same) {
    pad_left = (int)floor(((double)filterRows - (double)strideR) / (double)2);
    pad_right = filterRows - strideR - pad_left;
    pad_top = (int)floor(((double)filterColumns - (double)strideC) / (double)2);
    pad_bottom = filterColumns - strideC - pad_top;
  }
  
  int padded_row_total = rows - (pad_bottom + pad_top);
  int padded_col_total = columns - (pad_left + pad_right);
  
  float padded[padded_col_total * padded_row_total];
  
  for (int i = 0; i < padded_col_total * padded_row_total; i++) {
    padded[i] = 0.0f;
  }
  
  int padded_index = 0;
  for (int r = pad_top; r < rows - pad_bottom; r++) {
    for (int c = pad_left; c < columns - pad_right; c++) {
      int index = (r  * rows) + c;
      float w_r = working_result[index];
      padded[padded_index] = w_r;
      padded_index++;
    }
  }

  memcpy(result, padded, padded_col_total * padded_row_total * sizeof(float));
}
