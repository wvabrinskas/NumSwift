
#include "include/numswiftc.h"
#include "time.h"

extern void nsc_matmul_16(NSC_Size a_size,
                          NSC_Size b_size,
                          __fp16 *const *a,
                          __fp16 *const *b,
                          __fp16 **result) {
  
  int rowFirst = a_size.rows;
  int columnFirst = a_size.columns;
  int columnSecond = b_size.columns;
  
  // Multiplying firstMatrix and secondMatrix and storing in result.
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
  
  // Multiplying firstMatrix and secondMatrix and storing in result.
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
  
  // Dynamically allocate memory for the array of pointers (rows)
  __fp16 **working_signal;
  
  if (padding == same) {
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
    
    working_signal = (__fp16 **)malloc(padded_row_total * sizeof(__fp16 *));
    
    // Check if allocation was successful
    if (working_signal == NULL) {
      fprintf(stderr, "Memory allocation failed.\n");
      return; // Exit with an error code
    }
    
    // Dynamically allocate memory for each row (columns)
    for (int i = 0; i < padded_row_total; ++i) {
      working_signal[i] = (__fp16 *)malloc(padded_col_total * sizeof(__fp16));
      
      // Check if allocation was successful
      if (working_signal[i] == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return; // Exit with an error code
      }
    }
    
    int length = padded_row_total * padded_col_total;
    
    for (int i = 0; i < padded_row_total; i++) {
      for (int j = 0; j < padded_col_total; j++) {
        working_signal[i][j] = 0;
      }
    }
    
    if (result == NULL || signal == NULL)
      return;
    
    for (int r = 0; r < inputRows; r++) {
      for (int c = 0; c < inputColumns; c++) {
        int padded_c = c + paddingLeft;
        int padded_r = r + paddingTop;
        
        int index = (padded_r  * padded_row_total) + padded_c;
        working_signal[padded_r][padded_c] = signal[r][c];
      }
    }
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
      
      result[result_index_r][result_index_c] = sum;
      result_index_c++;
    }
    result_index_r++;
  }

  if (padding == same) {
    for (int i = 0; i < padded_row_total; ++i) {
      free(working_signal[i]);
    }
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
  
  // Dynamically allocate memory for the array of pointers (rows)
  float **working_signal;
  
  if (padding == same) {
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
    
    working_signal = (float **)malloc(padded_row_total * sizeof(float *));
    
    // Check if allocation was successful
    if (working_signal == NULL) {
      fprintf(stderr, "Memory allocation failed.\n");
      return; // Exit with an error code
    }
    
    // Dynamically allocate memory for each row (columns)
    for (int i = 0; i < padded_row_total; ++i) {
      working_signal[i] = (float *)malloc(padded_col_total * sizeof(float));
      
      // Check if allocation was successful
      if (working_signal[i] == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return; // Exit with an error code
      }
    }
    
    int length = padded_row_total * padded_col_total;
    
    for (int i = 0; i < padded_row_total; i++) {
      for (int j = 0; j < padded_col_total; j++) {
        working_signal[i][j] = 0;
      }
    }
    
    if (result == NULL || signal == NULL)
      return;
    
    for (int r = 0; r < inputRows; r++) {
      for (int c = 0; c < inputColumns; c++) {
        int padded_c = c + paddingLeft;
        int padded_r = r + paddingTop;
        
        int index = (padded_r  * padded_row_total) + padded_c;
        working_signal[padded_r][padded_c] = signal[r][c];
      }
    }
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
      
      result[result_index_r][result_index_c] = sum;
      result_index_c++;
    }
    result_index_r++;
  }

  if (padding == same) {
    for (int i = 0; i < padded_row_total; ++i) {
      free(working_signal[i]);
    }
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
