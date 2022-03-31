
#include "include/numswiftc.h"

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
  
  if (padding == 1) {
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
  
  int strideR = stride.rows;
  int strideC = stride.columns;
  
  int filterRows = filter_size.rows;
  int filterColumns = filter_size.columns;
  
  int padded_row_total = inputRows + paddingLeft + paddingRight;
  int padded_col_total = inputColumns + paddingTop + paddingBottom;
  
  int length = padded_row_total * padded_col_total;
  float *padded = malloc(length * sizeof(float));
  
  for (int i = 0; i < padded_row_total * padded_col_total; i++) {
    padded[i] = 0;
  }
  
  if (padded == NULL || input == NULL)
    return NULL;
  
  for (int r = 0; r < inputRows; r++) {
    for (int c = 0; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      int padded_r = r + paddingTop;
      
      int index = (padded_r  * padded_row_total) + padded_c;
      padded[index] = input[(r * inputRows) + c];
    }
  }
    
  memmove(result, padded, length * sizeof(float));
  
  free(padded);
}

extern void nsc_conv2d(const float signal[],
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
  
  float *working_signal = malloc(padded_row_total * padded_col_total * sizeof(float));
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

  float *mutable_result = malloc(expected_r * expected_c * sizeof(float));
  
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
          
          int signal_index = (current_data_row * rd) + current_data_col;
          int filter_index = (fr * rf) + fc;
          
          float s_data = padding == valid ? signal[signal_index] : working_signal[signal_index]; //do some checking of size here?
          float f_data = filter[filter_index]; //do some checking of size here?
          sum += s_data * f_data;
        }
      }
      
      mutable_result[result_index] = sum;
      result_index += 1;
    }
  }
  
  memmove(result, mutable_result, expected_r * expected_c * sizeof(float));
  
  free(mutable_result);
  free(working_signal);
}

extern void nsc_transConv2d(const float signal[],
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
  
  int length = rows * columns; //calculate expected output size
  
  float *working_result = malloc(length * sizeof(float));
  
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
  
  float *padded = malloc(padded_col_total * padded_row_total * sizeof(float));
  
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
  
  if (padded == NULL)
    return;
  
  memmove(result, padded, padded_col_total * padded_row_total * sizeof(float));
  
  free(padded);
  free(working_result);
}
