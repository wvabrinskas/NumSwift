#include "include/numswiftc_base.h"
#include "time.h"

// MARK: - Sum
extern _Float16 nsc_sum(const _Float16 array[]) {
  _Float16 sum = 0;

  int size = sizeof(*array);
  
  for (int i = 0; i < size; i++) {
    sum += array[i];
  }
  return sum;
}

// MARK: - Sum of Squares
extern _Float16 nsc_sum_of_squares(const _Float16 array[]) {

  int size = sizeof(*array);
  _Float16 sum = 0;
  for (int i = 0; i < size; i++) {
    sum += array[i] * array[i];
  }
  return sum;
}

// MARK: - Index of Min
extern NSC_IndexedValue nsc_index_of_min(const _Float16 array[]) {
  int size = sizeof(*array);
  _Float16 min = array[0];
  int index = 0;
  for (int i = 1; i < size; i++) {
    if (array[i] < min) {
      min = array[i];
      index = i;
    }
  }


  NSC_IndexedValue result;
  result.index = index; 
  result.value = min;

  return result;
}

// MARK: - Index of Max
extern NSC_IndexedValue nsc_index_of_max(const _Float16 array[]) {
  int size = sizeof(*array);
  _Float16 max = array[0];
  int index = 0;
  for (int i = 1; i < size; i++) {
    if (array[i] > max) {
      max = array[i];
      index = i;
    }
  }

  NSC_IndexedValue result;
  result.index = index; 
  result.value = max;

  return result;
}

// MARK: - Max
extern _Float16 nsc_max(const _Float16 array[]) {
  int size = sizeof(*array);
  _Float16 max = array[0];
  for (int i = 1; i < size; i++) {
    if (array[i] > max) {
      max = array[i];
    }
  }
  return max;
}

// MARK: - Min
extern _Float16 nsc_min(const _Float16 array[]) {
  int size = sizeof(*array);
  _Float16 min = array[0];
  for (int i = 1; i < size; i++) {
    if (array[i] < min) {
      min = array[i];
    }
  }
  return min;
}

// MARK: - Mean
extern _Float16 nsc_mean(const _Float16 array[]) {
  int size = sizeof(*array);
  return nsc_sum(array) / size;
}

// MARK: - Add

extern void nsc_add_scalar(const _Float16 lhs, const _Float16 rhs[], _Float16 *result) {
  int size = sizeof(*rhs);
  for (int i = 0; i < size; i++) {
    result[i] = rhs[i] + lhs;
  }
}

extern void nsc_add(const _Float16 lhs[], const _Float16 rhs[], _Float16 *result) {
  int size = sizeof(*lhs);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] + rhs[i];
  }
}

// MARK: - Sub

extern void nsc_sub(const _Float16 lhs[], const _Float16 rhs[], _Float16 *result) {
  int size = sizeof(*lhs);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] - rhs[i];
  }
}

// MARK: - Mult

extern void nsc_mult_scalar(const _Float16 lhs, const _Float16 rhs[], _Float16 *result) {
  int size = sizeof(*rhs);
  for (int i = 0; i < size; i++) {
    result[i] = lhs * rhs[i];
  }
}

extern void nsc_mult(const _Float16 lhs[], const _Float16 rhs[], _Float16 *result) {
  int size = sizeof(*lhs);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] * rhs[i];
  }
}

// MARK: - Div

extern void nsc_div(const _Float16 lhs[], const _Float16 rhs[], _Float16 *result) {
  int size = sizeof(*lhs);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] / rhs[i];
  }
}

extern void nsc_div_scalar_array(const _Float16 lhs, const _Float16 rhs[], _Float16 *result) {
  int size = sizeof(*rhs);
  for (int i = 0; i < size; i++) {
    result[i] = lhs / rhs[i];
  }
}

extern void nsc_div_array_scalar(const _Float16 lhs[], const _Float16 rhs, _Float16 *result) {
  int size = sizeof(*lhs);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] / rhs;
  }
}

extern void nsc_add2d_f16(NSC_Size size,
                          _Float16 *const *a,
                          _Float16 *const *b,
                          _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }
}

extern void nsc_sub2d_f16(NSC_Size size,
                          _Float16 *const *a,
                          _Float16 *const *b,
                          _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] - b[i][j];
    }
  }
}

extern void nsc_divide2d_f16(NSC_Size size,
                             _Float16 *const *a,
                             _Float16 *const *b,
                             _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] / b[i][j];
    }
  }
}

extern void nsc_mult2d_f16(NSC_Size size,
                           _Float16 *const *a,
                           _Float16 *const *b,
                           _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] * b[i][j];
    }
  }
}


/// Float 32

extern void nsc_add2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                      float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }
}

extern void nsc_sub2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                      float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] - b[i][j];
    }
  }
}

extern void nsc_divide2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                      float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] / b[i][j];
    }
  }
}

extern void nsc_mult2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                      float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] * b[i][j];
    }
  }
}

// 2D arithmetic with scalar operations
/// Float32 2D with scalar
extern void nsc_add2d_scalar(NSC_Size size,
                             float *const *a,
                             float scalar,
                             float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] + scalar;
    }
  }
}

extern void nsc_add2d_array_scalar(NSC_Size size,
                                   float *const *a,
                                   const float *scalar_array,
                                   float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] + scalar_array[i];
    }
  }
}

extern void nsc_sub2d_scalar(NSC_Size size,
                             float *const *a,
                             float scalar,
                             float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] - scalar;
    }
  }
}

extern void nsc_sub2d_array_scalar(NSC_Size size,
                                   float *const *a,
                                   const float *scalar_array,
                                   float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] - scalar_array[i];
    }
  }
}

extern void nsc_mult2d_scalar(NSC_Size size,
                              float *const *a,
                              float scalar,
                              float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] * scalar;
    }
  }
}

extern void nsc_mult2d_array_scalar(NSC_Size size,
                                    float *const *a,
                                    const float *scalar_array,
                                    float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] * scalar_array[i];
    }
  }
}

extern void nsc_divide2d_scalar(NSC_Size size,
                                float *const *a,
                                float scalar,
                                float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] / scalar;
    }
  }
}

extern void nsc_divide2d_array_scalar(NSC_Size size,
                                      float *const *a,
                                      const float *scalar_array,
                                      float **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] / scalar_array[i];
    }
  }
}

/// Float16 2D with scalar
extern void nsc_add2d_scalar_f16(NSC_Size size,
                                 _Float16 *const *a,
                                 _Float16 scalar,
                                 _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] + scalar;
    }
  }
}

extern void nsc_add2d_array_scalar_f16(NSC_Size size,
                                       _Float16 *const *a,
                                       const _Float16 *scalar_array,
                                       _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] + scalar_array[i];
    }
  }
}

extern void nsc_sub2d_scalar_f16(NSC_Size size,
                                 _Float16 *const *a,
                                 _Float16 scalar,
                                 _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] - scalar;
    }
  }
}

extern void nsc_sub2d_array_scalar_f16(NSC_Size size,
                                       _Float16 *const *a,
                                       const _Float16 *scalar_array,
                                       _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] - scalar_array[i];
    }
  }
}

extern void nsc_mult2d_scalar_f16(NSC_Size size,
                                  _Float16 *const *a,
                                  _Float16 scalar,
                                  _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] * scalar;
    }
  }
}

extern void nsc_mult2d_array_scalar_f16(NSC_Size size,
                                        _Float16 *const *a,
                                        const _Float16 *scalar_array,
                                        _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] * scalar_array[i];
    }
  }
}

extern void nsc_divide2d_scalar_f16(NSC_Size size,
                                    _Float16 *const *a,
                                    _Float16 scalar,
                                    _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] / scalar;
    }
  }
}

extern void nsc_divide2d_array_scalar_f16(NSC_Size size,
                                          _Float16 *const *a,
                                          const _Float16 *scalar_array,
                                          _Float16 **result) {
  
  int rows = size.rows;
  int columns = size.columns;

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < columns; ++j) {
      result[i][j] = a[i][j] / scalar_array[i];
    }
  }
}



