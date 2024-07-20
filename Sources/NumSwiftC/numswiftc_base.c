#include "include/numswiftc_base.h"
#include "time.h"

// MARK: - Sum
extern __fp16 nsc_sum(__fp16 *array) {
  __fp16 sum = 0;

  int size = sizeof(array) / sizeof(array[0]);
  
  for (int i = 0; i < size; i++) {
    sum += array[i];
  }
  return sum;
}

// MARK: - Sum of Squares
extern __fp16 nsc_sum_of_squares(__fp16 *array) {

  int size = sizeof(array) / sizeof(array[0]);
  __fp16 sum = 0;
  for (int i = 0; i < size; i++) {
    sum += array[i] * array[i];
  }
  return sum;
}

// MARK: - Index of Min
extern NSC_IndexedValue nsc_index_of_min(__fp16 *array) {
  int size = sizeof(array) / sizeof(array[0]);
  __fp16 min = array[0];
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
extern NSC_IndexedValue nsc_index_of_max(__fp16 *array) {
  int size = sizeof(array) / sizeof(array[0]);
  __fp16 max = array[0];
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
extern __fp16 nsc_max(__fp16 *array) {
  int size = sizeof(array) / sizeof(array[0]);
  __fp16 max = array[0];
  for (int i = 1; i < size; i++) {
    if (array[i] > max) {
      max = array[i];
    }
  }
  return max;
}

// MARK: - Min
extern __fp16 nsc_min(__fp16 *array) {
  int size = sizeof(array) / sizeof(array[0]);
  __fp16 min = array[0];
  for (int i = 1; i < size; i++) {
    if (array[i] < min) {
      min = array[i];
    }
  }
  return min;
}

// MARK: - Mean
extern __fp16 nsc_mean(__fp16 *array) {
  int size = sizeof(array) / sizeof(array[0]);
  return nsc_sum(array) / size;
}

// MARK: - Add

extern void nsc_add_scalar(__fp16 *lhs, __fp16 *rhs, __fp16 *result) {
  int size = sizeof(lhs) / sizeof(lhs[0]);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] + rhs[0];
  }
}

extern void nsc_add(__fp16 *lhs, __fp16 *rhs, __fp16 *result) {
  int size = sizeof(lhs) / sizeof(lhs[0]);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] + rhs[i];
  }
}

// MARK: - Sub

extern void nsc_sub(__fp16 *lhs, __fp16 *rhs, __fp16 *result) {
  int size = sizeof(lhs) / sizeof(lhs[0]);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] - rhs[i];
  }
}

// MARK: - Mult

extern void nsc_mult_scalar(__fp16 *lhs, __fp16 *rhs, __fp16 *result) {
  int size = sizeof(lhs) / sizeof(lhs[0]);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] * rhs[0];
  }
}

extern void nsc_mult(__fp16 *lhs, __fp16 *rhs, __fp16 *result) {
  int size = sizeof(lhs) / sizeof(lhs[0]);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] * rhs[i];
  }
}

// MARK: - Div

extern void nsc_div(__fp16 *lhs, __fp16 *rhs, __fp16 *result) {
  int size = sizeof(lhs) / sizeof(lhs[0]);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] / rhs[i];
  }
}

extern void nsc_div_scalar(__fp16 *lhs, __fp16 *rhs, __fp16 *result) {
  int size = sizeof(lhs) / sizeof(lhs[0]);
  for (int i = 0; i < size; i++) {
    result[i] = lhs[i] / rhs[0];
  }
}