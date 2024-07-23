#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
  static func sum(_ array: [Scalar]) -> Scalar
  static func sumOfSquares(_ array: [Scalar]) -> Scalar
  static func indexOfMin(_ array: [Scalar]) -> (UInt, Scalar)
  static func indexOfMax(_ array: [Scalar]) -> (UInt, Scalar)
  static func max(_ array: [Scalar]) -> Scalar
  static func min(_ array: [Scalar]) -> Scalar
  static func mean(_ array: [Scalar]) -> Scalar

  static func add(lhs: Scalar, rhs: [Scalar]) -> [Scalar]
  static func add(lhs: [Scalar], rhs: [Scalar]) -> [Scalar]
  
  static func sub(lhs: [Scalar], rhs: [Scalar]) -> [Scalar]
  
  static func mult(lhs: [Scalar], rhs: Scalar) -> [Scalar]
  static func mult(lhs: Scalar, rhs: [Scalar]) -> [Scalar]
  static func mult(lhs: [Scalar], rhs: [Scalar]) -> [Scalar]
  
  static func div(lhs: [Scalar], rhs: [Scalar]) -> [Scalar]
  static func div(lhs: [Scalar], rhs: Scalar) -> [Scalar]
  static func div(lhs: Scalar, rhs: [Scalar]) -> [Scalar]
*/

typedef struct {
  __fp16 value;
  int index;
} NSC_IndexedValue;

extern NSC_IndexedValue nsc_index_of_min(const __fp16 array[]);
extern NSC_IndexedValue nsc_index_of_max(const __fp16 array[]);
extern __fp16 nsc_max(const __fp16 array[]);
extern __fp16 nsc_min(const __fp16 array[]);
extern __fp16 nsc_mean(const __fp16 array[]);
extern __fp16 nsc_sum(const __fp16 array[]);
extern __fp16 nsc_sum_of_squares(const __fp16 array[]);

extern void nsc_add_scalar(const __fp16 lhs, const __fp16 rhs[], __fp16 *result);
extern void nsc_add(const __fp16 lhs[], const __fp16 rhs[], __fp16 *result);


extern void nsc_sub(const __fp16 lhs[], const __fp16 rhs[], __fp16 *result);


extern void nsc_mult_scalar(const __fp16 lhs, const __fp16 rhs[], __fp16 *result);
extern void nsc_mult(const __fp16 lhs[], const __fp16 rhs[], __fp16 *result);


extern void nsc_div_scalar_array(const __fp16 lhs, const __fp16 rhs[], __fp16 *result);

extern void nsc_div_array_scalar(const __fp16 lhs[], const __fp16 rhs, __fp16 *result);

extern void nsc_div(const __fp16 lhs[], const __fp16 rhs[], __fp16 *result);



