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

extern NSC_IndexedValue nsc_index_of_min(__fp16 *array);
extern NSC_IndexedValue nsc_index_of_max(__fp16 *array);
extern __fp16 nsc_max(__fp16 *array);
extern __fp16 nsc_min(__fp16 *array);
extern __fp16 nsc_mean(__fp16 *array);
extern __fp16 nsc_sum(__fp16 *array);
extern __fp16 nsc_sum_of_squares(__fp16 *array);

extern void nsc_add_scalar(__fp16 *lhs, __fp16 *rhs, __fp16 *result);
extern void nsc_add(__fp16 *lhs, __fp16 *rhs, __fp16 *result);

extern void nsc_sub(__fp16 *lhs, __fp16 *rhs, __fp16 *result);

extern void nsc_mult_scalar(__fp16 *lhs, __fp16 *rhs, __fp16 *result);
extern void nsc_mult(__fp16 *lhs, __fp16 *rhs, __fp16 *result);

extern void nsc_div_scalar(__fp16 *lhs, __fp16 *rhs, __fp16 *result);
extern void nsc_div(__fp16 *lhs, __fp16 *rhs, __fp16 *result);



