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
  _Float16 value;
  int index;
} NSC_IndexedValue;

typedef struct {
  int rows;
  int columns;
  int depth;
} NSC_Size;

extern NSC_IndexedValue nsc_index_of_min(const _Float16 array[]);
extern NSC_IndexedValue nsc_index_of_max(const _Float16 array[]);
extern _Float16 nsc_max(const _Float16 array[]);
extern _Float16 nsc_min(const _Float16 array[]);
extern _Float16 nsc_mean(const _Float16 array[]);
extern _Float16 nsc_sum(const _Float16 array[]);
extern _Float16 nsc_sum_of_squares(const _Float16 array[]);

extern void nsc_add_scalar(const _Float16 lhs, const _Float16 rhs[], _Float16 *result);
extern void nsc_add(const _Float16 lhs[], const _Float16 rhs[], _Float16 *result);

extern void nsc_mult2d_f16(NSC_Size size,
                           _Float16 *const *a,
                           _Float16 *const *b,
                           _Float16 **result);



extern void nsc_sub(const _Float16 lhs[], const _Float16 rhs[], _Float16 *result);


extern void nsc_mult_scalar(const _Float16 lhs, const _Float16 rhs[], _Float16 *result);
extern void nsc_mult(const _Float16 lhs[], const _Float16 rhs[], _Float16 *result);


extern void nsc_div_scalar_array(const _Float16 lhs, const _Float16 rhs[], _Float16 *result);

extern void nsc_div_array_scalar(const _Float16 lhs[], const _Float16 rhs, _Float16 *result);

extern void nsc_div(const _Float16 lhs[], const _Float16 rhs[], _Float16 *result);


// Arithmetic functions

/// Float32 2D
extern void nsc_add2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                      float **result);

extern void nsc_sub2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                      float **result);

extern void nsc_divide2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                         float **result);

extern void nsc_mult2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                       float **result);

/// Float16 2D
extern void nsc_add2d(NSC_Size size,
                      float *const *a,
                      float *const *b,
                      float **result);

extern void nsc_add2d_f16(NSC_Size size,
                          _Float16 *const *a,
                          _Float16 *const *b,
                          _Float16 **result);

extern void nsc_sub2d_f16(NSC_Size size,
                          _Float16 *const *a,
                          _Float16 *const *b,
                          _Float16 **result);

extern void nsc_divide2d_f16(NSC_Size size,
                             _Float16 *const *a,
                             _Float16 *const *b,
                             _Float16 **result);

// 2D arithmetic with scalar operations
/// Float32 2D with scalar
extern void nsc_add2d_scalar(NSC_Size size,
                             float *const *a,
                             float scalar,
                             float **result);

extern void nsc_add2d_array_scalar(NSC_Size size,
                                   float *const *a,
                                   const float *scalar_array,
                                   float **result);

extern void nsc_sub2d_scalar(NSC_Size size,
                             float *const *a,
                             float scalar,
                             float **result);

extern void nsc_sub2d_array_scalar(NSC_Size size,
                                   float *const *a,
                                   const float *scalar_array,
                                   float **result);

extern void nsc_mult2d_scalar(NSC_Size size,
                              float *const *a,
                              float scalar,
                              float **result);

extern void nsc_mult2d_array_scalar(NSC_Size size,
                                    float *const *a,
                                    const float *scalar_array,
                                    float **result);

extern void nsc_divide2d_scalar(NSC_Size size,
                                float *const *a,
                                float scalar,
                                float **result);

extern void nsc_divide2d_array_scalar(NSC_Size size,
                                      float *const *a,
                                      const float *scalar_array,
                                      float **result);

/// Float16 2D with scalar
extern void nsc_add2d_scalar_f16(NSC_Size size,
                                 _Float16 *const *a,
                                 _Float16 scalar,
                                 _Float16 **result);

extern void nsc_add2d_array_scalar_f16(NSC_Size size,
                                       _Float16 *const *a,
                                       const _Float16 *scalar_array,
                                       _Float16 **result);

extern void nsc_sub2d_scalar_f16(NSC_Size size,
                                 _Float16 *const *a,
                                 _Float16 scalar,
                                 _Float16 **result);

extern void nsc_sub2d_array_scalar_f16(NSC_Size size,
                                       _Float16 *const *a,
                                       const _Float16 *scalar_array,
                                       _Float16 **result);

extern void nsc_mult2d_scalar_f16(NSC_Size size,
                                  _Float16 *const *a,
                                  _Float16 scalar,
                                  _Float16 **result);

extern void nsc_mult2d_array_scalar_f16(NSC_Size size,
                                        _Float16 *const *a,
                                        const _Float16 *scalar_array,
                                        _Float16 **result);

extern void nsc_divide2d_scalar_f16(NSC_Size size,
                                    _Float16 *const *a,
                                    _Float16 scalar,
                                    _Float16 **result);

extern void nsc_divide2d_array_scalar_f16(NSC_Size size,
                                          _Float16 *const *a,
                                          const _Float16 *scalar_array,
                                          _Float16 **result);
