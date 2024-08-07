#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <numswiftc_base.h>

typedef struct {
  int rows;
  int columns;
} NSC_Size;

typedef enum {
  valid = 0,
  same = 1
} NSC_Padding;
//

extern void nsc_transpose_2d(float *const *input,
                             float **result,
                             NSC_Size input_size);

extern void nsc_transpose_2d_16(__fp16 *const *input,
                                __fp16 **result,
                                NSC_Size input_size);

extern void nsc_stride_pad_2D_f16(__fp16 *const *input,
                                  __fp16 **result,
                                  NSC_Size input_size,
                                  NSC_Size stride_size);

extern void nsc_stride_pad_2D(float *const *input,
                              float **result,
                              NSC_Size input_size,
                              NSC_Size stride_size);

extern void nsc_specific_zero_pad_2d(float *const *input,
                                     float **result,
                                     NSC_Size input_size,
                                     int paddingTop,
                                     int paddingBottom,
                                     int paddingLeft,
                                     int paddingRight);

extern void nsc_conv2d(float *const *signal,
                       float *const *filter,
                       float **result,
                       NSC_Size stride,
                       NSC_Padding padding,
                       NSC_Size filter_size,
                       NSC_Size input_size);

extern void nsc_conv2d_f16(__fp16 *const *signal,
                       __fp16 *const *filter,
                       __fp16 **result,
                       NSC_Size stride,
                       NSC_Padding padding,
                       NSC_Size filter_size,
                       NSC_Size input_size);

extern void nsc_matmul(NSC_Size a_size,
                       NSC_Size b_size,
                       float *const *a,
                       float *const *b,
                       float **result);

extern void nsc_matmul_16(NSC_Size a_size,
                          NSC_Size b_size,
                          __fp16 *const *a,
                          __fp16 *const *b,
                          __fp16 **result);

extern void nsc_flatten2d(NSC_Size input_size,
                          float *const *input,
                          float *result);

extern void nsc_flatten2d_16(NSC_Size input_size,
                          __fp16 *const *input,
                          __fp16 *result);

extern void random_array(const int size, double *result);

extern double nsc_perlin_noise(const double x,
                               const double y,
                               const double z,
                               const double amplitude,
                               const int octaves,
                               const int size,
                               const double* perlin_seed);

extern void nsc_stride_pad(const float input[],
                           float *result,
                           NSC_Size input_size,
                           NSC_Size stride_size);

extern void nsc_stride_pad_f16(const __fp16 input[],
                           __fp16 *result,
                           NSC_Size input_size,
                           NSC_Size stride_size);

extern void nsc_conv1d(const float signal[],
                       const float filter[],
                       float *result,
                       NSC_Size stride,
                       NSC_Padding padding,
                       NSC_Size filter_size,
                       NSC_Size input_size);

extern void nsc_conv1d_f16(const __fp16 signal[],
                       const __fp16 filter[],
                       __fp16 *result,
                       NSC_Size stride,
                       NSC_Padding padding,
                       NSC_Size filter_size,
                       NSC_Size input_size);

extern void nsc_array_mod(const int rows, const int columns, float result[][columns]);

extern void nsc_transConv2d_f16(__fp16 *const *signal,
                            __fp16 *const *filter,
                            __fp16 **result,
                            NSC_Size stride,
                            NSC_Padding padding,
                            NSC_Size filter_size,
                            NSC_Size input_size);

extern void nsc_transConv1d_f16(const __fp16 signal[],
                            const __fp16 filter[],
                            __fp16 *result,
                            NSC_Size stride,
                            NSC_Padding padding,
                            NSC_Size filter_size,
                            NSC_Size input_size);

extern void nsc_transConv2d(float *const *signal,
                            float *const *filter,
                            float **result,
                            NSC_Size stride,
                            NSC_Padding padding,
                            NSC_Size filter_size,
                            NSC_Size input_size);

extern void nsc_transConv1d(const float signal[],
                            const float filter[],
                            float *result,
                            NSC_Size stride,
                            NSC_Padding padding,
                            NSC_Size filter_size,
                            NSC_Size input_size);

extern void nsc_zero_pad(const float input[],
                         float *result,
                         NSC_Size filter_size,
                         NSC_Size input_size,
                         NSC_Size stride);

extern void nsc_zero_pad_f16(const __fp16 input[],
                         __fp16 *result,
                         NSC_Size filter_size,
                         NSC_Size input_size,
                         NSC_Size stride);

extern void nsc_specific_zero_pad(const float input[],
                                  float *result,
                                  NSC_Size input_size,
                                  int paddingTop,
                                  int paddingBottom,
                                  int paddingLeft,
                                  int paddingRight);

extern void nsc_specific_zero_pad_2d_f16(__fp16 *const *input,
                                     __fp16 **result,
                                     NSC_Size input_size,
                                     int paddingTop,
                                     int paddingBottom,
                                     int paddingLeft,
                                     int paddingRight);

extern void nsc_padding_calculation(NSC_Size stride,
                                    NSC_Padding padding,
                                    NSC_Size filter_size,
                                    NSC_Size input_size,
                                    int *paddingTop,
                                    int *paddingBottom,
                                    int *paddingLeft,
                                    int *paddingRight);
