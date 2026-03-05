//
//  NumSwiftFlatPointer.swift
//  NumSwift
//
//  Created by William Vabrinskas on 3/4/26.
//

import Accelerate
import Foundation
import NumSwiftC

// MARK: - Pointer-Based Flat Array Operations

/// Provides pointer-accepting overloads of `NumSwiftFlat` operations.
/// These accept `UnsafePointer` / `UnsafeMutablePointer` directly,
/// avoiding intermediate `Array` or `ContiguousArray` allocations.
/// Callers are responsible for ensuring pointer validity and correct `count`.
public extension NumSwiftFlat {

  // MARK: - Element-wise Arithmetic (Float, pointer-to-pointer)

  /// `result[i] = a[i] + b[i]`
  static func add(_ a: UnsafePointer<Float>,
                  _ b: UnsafePointer<Float>,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    vDSP_vadd(a, 1, b, 1, result, 1, vDSP_Length(count))
  }

  /// `result[i] = a[i] - b[i]`
  static func sub(_ a: UnsafePointer<Float>,
                  _ b: UnsafePointer<Float>,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    // vDSP_vsub computes C = B - A, so pass (b, a) to get a - b
    vDSP_vsub(b, 1, a, 1, result, 1, vDSP_Length(count))
  }

  /// `result[i] = a[i] * b[i]`
  static func mul(_ a: UnsafePointer<Float>,
                  _ b: UnsafePointer<Float>,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    vDSP_vmul(a, 1, b, 1, result, 1, vDSP_Length(count))
  }

  /// `result[i] = a[i] / b[i]`
  static func div(_ a: UnsafePointer<Float>,
                  _ b: UnsafePointer<Float>,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    // vDSP_vdiv computes C = B / A, so pass (b, a) to get a / b
    vDSP_vdiv(b, 1, a, 1, result, 1, vDSP_Length(count))
  }

  // MARK: - Scalar Arithmetic (Float)

  /// `result[i] = a[i] + scalar`
  static func add(_ a: UnsafePointer<Float>,
                  scalar: Float,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    var s = scalar
    vDSP_vsadd(a, 1, &s, result, 1, vDSP_Length(count))
  }

  /// `result[i] = a[i] - scalar`
  static func sub(_ a: UnsafePointer<Float>,
                  scalar: Float,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    nsc_sub_scalar(a, scalar, Int32(count), result)
  }

  /// `result[i] = scalar - a[i]`
  static func sub(scalar: Float,
                  _ a: UnsafePointer<Float>,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    var s = scalar
    vDSP_vneg(a, 1, result, 1, vDSP_Length(count))
    vDSP_vsadd(result, 1, &s, result, 1, vDSP_Length(count))
  }

  /// `result[i] = a[i] * scalar`
  static func mul(_ a: UnsafePointer<Float>,
                  scalar: Float,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    var s = scalar
    vDSP_vsmul(a, 1, &s, result, 1, vDSP_Length(count))
  }

  /// `result[i] = a[i] / scalar`
  static func div(_ a: UnsafePointer<Float>,
                  scalar: Float,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    var s = scalar
    vDSP_vsdiv(a, 1, &s, result, 1, vDSP_Length(count))
  }

  /// `result[i] = scalar / a[i]`
  static func div(scalar: Float,
                  _ a: UnsafePointer<Float>,
                  result: UnsafeMutablePointer<Float>,
                  count: Int) {
    var s = scalar
    vDSP_svdiv(&s, a, 1, result, 1, vDSP_Length(count))
  }

  // MARK: - Reductions (Float)

  /// Returns the sum of all elements.
  static func sum(_ a: UnsafePointer<Float>, count: Int) -> Float {
    var result: Float = 0
    vDSP_sve(a, 1, &result, vDSP_Length(count))
    return result
  }

  /// Returns the mean of all elements.
  static func mean(_ a: UnsafePointer<Float>, count: Int) -> Float {
    var result: Float = 0
    vDSP_meanv(a, 1, &result, vDSP_Length(count))
    return result
  }

  /// Returns the sum of squares of all elements.
  static func sumOfSquares(_ a: UnsafePointer<Float>, count: Int) -> Float {
    var result: Float = 0
    vDSP_svesq(a, 1, &result, vDSP_Length(count))
    return result
  }

  // MARK: - Negation (Float)

  /// `result[i] = -a[i]`
  static func negate(_ a: UnsafePointer<Float>,
                     result: UnsafeMutablePointer<Float>,
                     count: Int) {
    vDSP_vneg(a, 1, result, 1, vDSP_Length(count))
  }

  // MARK: - Square Root (Float)

  /// `result[i] = sqrt(a[i])`
  static func sqrt(_ a: UnsafePointer<Float>,
                   result: UnsafeMutablePointer<Float>,
                   count: Int) {
    var n = Int32(count)
    vvsqrtf(result, a, &n)
  }

  // MARK: - Clip (Float)

  /// Clamp every element to `[-limit, limit]`.
  static func clip(_ a: UnsafePointer<Float>,
                   result: UnsafeMutablePointer<Float>,
                   count: Int,
                   limit: Float) {
    var low = -limit
    var high = limit
    vDSP_vclip(a, 1, &low, &high, result, 1, vDSP_Length(count))
  }

  // MARK: - Matrix Transpose (Float)

  /// Transpose a row-major matrix in-place (a -> result with swapped dimensions).
  static func transpose(_ a: UnsafePointer<Float>,
                        result: UnsafeMutablePointer<Float>,
                        rows: Int,
                        columns: Int) {
    vDSP_mtrans(a, 1, result, 1, vDSP_Length(columns), vDSP_Length(rows))
  }

  // MARK: - Matrix Multiplication (Float)

  /// `result = a[aRows x aCols] * b[bRows x bCols]`  (row-major)
  static func matmul(_ a: UnsafePointer<Float>,
                     _ b: UnsafePointer<Float>,
                     result: UnsafeMutablePointer<Float>,
                     aRows: Int,
                     aCols: Int,
                     bRows: Int,
                     bCols: Int) {
    precondition(aCols == bRows, "A columns (\(aCols)) must equal B rows (\(bRows))")
    nsc_matmul1d(.init(rows: Int32(aRows), columns: Int32(aCols), depth: 1),
                 .init(rows: Int32(bRows), columns: Int32(bCols), depth: 1),
                 a, b, result)
  }

  // MARK: - Convolution (Float)

  /// 2D convolution on flat row-major data, writing into caller-provided result buffer.
  static func conv2d(signal: UnsafePointer<Float>,
                     filter: UnsafePointer<Float>,
                     result: UnsafeMutablePointer<Float>,
                     strides: (Int, Int) = (1, 1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int)) {
    let nscPadding: NSC_Padding = padding == .same ? same : valid
    nsc_conv1d(signal, filter, result,
               .init(rows: Int32(strides.0), columns: Int32(strides.1), depth: 1),
               nscPadding,
               .init(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns), depth: 1),
               .init(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns), depth: 1))
  }

  /// Transposed 2D convolution on flat row-major data, writing into caller-provided result buffer.
  static func transConv2d(signal: UnsafePointer<Float>,
                          filter: UnsafePointer<Float>,
                          result: UnsafeMutablePointer<Float>,
                          strides: (Int, Int) = (1, 1),
                          padding: NumSwift.ConvPadding = .valid,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int)) {
    let nscPadding: NSC_Padding = padding == .same ? same : valid
    nsc_transConv1d(signal, filter, result,
                    .init(rows: Int32(strides.0), columns: Int32(strides.1), depth: 1),
                    nscPadding,
                    .init(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns), depth: 1),
                    .init(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns), depth: 1))
  }

  // MARK: - Zero-pad and Stride-pad (Float, pointer-to-pointer)

  /// Zero-pad flat row-major 2D data, writing into caller-provided result buffer.
  /// Caller must ensure result has capacity (inputRows + top + bottom) * (inputCols + left + right).
  static func zeroPad1D(signal: UnsafePointer<Float>,
                        result: UnsafeMutablePointer<Float>,
                        padding: NumSwiftPadding,
                        inputSize: (rows: Int, columns: Int)) {
    guard padding.right > 0 || padding.left > 0 || padding.top > 0 || padding.bottom > 0 else {
      result.update(from: signal, count: inputSize.rows * inputSize.columns)
      return
    }
    nsc_specific_zero_pad_1d(signal, result,
                            .init(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns), depth: 1),
                            Int32(padding.top), Int32(padding.bottom),
                            Int32(padding.left), Int32(padding.right))
  }

  /// Stride-pad flat row-major 2D data, writing into caller-provided result buffer.
  /// Caller must ensure result has capacity for the padded size.
  static func stridePad1D(signal: UnsafePointer<Float>,
                          result: UnsafeMutablePointer<Float>,
                          strides: (rows: Int, columns: Int),
                          signalSize: (rows: Int, columns: Int)) {
    guard strides.rows - 1 > 0 || strides.columns - 1 > 0 else {
      result.update(from: signal, count: signalSize.rows * signalSize.columns)
      return
    }
    nsc_stride_pad(signal, result,
                   .init(rows: Int32(signalSize.rows), columns: Int32(signalSize.columns), depth: 1),
                   .init(rows: Int32(strides.rows), columns: Int32(strides.columns), depth: 1))
  }

  /// Flip 180 degrees (reverse rows, reverse each row's columns), writing into caller-provided result buffer.
  static func flip180(signal: UnsafePointer<Float>,
                      result: UnsafeMutablePointer<Float>,
                      rows: Int,
                      columns: Int) {
    for r in 0..<rows {
      let srcRow = rows - 1 - r
      let srcStart = srcRow * columns
      let dstStart = r * columns
      for c in 0..<columns {
        result[dstStart + c] = signal[srcStart + (columns - 1 - c)]
      }
    }
  }
}

// MARK: - Float16 Pointer APIs

#if arch(arm64)
public extension NumSwiftFlat {

  // MARK: - Element-wise Arithmetic (Float16, pointer-to-pointer)

  /// `result[i] = a[i] + b[i]`
  static func add(_ a: UnsafePointer<Float16>,
                  _ b: UnsafePointer<Float16>,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_add(a, b, Int32(count), result)
  }

  /// `result[i] = a[i] - b[i]`
  static func sub(_ a: UnsafePointer<Float16>,
                  _ b: UnsafePointer<Float16>,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_sub(a, b, Int32(count), result)
  }

  /// `result[i] = a[i] * b[i]`
  static func mul(_ a: UnsafePointer<Float16>,
                  _ b: UnsafePointer<Float16>,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_mult(a, b, Int32(count), result)
  }

  /// `result[i] = a[i] / b[i]`
  static func div(_ a: UnsafePointer<Float16>,
                  _ b: UnsafePointer<Float16>,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_div(a, b, Int32(count), result)
  }

  // MARK: - Scalar Arithmetic (Float16)

  /// `result[i] = a[i] + scalar`
  static func add(_ a: UnsafePointer<Float16>,
                  scalar: Float16,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_add_scalar(scalar, a, Int32(count), result)
  }

  /// `result[i] = a[i] - scalar`
  static func sub(_ a: UnsafePointer<Float16>,
                  scalar: Float16,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_sub_scalar_f16(a, scalar, Int32(count), result)
  }

  /// `result[i] = scalar - a[i]`
  static func sub(scalar: Float16,
                  _ a: UnsafePointer<Float16>,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_sub_scalar_array_f16(scalar, a, Int32(count), result)
  }

  /// `result[i] = a[i] * scalar`
  static func mul(_ a: UnsafePointer<Float16>,
                  scalar: Float16,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_mult_scalar(scalar, a, Int32(count), result)
  }

  /// `result[i] = a[i] / scalar`
  static func div(_ a: UnsafePointer<Float16>,
                  scalar: Float16,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_div_array_scalar(a, scalar, Int32(count), result)
  }

  /// `result[i] = scalar / a[i]`
  static func div(scalar: Float16,
                  _ a: UnsafePointer<Float16>,
                  result: UnsafeMutablePointer<Float16>,
                  count: Int) {
    nsc_div_scalar_array(scalar, a, Int32(count), result)
  }

  // MARK: - Reductions (Float16)

  /// Returns the sum of all elements.
  static func sum(_ a: UnsafePointer<Float16>, count: Int) -> Float16 {
    nsc_sum(a, Int32(count))
  }

  /// Returns the mean of all elements.
  static func mean(_ a: UnsafePointer<Float16>, count: Int) -> Float16 {
    nsc_mean(a, Int32(count))
  }

  /// Returns the sum of squares of all elements.
  static func sumOfSquares(_ a: UnsafePointer<Float16>, count: Int) -> Float16 {
    nsc_sum_of_squares(a, Int32(count))
  }

  // MARK: - Negation (Float16)

  /// `result[i] = -a[i]`
  static func negate(_ a: UnsafePointer<Float16>,
                     result: UnsafeMutablePointer<Float16>,
                     count: Int) {
    for i in 0..<count { result[i] = -a[i] }
  }

  // MARK: - Square Root (Float16)

  /// `result[i] = sqrt(a[i])`
  static func sqrt(_ a: UnsafePointer<Float16>,
                   result: UnsafeMutablePointer<Float16>,
                   count: Int) {
    for i in 0..<count { result[i] = Float16(Foundation.sqrt(Float(a[i]))) }
  }

  // MARK: - Clip (Float16)

  /// Clamp every element to `[-limit, limit]`.
  static func clip(_ a: UnsafePointer<Float16>,
                   result: UnsafeMutablePointer<Float16>,
                   count: Int,
                   limit: Float16) {
    for i in 0..<count { result[i] = Swift.max(-limit, Swift.min(limit, a[i])) }
  }

  // MARK: - Matrix Transpose (Float16)

  /// Transpose a row-major matrix (a -> result with swapped dimensions).
  static func transpose(_ a: UnsafePointer<Float16>,
                        result: UnsafeMutablePointer<Float16>,
                        rows: Int,
                        columns: Int) {
    for r in 0..<rows {
      for c in 0..<columns {
        result[c * rows + r] = a[r * columns + c]
      }
    }
  }

  // MARK: - Matrix Multiplication (Float16)

  /// `result = a[aRows x aCols] * b[bRows x bCols]`  (row-major)
  static func matmul(_ a: UnsafePointer<Float16>,
                     _ b: UnsafePointer<Float16>,
                     result: UnsafeMutablePointer<Float16>,
                     aRows: Int,
                     aCols: Int,
                     bRows: Int,
                     bCols: Int) {
    precondition(aCols == bRows, "A columns (\(aCols)) must equal B rows (\(bRows))")
    nsc_matmul1d_16(.init(rows: Int32(aRows), columns: Int32(aCols), depth: 1),
                    .init(rows: Int32(bRows), columns: Int32(bCols), depth: 1),
                    a, b, result)
  }

  // MARK: - Convolution (Float16)

  /// 2D convolution on flat row-major data, writing into caller-provided result buffer.
  static func conv2d(signal: UnsafePointer<Float16>,
                     filter: UnsafePointer<Float16>,
                     result: UnsafeMutablePointer<Float16>,
                     strides: (Int, Int) = (1, 1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int)) {
    let nscPadding: NSC_Padding = padding == .same ? same : valid
    nsc_conv1d_f16(signal, filter, result,
                   .init(rows: Int32(strides.0), columns: Int32(strides.1), depth: 1),
                   nscPadding,
                   .init(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns), depth: 1),
                   .init(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns), depth: 1))
  }

  /// Transposed 2D convolution on flat row-major data, writing into caller-provided result buffer.
  static func transConv2d(signal: UnsafePointer<Float16>,
                          filter: UnsafePointer<Float16>,
                          result: UnsafeMutablePointer<Float16>,
                          strides: (Int, Int) = (1, 1),
                          padding: NumSwift.ConvPadding = .valid,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int)) {
    let nscPadding: NSC_Padding = padding == .same ? same : valid
    nsc_transConv1d_f16(signal, filter, result,
                        .init(rows: Int32(strides.0), columns: Int32(strides.1), depth: 1),
                        nscPadding,
                        .init(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns), depth: 1),
                        .init(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns), depth: 1))
  }

  // MARK: - Zero-pad, Stride-pad, Flip180 (Float16, pointer-to-pointer)

  /// Zero-pad flat row-major 2D data, writing into caller-provided result buffer.
  static func zeroPad1D(signal: UnsafePointer<Float16>,
                        result: UnsafeMutablePointer<Float16>,
                        padding: NumSwiftPadding,
                        inputSize: (rows: Int, columns: Int)) {
    guard padding.right > 0 || padding.left > 0 || padding.top > 0 || padding.bottom > 0 else {
      result.update(from: signal, count: inputSize.rows * inputSize.columns)
      return
    }
    let outRows = inputSize.rows + padding.top + padding.bottom
    let outCols = inputSize.columns + padding.left + padding.right
    for i in 0..<(outRows * outCols) { result[i] = 0 }
    for r in 0..<inputSize.rows {
      let srcStart = r * inputSize.columns
      let dstStart = (r + padding.top) * outCols + padding.left
      for c in 0..<inputSize.columns {
        result[dstStart + c] = signal[srcStart + c]
      }
    }
  }

  /// Stride-pad flat row-major 2D data, writing into caller-provided result buffer.
  static func stridePad1D(signal: UnsafePointer<Float16>,
                          result: UnsafeMutablePointer<Float16>,
                          strides: (rows: Int, columns: Int),
                          signalSize: (rows: Int, columns: Int)) {
    guard strides.rows - 1 > 0 || strides.columns - 1 > 0 else {
      result.update(from: signal, count: signalSize.rows * signalSize.columns)
      return
    }
    nsc_stride_pad_f16(signal, result,
                       .init(rows: Int32(signalSize.rows), columns: Int32(signalSize.columns), depth: 1),
                       .init(rows: Int32(strides.rows), columns: Int32(strides.columns), depth: 1))
  }

  /// Flip 180 degrees, writing into caller-provided result buffer.
  static func flip180(signal: UnsafePointer<Float16>,
                      result: UnsafeMutablePointer<Float16>,
                      rows: Int,
                      columns: Int) {
    for r in 0..<rows {
      let srcRow = rows - 1 - r
      let srcStart = srcRow * columns
      let dstStart = r * columns
      for c in 0..<columns {
        result[dstStart + c] = signal[srcStart + (columns - 1 - c)]
      }
    }
  }
}
#endif
