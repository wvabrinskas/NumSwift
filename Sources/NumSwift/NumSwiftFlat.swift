//
//  NumSwiftFlat.swift
//  NumSwift
//
//  Created by William Vabrinskas on 2/6/26.
//

import Accelerate
import Foundation

// MARK: - Flat Array Operations (Accelerate-backed)

/// Provides high-performance flat-array operations on `[Float]` using Apple's Accelerate framework.
/// These avoid the overhead of nested `[[[Float]]]` arrays and enable direct SIMD/vectorized computation.
public enum NumSwiftFlat  {
  
  // MARK: - Element-wise Arithmetic (array + array)
  
  /// Element-wise addition using vDSP_vadd
  /// we dont need count check here because if two arrays are different sizes we'll just add the number of elements in A
  public static func add(_ a: [Float], _ b: [Float]) -> [Float] {
    let count = a.count
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          vDSP_vadd(aBuf.baseAddress!, 1, bBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  /// Element-wise subtraction (a - b) using vDSP_vsub
  /// we dont need count check here because if two arrays are different sizes we'll just sub the number of elements in A
  public static func subtract(_ a: [Float], _ b: [Float]) -> [Float] {
    let count = a.count
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          // vDSP_vsub computes C = B - A (note reversed order)
          vDSP_vsub(bBuf.baseAddress!, 1, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  /// Element-wise multiplication using vDSP_vmul
  /// we dont need count check here because if two arrays are different sizes we'll just multiply the number of elements in A
  public static func multiply(_ a: [Float], _ b: [Float]) -> [Float] {
    let count = a.count
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          vDSP_vmul(aBuf.baseAddress!, 1, bBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  /// Element-wise division (a / b) using vDSP_vdiv
  /// we dont need count check here because if two arrays are different sizes we'll just divide the number of elements in A
  public static func divide(_ a: [Float], _ b: [Float]) -> [Float] {
    let count = a.count
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          // vDSP_vdiv computes C = B / A (note reversed order)
          vDSP_vdiv(bBuf.baseAddress!, 1, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  // MARK: - Scalar Arithmetic (array op scalar)
  
  /// Add scalar to every element using vDSP_vsadd
  public static func add(_ a: [Float], scalar: Float) -> [Float] {
    let count = a.count
    var s = scalar
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsadd(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  /// Multiply every element by scalar using vDSP_vsmul
  public static func multiply(_ a: [Float], scalar: Float) -> [Float] {
    let count = a.count
    var s = scalar
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsmul(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  /// Divide every element by scalar using vDSP_vsdiv
  public static func divide(_ a: [Float], scalar: Float) -> [Float] {
    let count = a.count
    var s = scalar
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsdiv(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  /// Subtract scalar from every element: result = a - scalar
  public static func subtract(_ a: [Float], scalar: Float) -> [Float] {
    return add(a, scalar: -scalar)
  }
  
  /// Scalar minus every element: result = scalar - a. Uses vDSP_vneg + vDSP_vsadd.
  public static func subtract(scalar: Float, _ a: [Float]) -> [Float] {
    let count = a.count
    var c = [Float](repeating: 0, count: count)
    var s = scalar
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        // negate a
        vDSP_vneg(aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        // add scalar
        vDSP_vsadd(cBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  /// Scalar divided by every element: result = scalar / a[i]
  public static func divide(scalar: Float, _ a: [Float]) -> [Float] {
    let count = a.count
    var s = scalar
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_svdiv(&s, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  // MARK: - Negation
  
  /// Negate every element using vDSP_vneg
  public static func negate(_ a: [Float]) -> [Float] {
    let count = a.count
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vneg(aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  // MARK: - Reductions
  
  /// Sum of all elements using vDSP_sve
  public static func sum(_ a: [Float]) -> Float {
    var result: Float = 0
    a.withUnsafeBufferPointer { aBuf in
      vDSP_sve(aBuf.baseAddress!, 1, &result, vDSP_Length(a.count))
    }
    return result
  }
  
  /// Sum of squares using vDSP_svesq
  public static func sumOfSquares(_ a: [Float]) -> Float {
    var result: Float = 0
    a.withUnsafeBufferPointer { aBuf in
      vDSP_svesq(aBuf.baseAddress!, 1, &result, vDSP_Length(a.count))
    }
    return result
  }
  
  /// Mean of all elements using vDSP_meanv
  public static func mean(_ a: [Float]) -> Float {
    var result: Float = 0
    a.withUnsafeBufferPointer { aBuf in
      vDSP_meanv(aBuf.baseAddress!, 1, &result, vDSP_Length(a.count))
    }
    return result
  }
  
  // MARK: - Square Root
  
  /// Element-wise square root using vForce
  public static func sqrt(_ a: [Float]) -> [Float] {
    let count = a.count
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        var n = Int32(count)
        vvsqrtf(cBuf.baseAddress!, aBuf.baseAddress!, &n)
      }
    }
    return c
  }
  
  // MARK: - Matrix Transpose
  
  /// Transpose a single 2D matrix stored as flat row-major using vDSP_mtrans
  /// rows: is the number of rows in the input matrix
  /// columns: is the number of columns in the input matrix
  public static func transpose(_ a: [Float], rows: Int, columns: Int) -> [Float] {
    var c = [Float](repeating: 0, count: rows * columns)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_mtrans(aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(columns), vDSP_Length(rows))
      }
    }
    return c
  }
  
  // MARK: - Matrix Multiplication
  
  /// Performs matrix multiplication on flat [Float] using Accelerate's vDSP_mmul.
  public static func matmul(_ a: [Float],
                            _ b: [Float],
                            aRows: Int,
                            aCols: Int,
                            bRows: Int,
                            bCols: Int) -> [Float] {
    precondition(aCols == bRows, "A columns (\(aCols)) must equal B rows (\(bRows))")
    
    let out = NumSwiftC.matmul1d(Array(a), b: Array(b),
                                 aSize: (aRows, aCols),
                                 bSize: (bRows, bCols))
    
    return out
  }
  
  // MARK: - Clip
  
  /// Clip all values to [-limit, limit] using vDSP_vclip
  public static func clip(_ a: [Float], to limit: Float) -> [Float] {
    let count = a.count
    var low = -limit
    var high = limit
    var c = [Float](repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vclip(aBuf.baseAddress!, 1, &low, &high, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  // MARK: - Float16 Support
  
  // MARK: - Convolution (flat row-major)
  
  /// 2D convolution on flat row-major arrays via the C `nsc_conv1d` function.
  public static func conv2d(signal: [Float],
                            filter: [Float],
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> [Float] {
    let result = NumSwiftC.conv1d(signal: Array(signal),
                                  filter: Array(filter),
                                  strides: strides,
                                  padding: padding,
                                  filterSize: filterSize,
                                  inputSize: inputSize)
    return result
  }
  
  /// Transposed 2D convolution on flat row-major arrays via the C `nsc_transConv1d` function.
  public static func transConv2d(signal: [Float],
                                 filter: [Float],
                                 strides: (Int, Int) = (1,1),
                                 padding: NumSwift.ConvPadding = .valid,
                                 filterSize: (rows: Int, columns: Int),
                                 inputSize: (rows: Int, columns: Int)) -> [Float] {
    let result = NumSwiftC.transConv1d(signal: Array(signal),
                                       filter: Array(filter),
                                       strides: strides,
                                       padding: padding,
                                       filterSize: filterSize,
                                       inputSize: inputSize)
    return result
  }
  
  /// Zero-pad a flat row-major 2D array with explicit padding values.
  public static func zeroPad(signal: [Float],
                             padding: NumSwiftPadding,
                             inputSize: (rows: Int, columns: Int)) -> [Float] {
    guard padding.right > 0 || padding.left > 0 || padding.top > 0 || padding.bottom > 0 else {
      return signal
    }
    
    let expectedRows = inputSize.rows + padding.top + padding.bottom
    let expectedColumns = inputSize.columns + padding.left + padding.right
    var result = [Float](repeating: 0, count: expectedRows * expectedColumns)
    
    // Copy original data into the padded result at the correct offsets
    for r in 0..<inputSize.rows {
      for c in 0..<inputSize.columns {
        result[(r + padding.top) * expectedColumns + (c + padding.left)] = signal[r * inputSize.columns + c]
      }
    }
    
    return result
  }
  
  /// Zero-pad a flat row-major 2D array computed from filter/input sizes.
  public static func zeroPad(signal: [Float],
                             filterSize: (rows: Int, columns: Int),
                             inputSize: (rows: Int, columns: Int),
                             stride: (Int, Int) = (1,1)) -> [Float] {
    let result = NumSwiftC.zeroPad(signal: Array(signal),
                                   filterSize: filterSize,
                                   inputSize: inputSize,
                                   stride: stride)
    return result
  }
  
  /// Stride-pad a flat row-major 2D array (inserts zeros between elements).
  /// Places original values at stride intervals in a larger zero-filled output.
  public static func stridePad(signal: [Float],
                               strides: (rows: Int, columns: Int),
                               inputSize: (rows: Int, columns: Int)) -> [Float] {
    guard strides.rows - 1 > 0 || strides.columns - 1 > 0 else { return signal }
    
    let newRows = inputSize.rows + ((strides.rows - 1) * (inputSize.rows - 1))
    let newColumns = inputSize.columns + ((strides.columns - 1) * (inputSize.columns - 1))
    var result = [Float](repeating: 0, count: newRows * newColumns)
    
    for r in 0..<inputSize.rows {
      for c in 0..<inputSize.columns {
        result[r * strides.rows * newColumns + c * strides.columns] = signal[r * inputSize.columns + c]
      }
    }
    
    return result
  }
  
  /// Returns the shape of a stride-padded result without actually padding.
  public static func stridePadShape(inputSize: (rows: Int, columns: Int),
                                    strides: (rows: Int, columns: Int)) -> (rows: Int, columns: Int) {
    let newRows = inputSize.rows + ((strides.rows - 1) * (inputSize.rows - 1))
    let newColumns = inputSize.columns + ((strides.columns - 1) * (inputSize.columns - 1))
    return (newRows, newColumns)
  }
  
  /// Flip a flat row-major 2D matrix 180 degrees (reverse all elements, then reverse each row).
  /// Equivalent to reversing the entire array then reversing each row segment.
  public static func flip180(_ a: [Float], rows: Int, columns: Int) -> [Float] {
    var result = [Float](repeating: 0, count: a.count)
    // flip180 = reverse rows, then reverse each row's columns
    for r in 0..<rows {
      let srcRow = rows - 1 - r
      let srcStart = srcRow * columns
      let dstStart = r * columns
      for c in 0..<columns {
        result[dstStart + c] = a[srcStart + (columns - 1 - c)]
      }
    }
    return result
  }
  
  /// Padding calculation utility (delegates to NumSwiftC).
  public static func paddingCalculation(strides: (Int, Int) = (1,1),
                                        padding: NumSwift.ConvPadding = .valid,
                                        filterSize: (rows: Int, columns: Int),
                                        inputSize: (rows: Int, columns: Int)) -> (top: Int, bottom: Int, left: Int, right: Int) {
    return NumSwiftC.paddingCalculation(strides: strides, padding: padding, filterSize: filterSize, inputSize: inputSize)
  }
  
  #if arch(arm64)
  // Float16 versions use manual loops (no Accelerate support for Float16).
  // The compiler can auto-vectorize these for ARM NEON.
  
  public static func add(_ a: [Float16], _ b: [Float16]) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] + b[i] }
    return c
  }
  
  public static func subtract(_ a: [Float16], _ b: [Float16]) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] - b[i] }
    return c
  }
  
  public static func multiply(_ a: [Float16], _ b: [Float16]) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] * b[i] }
    return c
  }
  
  public static func divide(_ a: [Float16], _ b: [Float16]) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] / b[i] }
    return c
  }
  
  public static func add(_ a: [Float16], scalar: Float16) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] + scalar }
    return c
  }
  
  public static func multiply(_ a: [Float16], scalar: Float16) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] * scalar }
    return c
  }
  
  public static func divide(_ a: [Float16], scalar: Float16) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] / scalar }
    return c
  }
  
  public static func subtract(_ a: [Float16], scalar: Float16) -> [Float16] {
    return add(a, scalar: -scalar)
  }
  
  public static func subtract(scalar: Float16, _ a: [Float16]) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = scalar - a[i] }
    return c
  }
  
  public static func divide(scalar: Float16, _ a: [Float16]) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = scalar / a[i] }
    return c
  }
  
  public static func negate(_ a: [Float16]) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = -a[i] }
    return c
  }
  
  public static func sum(_ a: [Float16]) -> Float16 {
    var result: Float16 = 0
    for i in 0..<a.count { result += a[i] }
    return result
  }
  
  public static func sumOfSquares(_ a: [Float16]) -> Float16 {
    var result: Float16 = 0
    for i in 0..<a.count { result += a[i] * a[i] }
    return result
  }
  
  public static func mean(_ a: [Float16]) -> Float16 {
    guard !a.isEmpty else { return 0 }
    return sum(a) / Float16(a.count)
  }
  
  public static func sqrt(_ a: [Float16]) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = Float16(Foundation.sqrt(Float(a[i]))) }
    return c
  }
  
  public static func transpose(_ a: [Float16], rows: Int, columns: Int) -> [Float16] {
    var c = [Float16](repeating: 0, count: rows * columns)
    for r in 0..<rows {
      for col in 0..<columns {
        c[col * rows + r] = a[r * columns + col]
      }
    }
    return c
  }
  
  public static func clip(_ a: [Float16], to limit: Float16) -> [Float16] {
    let count = a.count
    var c = [Float16](repeating: 0, count: count)
    for i in 0..<count { c[i] = Swift.max(-limit, Swift.min(limit, a[i])) }
    return c
  }
  
  public static func matmul(_ a: [Float16],
                            _ b: [Float16],
                            aRows: Int,
                            aCols: Int,
                            bRows: Int,
                            bCols: Int) -> [Float16] {
    precondition(aCols == bRows, "A columns (\(aCols)) must equal B rows (\(bRows))")
    
    let out = NumSwiftC.matmul1d(Array(a), b: Array(b),
                                 aSize: (aRows, aCols),
                                 bSize: (bRows, bCols))
    
    return out
  }
  
  // MARK: - Float16 Convolution / Padding
  
  public static func conv2d(signal: [Float16],
                            filter: [Float16],
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> [Float16] {
    let result = NumSwiftC.conv1d(signal: Array(signal),
                                  filter: Array(filter),
                                  strides: strides,
                                  padding: padding,
                                  filterSize: filterSize,
                                  inputSize: inputSize)
    return result
  }
  
  public static func transConv2d(signal: [Float16],
                                 filter: [Float16],
                                 strides: (Int, Int) = (1,1),
                                 padding: NumSwift.ConvPadding = .valid,
                                 filterSize: (rows: Int, columns: Int),
                                 inputSize: (rows: Int, columns: Int)) -> [Float16] {
    let result = NumSwiftC.transConv1d(signal: Array(signal),
                                       filter: Array(filter),
                                       strides: strides,
                                       padding: padding,
                                       filterSize: filterSize,
                                       inputSize: inputSize)
    return result
  }
  
  public static func zeroPad(signal: [Float16],
                             padding: NumSwiftPadding,
                             inputSize: (rows: Int, columns: Int)) -> [Float16] {
    guard padding.right > 0 || padding.left > 0 || padding.top > 0 || padding.bottom > 0 else {
      return signal
    }
    // No flat C function for Float16 specific_zero_pad; do it in Swift
    let expectedRows = inputSize.rows + padding.top + padding.bottom
    let expectedColumns = inputSize.columns + padding.left + padding.right
    var result = [Float16](repeating: 0, count: expectedRows * expectedColumns)
    for r in 0..<inputSize.rows {
      for c in 0..<inputSize.columns {
        result[(r + padding.top) * expectedColumns + (c + padding.left)] = signal[r * inputSize.columns + c]
      }
    }
    return result
  }
  
  public static func zeroPad(signal: [Float16],
                             filterSize: (rows: Int, columns: Int),
                             inputSize: (rows: Int, columns: Int),
                             stride: (Int, Int) = (1,1)) -> [Float16] {
    let padding = NumSwiftC.paddingCalculation(strides: stride, padding: .same, filterSize: filterSize, inputSize: inputSize)
    let numPadding = NumSwiftPadding(top: padding.top, left: padding.left, right: padding.right, bottom: padding.bottom)
    return zeroPad(signal: signal, padding: numPadding, inputSize: inputSize)
  }
  
  public static func stridePad(signal: [Float16],
                               strides: (rows: Int, columns: Int),
                               inputSize: (rows: Int, columns: Int)) -> [Float16] {
    guard strides.rows - 1 > 0 || strides.columns - 1 > 0 else { return signal }
    
    let newRows = inputSize.rows + ((strides.rows - 1) * (inputSize.rows - 1))
    let newColumns = inputSize.columns + ((strides.columns - 1) * (inputSize.columns - 1))
    var result = [Float16](repeating: 0, count: newRows * newColumns)
    
    for r in 0..<inputSize.rows {
      for c in 0..<inputSize.columns {
        result[r * strides.rows * newColumns + c * strides.columns] = signal[r * inputSize.columns + c]
      }
    }
    
    return result
  }
  
  public static func flip180(_ a: [Float16], rows: Int, columns: Int) -> [Float16] {
    var result = [Float16](repeating: 0, count: a.count)
    for r in 0..<rows {
      let srcRow = rows - 1 - r
      let srcStart = srcRow * columns
      let dstStart = r * columns
      for c in 0..<columns {
        result[dstStart + c] = a[srcStart + (columns - 1 - c)]
      }
    }
    return result
  }
  #endif
}

// MARK: ContiguousArray

extension NumSwiftFlat {
  
  // MARK: - Element-wise Arithmetic (array + array)
  
  /// Element-wise addition using vDSP_vadd
  /// we dont need count check here because if two arrays are different sizes we'll just add the number of elements in A
  public static func add(_ a: ContiguousArray<Float>, _ b: ContiguousArray<Float>) -> ContiguousArray<Float> {
    let count = a.count
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          vDSP_vadd(aBuf.baseAddress!, 1, bBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  /// Element-wise subtraction (a - b) using vDSP_vsub
  /// we dont need count check here because if two arrays are different sizes we'll just sub the number of elements in A
  public static func subtract(_ a: ContiguousArray<Float>, _ b: ContiguousArray<Float>) -> ContiguousArray<Float> {
    let count = a.count
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          // vDSP_vsub computes C = B - A (note reversed order)
          vDSP_vsub(bBuf.baseAddress!, 1, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  /// Element-wise multiplication using vDSP_vmul
  /// we dont need count check here because if two arrays are different sizes we'll just multiply the number of elements in A
  public static func multiply(_ a: ContiguousArray<Float>, _ b: ContiguousArray<Float>) -> ContiguousArray<Float> {
    let count = a.count
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          vDSP_vmul(aBuf.baseAddress!, 1, bBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  /// Element-wise division (a / b) using vDSP_vdiv
  /// we dont need count check here because if two arrays are different sizes we'll just divide the number of elements in A
  public static func divide(_ a: ContiguousArray<Float>, _ b: ContiguousArray<Float>) -> ContiguousArray<Float> {
    let count = a.count
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          // vDSP_vdiv computes C = B / A (note reversed order)
          vDSP_vdiv(bBuf.baseAddress!, 1, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  // MARK: - Scalar Arithmetic (array op scalar)
  
  /// Add scalar to every element using vDSP_vsadd
  public static func add(_ a: ContiguousArray<Float>, scalar: Float) -> ContiguousArray<Float> {
    let count = a.count
    var s = scalar
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsadd(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  /// Multiply every element by scalar using vDSP_vsmul
  public static func multiply(_ a: ContiguousArray<Float>, scalar: Float) -> ContiguousArray<Float> {
    let count = a.count
    var s = scalar
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsmul(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  /// Divide every element by scalar using vDSP_vsdiv
  public static func divide(_ a: ContiguousArray<Float>, scalar: Float) -> ContiguousArray<Float> {
    let count = a.count
    var s = scalar
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsdiv(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  /// Subtract scalar from every element: result = a - scalar
  public static func subtract(_ a: ContiguousArray<Float>, scalar: Float) -> ContiguousArray<Float> {
    return add(a, scalar: -scalar)
  }
  
  /// Scalar minus every element: result = scalar - a. Uses vDSP_vneg + vDSP_vsadd.
  public static func subtract(scalar: Float, _ a: ContiguousArray<Float>) -> ContiguousArray<Float> {
    let count = a.count
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    var s = scalar
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        // negate a
        vDSP_vneg(aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        // add scalar
        vDSP_vsadd(cBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  /// Scalar divided by every element: result = scalar / a[i]
  public static func divide(scalar: Float, _ a: ContiguousArray<Float>) -> ContiguousArray<Float> {
    let count = a.count
    var s = scalar
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_svdiv(&s, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  // MARK: - Negation
  
  /// Negate every element using vDSP_vneg
  public static func negate(_ a: ContiguousArray<Float>) -> ContiguousArray<Float> {
    let count = a.count
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vneg(aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  // MARK: - Reductions
  
  /// Sum of all elements using vDSP_sve
  public static func sum(_ a: ContiguousArray<Float>) -> Float {
    var result: Float = 0
    a.withUnsafeBufferPointer { aBuf in
      vDSP_sve(aBuf.baseAddress!, 1, &result, vDSP_Length(a.count))
    }
    return result
  }
  
  /// Sum of squares using vDSP_svesq
  public static func sumOfSquares(_ a: ContiguousArray<Float>) -> Float {
    var result: Float = 0
    a.withUnsafeBufferPointer { aBuf in
      vDSP_svesq(aBuf.baseAddress!, 1, &result, vDSP_Length(a.count))
    }
    return result
  }
  
  /// Mean of all elements using vDSP_meanv
  public static func mean(_ a: ContiguousArray<Float>) -> Float {
    var result: Float = 0
    a.withUnsafeBufferPointer { aBuf in
      vDSP_meanv(aBuf.baseAddress!, 1, &result, vDSP_Length(a.count))
    }
    return result
  }
  
  // MARK: - Square Root
  
  /// Element-wise square root using vForce
  public static func sqrt(_ a: ContiguousArray<Float>) -> ContiguousArray<Float> {
    let count = a.count
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        var n = Int32(count)
        vvsqrtf(cBuf.baseAddress!, aBuf.baseAddress!, &n)
      }
    }
    return c
  }
  
  // MARK: - Matrix Transpose
  
  /// Transpose a single 2D matrix stored as flat row-major using vDSP_mtrans
  /// rows: is the number of rows in the input matrix
  /// columns: is the number of columns in the input matrix
  public static func transpose(_ a: ContiguousArray<Float>, rows: Int, columns: Int) -> ContiguousArray<Float> {
    var c = ContiguousArray<Float>(repeating: 0, count: rows * columns)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_mtrans(aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(columns), vDSP_Length(rows))
      }
    }
    return c
  }
  
  // MARK: - Matrix Multiplication
  
  /// Performs matrix multiplication on flat ContiguousArray<Float> using Accelerate's vDSP_mmul.
  public static func matmul(_ a: ContiguousArray<Float>,
                            _ b: ContiguousArray<Float>,
                            aRows: Int,
                            aCols: Int,
                            bRows: Int,
                            bCols: Int) -> ContiguousArray<Float> {
    precondition(aCols == bRows, "A columns (\(aCols)) must equal B rows (\(bRows))")
    
    let out = NumSwiftC.matmul1d(Array(a), b: Array(b),
                                 aSize: (aRows, aCols),
                                 bSize: (bRows, bCols))
    
    return ContiguousArray(out)
  }
  
  // MARK: - Clip
  
  /// Clip all values to [-limit, limit] using vDSP_vclip
  public static func clip(_ a: ContiguousArray<Float>, to limit: Float) -> ContiguousArray<Float> {
    let count = a.count
    var low = -limit
    var high = limit
    var c = ContiguousArray<Float>(repeating: 0, count: count)
    a.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vclip(aBuf.baseAddress!, 1, &low, &high, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  // MARK: - Float16 Support
  
  // MARK: - Convolution (flat row-major)
  
  /// 2D convolution on flat row-major arrays via the C `nsc_conv1d` function.
  public static func conv2d(signal: ContiguousArray<Float>,
                            filter: ContiguousArray<Float>,
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> ContiguousArray<Float> {
    let result = NumSwiftC.conv1d(signal: Array(signal),
                                  filter: Array(filter),
                                  strides: strides,
                                  padding: padding,
                                  filterSize: filterSize,
                                  inputSize: inputSize)
    return ContiguousArray(result)
  }
  
  /// Transposed 2D convolution on flat row-major arrays via the C `nsc_transConv1d` function.
  public static func transConv2d(signal: ContiguousArray<Float>,
                                 filter: ContiguousArray<Float>,
                                 strides: (Int, Int) = (1,1),
                                 padding: NumSwift.ConvPadding = .valid,
                                 filterSize: (rows: Int, columns: Int),
                                 inputSize: (rows: Int, columns: Int)) -> ContiguousArray<Float> {
    let result = NumSwiftC.transConv1d(signal: Array(signal),
                                       filter: Array(filter),
                                       strides: strides,
                                       padding: padding,
                                       filterSize: filterSize,
                                       inputSize: inputSize)
    return ContiguousArray(result)
  }
  
  /// Zero-pad a flat row-major 2D array with explicit padding values.
  public static func zeroPad(signal: ContiguousArray<Float>,
                             padding: NumSwiftPadding,
                             inputSize: (rows: Int, columns: Int)) -> ContiguousArray<Float> {
    guard padding.right > 0 || padding.left > 0 || padding.top > 0 || padding.bottom > 0 else {
      return signal
    }
    
    let expectedRows = inputSize.rows + padding.top + padding.bottom
    let expectedColumns = inputSize.columns + padding.left + padding.right
    var result = ContiguousArray<Float>(repeating: 0, count: expectedRows * expectedColumns)
    
    // Copy original data into the padded result at the correct offsets
    for r in 0..<inputSize.rows {
      for c in 0..<inputSize.columns {
        result[(r + padding.top) * expectedColumns + (c + padding.left)] = signal[r * inputSize.columns + c]
      }
    }
    
    return result
  }
  
  /// Zero-pad a flat row-major 2D array computed from filter/input sizes.
  public static func zeroPad(signal: ContiguousArray<Float>,
                             filterSize: (rows: Int, columns: Int),
                             inputSize: (rows: Int, columns: Int),
                             stride: (Int, Int) = (1,1)) -> ContiguousArray<Float> {
    let result = NumSwiftC.zeroPad(signal: Array(signal),
                                   filterSize: filterSize,
                                   inputSize: inputSize,
                                   stride: stride)
    return ContiguousArray(result)
  }
  
  /// Stride-pad a flat row-major 2D array (inserts zeros between elements).
  /// Places original values at stride intervals in a larger zero-filled output.
  public static func stridePad(signal: ContiguousArray<Float>,
                               strides: (rows: Int, columns: Int),
                               inputSize: (rows: Int, columns: Int)) -> ContiguousArray<Float> {
    guard strides.rows - 1 > 0 || strides.columns - 1 > 0 else { return signal }
    
    let newRows = inputSize.rows + ((strides.rows - 1) * (inputSize.rows - 1))
    let newColumns = inputSize.columns + ((strides.columns - 1) * (inputSize.columns - 1))
    var result = ContiguousArray<Float>(repeating: 0, count: newRows * newColumns)
    
    for r in 0..<inputSize.rows {
      for c in 0..<inputSize.columns {
        result[r * strides.rows * newColumns + c * strides.columns] = signal[r * inputSize.columns + c]
      }
    }
    
    return result
  }
  
  /// Flip a flat row-major 2D matrix 180 degrees (reverse all elements, then reverse each row).
  /// Equivalent to reversing the entire array then reversing each row segment.
  public static func flip180(_ a: ContiguousArray<Float>, rows: Int, columns: Int) -> ContiguousArray<Float> {
    var result = ContiguousArray<Float>(repeating: 0, count: a.count)
    // flip180 = reverse rows, then reverse each row's columns
    for r in 0..<rows {
      let srcRow = rows - 1 - r
      let srcStart = srcRow * columns
      let dstStart = r * columns
      for c in 0..<columns {
        result[dstStart + c] = a[srcStart + (columns - 1 - c)]
      }
    }
    return result
  }
  
  #if arch(arm64)
  // Float16 versions use manual loops (no Accelerate support for Float16).
  // The compiler can auto-vectorize these for ARM NEON.
  
  public static func add(_ a: ContiguousArray<Float16>, _ b: ContiguousArray<Float16>) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] + b[i] }
    return c
  }
  
  public static func subtract(_ a: ContiguousArray<Float16>, _ b: ContiguousArray<Float16>) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] - b[i] }
    return c
  }
  
  public static func multiply(_ a: ContiguousArray<Float16>, _ b: ContiguousArray<Float16>) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] * b[i] }
    return c
  }
  
  public static func divide(_ a: ContiguousArray<Float16>, _ b: ContiguousArray<Float16>) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] / b[i] }
    return c
  }
  
  public static func add(_ a: ContiguousArray<Float16>, scalar: Float16) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] + scalar }
    return c
  }
  
  public static func multiply(_ a: ContiguousArray<Float16>, scalar: Float16) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] * scalar }
    return c
  }
  
  public static func divide(_ a: ContiguousArray<Float16>, scalar: Float16) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = a[i] / scalar }
    return c
  }
  
  public static func subtract(_ a: ContiguousArray<Float16>, scalar: Float16) -> ContiguousArray<Float16> {
    return add(a, scalar: -scalar)
  }
  
  public static func subtract(scalar: Float16, _ a: ContiguousArray<Float16>) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = scalar - a[i] }
    return c
  }
  
  public static func divide(scalar: Float16, _ a: ContiguousArray<Float16>) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = scalar / a[i] }
    return c
  }
  
  public static func negate(_ a: ContiguousArray<Float16>) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = -a[i] }
    return c
  }
  
  public static func sum(_ a: ContiguousArray<Float16>) -> Float16 {
    var result: Float16 = 0
    for i in 0..<a.count { result += a[i] }
    return result
  }
  
  public static func sumOfSquares(_ a: ContiguousArray<Float16>) -> Float16 {
    var result: Float16 = 0
    for i in 0..<a.count { result += a[i] * a[i] }
    return result
  }
  
  public static func mean(_ a: ContiguousArray<Float16>) -> Float16 {
    guard !a.isEmpty else { return 0 }
    return sum(a) / Float16(a.count)
  }
  
  public static func sqrt(_ a: ContiguousArray<Float16>) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = Float16(Foundation.sqrt(Float(a[i]))) }
    return c
  }
  
  public static func transpose(_ a: ContiguousArray<Float16>, rows: Int, columns: Int) -> ContiguousArray<Float16> {
    var c = ContiguousArray<Float16>(repeating: 0, count: rows * columns)
    for r in 0..<rows {
      for col in 0..<columns {
        c[col * rows + r] = a[r * columns + col]
      }
    }
    return c
  }
  
  public static func clip(_ a: ContiguousArray<Float16>, to limit: Float16) -> ContiguousArray<Float16> {
    let count = a.count
    var c = ContiguousArray<Float16>(repeating: 0, count: count)
    for i in 0..<count { c[i] = Swift.max(-limit, Swift.min(limit, a[i])) }
    return c
  }
  
  public static func matmul(_ a: ContiguousArray<Float16>,
                            _ b: ContiguousArray<Float16>,
                            aRows: Int,
                            aCols: Int,
                            bRows: Int,
                            bCols: Int) -> ContiguousArray<Float16> {
    precondition(aCols == bRows, "A columns (\(aCols)) must equal B rows (\(bRows))")
    
    let out = NumSwiftC.matmul1d(Array(a), b: Array(b),
                                 aSize: (aRows, aCols),
                                 bSize: (bRows, bCols))
    
    return ContiguousArray(out)
  }
  
  // MARK: - Float16 Convolution / Padding
  
  public static func conv2d(signal: ContiguousArray<Float16>,
                            filter: ContiguousArray<Float16>,
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> ContiguousArray<Float16> {
    let result = NumSwiftC.conv1d(signal: Array(signal),
                                  filter: Array(filter),
                                  strides: strides,
                                  padding: padding,
                                  filterSize: filterSize,
                                  inputSize: inputSize)
    return ContiguousArray(result)
  }
  
  public static func transConv2d(signal: ContiguousArray<Float16>,
                                 filter: ContiguousArray<Float16>,
                                 strides: (Int, Int) = (1,1),
                                 padding: NumSwift.ConvPadding = .valid,
                                 filterSize: (rows: Int, columns: Int),
                                 inputSize: (rows: Int, columns: Int)) -> ContiguousArray<Float16> {
    let result = NumSwiftC.transConv1d(signal: Array(signal),
                                       filter: Array(filter),
                                       strides: strides,
                                       padding: padding,
                                       filterSize: filterSize,
                                       inputSize: inputSize)
    return ContiguousArray(result)
  }
  
  public static func zeroPad(signal: ContiguousArray<Float16>,
                             padding: NumSwiftPadding,
                             inputSize: (rows: Int, columns: Int)) -> ContiguousArray<Float16> {
    guard padding.right > 0 || padding.left > 0 || padding.top > 0 || padding.bottom > 0 else {
      return signal
    }
    // No flat C function for Float16 specific_zero_pad; do it in Swift
    let expectedRows = inputSize.rows + padding.top + padding.bottom
    let expectedColumns = inputSize.columns + padding.left + padding.right
    var result = ContiguousArray<Float16>(repeating: 0, count: expectedRows * expectedColumns)
    for r in 0..<inputSize.rows {
      for c in 0..<inputSize.columns {
        result[(r + padding.top) * expectedColumns + (c + padding.left)] = signal[r * inputSize.columns + c]
      }
    }
    return result
  }
  
  public static func zeroPad(signal: ContiguousArray<Float16>,
                             filterSize: (rows: Int, columns: Int),
                             inputSize: (rows: Int, columns: Int),
                             stride: (Int, Int) = (1,1)) -> ContiguousArray<Float16> {
    let padding = NumSwiftC.paddingCalculation(strides: stride, padding: .same, filterSize: filterSize, inputSize: inputSize)
    let numPadding = NumSwiftPadding(top: padding.top, left: padding.left, right: padding.right, bottom: padding.bottom)
    return zeroPad(signal: signal, padding: numPadding, inputSize: inputSize)
  }
  
  public static func stridePad(signal: ContiguousArray<Float16>,
                               strides: (rows: Int, columns: Int),
                               inputSize: (rows: Int, columns: Int)) -> ContiguousArray<Float16> {
    guard strides.rows - 1 > 0 || strides.columns - 1 > 0 else { return signal }
    
    let newRows = inputSize.rows + ((strides.rows - 1) * (inputSize.rows - 1))
    let newColumns = inputSize.columns + ((strides.columns - 1) * (inputSize.columns - 1))
    var result = ContiguousArray<Float16>(repeating: 0, count: newRows * newColumns)
    
    for r in 0..<inputSize.rows {
      for c in 0..<inputSize.columns {
        result[r * strides.rows * newColumns + c * strides.columns] = signal[r * inputSize.columns + c]
      }
    }
    
    return result
  }
  
  public static func flip180(_ a: ContiguousArray<Float16>, rows: Int, columns: Int) -> ContiguousArray<Float16> {
    var result = ContiguousArray<Float16>(repeating: 0, count: a.count)
    for r in 0..<rows {
      let srcRow = rows - 1 - r
      let srcStart = srcRow * columns
      let dstStart = r * columns
      for c in 0..<columns {
        result[dstStart + c] = a[srcStart + (columns - 1 - c)]
      }
    }
    return result
  }
  #endif
}
