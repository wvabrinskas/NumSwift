//
//  Scalar32Arithmetic.swift
//  NumSwift
//
//  Created by William Vabrinskas on 2/10/26.
//

import Foundation
import Accelerate
import NumSwiftC

struct FloatArrayArithmetic: FloatArithmeticBase {
  typealias Scalar = Float
  typealias Value = [Scalar]
  
  static func sum(_ array: Value) -> Scalar {
    vDSP.sum(array)
  }
  
  static func sumOfSquares(_ array: Value) -> Scalar {
    let stride = vDSP_Stride(1)
    let n = vDSP_Length(array.count)

    var c: Scalar = .nan

    vDSP_svesq(array,
               stride,
               &c,
               n)
    return c
  }
  
  static func indexOfMin(_ array: Value) -> (UInt, Scalar) {
    vDSP.indexOfMinimum(array)
  }
  
  static func indexOfMax(_ array: Value) -> (UInt, Scalar) {
    vDSP.indexOfMaximum(array)
  }
  
  static func max(_ array: Value) -> Scalar {
    vDSP.maximum(array)
  }
  
  static func min(_ array: Value) -> Scalar {
    vDSP.minimum(array)
  }
  
  static func mean(_ array: Value) -> Scalar {
    vDSP.mean(array)
  }
  
  static func add(lhs: Scalar, rhs: Value) -> Value {
    vDSP.add(lhs, rhs)
  }
  
  static func add(lhs: Value, rhs: Value) -> Value {
    vDSP.add(rhs, lhs)
  }
  
  static func sub(lhs: Value, rhs: Value) -> Value {
    vDSP.subtract(lhs, rhs)
  }
  
  static func sub(lhs: Value, rhs: Scalar) -> Value {
    var c = Value(repeating: 0, count: lhs.count)
    nsc_sub_scalar(lhs, rhs, Int32(lhs.count), &c)
    return c
  }
  
  /// Scalar minus every element: result = scalar - a. Uses vDSP_vneg + vDSP_vsadd.
  static func sub(lhs: Scalar, rhs: Value) -> Value {
    let count = rhs.count
    var c = Value(repeating: 0, count: count)
    var s = lhs
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        // negate a
        vDSP_vneg(aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        // add scalar
        vDSP_vsadd(cBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  static func mult(lhs: Value, rhs: Scalar) -> Value {
    vDSP.multiply(rhs, lhs)
  }
  
  static func mult(lhs: Scalar, rhs: Value) -> Value {
    vDSP.multiply(lhs, rhs)
  }
  
  static func mult(lhs: Value, rhs: Value) -> Value {
    vDSP.multiply(lhs, rhs)
  }
  
  static func div(lhs: Value, rhs: Value) -> Value {
    vDSP.divide(lhs, rhs)
  }
  
  static func div(lhs: Value, rhs: Scalar) -> Value {
    vDSP.divide(lhs, rhs)
  }
  
  static func div(lhs: Scalar, rhs: Value) -> Value {
    vDSP.divide(lhs, rhs)
  }
  
}
