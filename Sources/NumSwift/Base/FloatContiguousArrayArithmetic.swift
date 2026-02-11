//
//  Float32ContinuousArithmetic.swift
//  NumSwift
//
//  Created by William Vabrinskas on 2/10/26.
//

import Accelerate
import Foundation
import NumSwiftC

struct FloatContiguousArrayArithmetic: FloatArithmeticBase {
  typealias Scalar = Float
  typealias Value = ContiguousArray<Scalar>
  
  static func sum(_ array: Value) -> Scalar {
    vDSP.sum(array)
  }
  
  static func sumOfSquares(_ array: Value) -> Scalar {
    var result: Scalar = 0
    array.withUnsafeBufferPointer { aBuf in
      vDSP_svesq(aBuf.baseAddress!, 1, &result, vDSP_Length(array.count))
    }
    return result
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
    let count = rhs.count
    var s = lhs
    var c = Value(repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsadd(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  static func add(lhs: Value, rhs: Value) -> Value {
    let count = lhs.count
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      rhs.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          vDSP_vadd(aBuf.baseAddress!, 1, bBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  static func sub(lhs: Value, rhs: Value) -> Value {
    let count = lhs.count
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      rhs.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          // vDSP_vsub computes C = B - A (note reversed order)
          vDSP_vsub(bBuf.baseAddress!, 1, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  public static func sub(lhs: Value, rhs: Scalar) -> Value {
    let count = lhs.count
    var s = rhs
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        nsc_sub_scalar(aBuf.baseAddress!, s, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }
  
  /// Scalar minus every element: result = scalar - a. Uses vDSP_vneg + vDSP_vsadd.
  static func sub(lhs: Scalar, rhs: Value) -> Value {
    let count = rhs.count
    var c = ContiguousArray<Float>(repeating: 0, count: count)
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
    let count = lhs.count
    var s = rhs
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsmul(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  static func mult(lhs: Scalar, rhs: Value) -> Value {
    let count = rhs.count
    var s = lhs
    var c = Value(repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsmul(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  static func mult(lhs: Value, rhs: Value) -> Value {
    let count = lhs.count
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      rhs.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          vDSP_vmul(aBuf.baseAddress!, 1, bBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  static func div(lhs: Value, rhs: Value) -> Value {
    let count = lhs.count
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      rhs.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          // vDSP_vdiv computes C = B / A (note reversed order)
          vDSP_vdiv(bBuf.baseAddress!, 1, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
        }
      }
    }
    return c
  }
  
  static func div(lhs: Value, rhs: Scalar) -> Value {
    let count = lhs.count
    var s = rhs
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsdiv(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
  
  static func div(lhs: Scalar, rhs: Value) -> Value {
    let count = rhs.count
    var s = lhs
    var c = Value(repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_svdiv(&s, aBuf.baseAddress!, 1, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }
}
