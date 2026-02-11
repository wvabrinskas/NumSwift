//
//  ScalarContiguousArrayArithmetic.swift
//  NumSwift
//
//  Created by William Vabrinskas on 2/10/26.
//


//
//  Float32ContinuousArithmetic.swift
//  NumSwift
//
//  Created by William Vabrinskas on 2/10/26.
//

import Accelerate
import Foundation
import NumSwiftC

#if arch(arm64)
struct Float16ContiguousArrayArithmetic: FloatArithmeticBase {
  typealias Scalar = Float16
  typealias Value = ContiguousArray<Scalar>
  
  public static func sum(_ array: Value) -> Scalar {
    var c: Scalar = .nan

    array.withUnsafeBufferPointer { aBuf in
      c = nsc_sum(aBuf.baseAddress!, Int32(array.count))
    }

    return c
  }
  
  public static func sumOfSquares(_ array: Value) -> Scalar {
    var c: Scalar = .nan

    array.withUnsafeBufferPointer { aBuf in
      c = nsc_sum_of_squares(aBuf.baseAddress!, Int32(array.count))
    }

    return c
  }
  
  public static func indexOfMin(_ array: Value) -> (UInt, Scalar) {
    var out: NSC_IndexedValue = .init(value: 0, index: 0)

    array.withUnsafeBufferPointer { aBuf in
      out = nsc_index_of_min(aBuf.baseAddress!, Int32(array.count))
    }

    return (UInt(out.index), out.value)
  }
  
  public static func indexOfMax(_ array: Value) -> (UInt, Scalar) {
    var out: NSC_IndexedValue = .init(value: 0, index: 0)

    array.withUnsafeBufferPointer { aBuf in
      out = nsc_index_of_max(aBuf.baseAddress!, Int32(array.count))
    }

    return (UInt(out.index), out.value)
  }
  
  public static func max(_ array: Value) -> Scalar {
    var c: Scalar = .nan

    array.withUnsafeBufferPointer { aBuf in
      c = nsc_max(aBuf.baseAddress!, Int32(array.count))
    }

    return c  }
  
  public static func min(_ array: Value) -> Scalar {
    var c: Scalar = .nan

    array.withUnsafeBufferPointer { aBuf in
      c = nsc_min(aBuf.baseAddress!, Int32(array.count))
    }

    return c
  }
  
  public static func mean(_ array: Value) -> Scalar {
    var c: Scalar = .nan

    array.withUnsafeBufferPointer { aBuf in
      c = nsc_mean(aBuf.baseAddress!, Int32(array.count))
    }

    return c
  }
  
  public static func add(lhs: Scalar, rhs: Value) -> Value {
    let count = rhs.count
    var s = lhs
    var c = Value(repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        nsc_add_scalar(s, aBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }
  
  public static func add(lhs: Value, rhs: Value) -> Value {
    let count = lhs.count
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      rhs.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          nsc_add(aBuf.baseAddress!, bBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
        }
      }
    }
    return c
  }
  
  public static func sub(lhs: Value, rhs: Value) -> Value {
    let count = lhs.count
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      rhs.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          nsc_sub(aBuf.baseAddress!, bBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
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
        nsc_sub_scalar_f16(aBuf.baseAddress!, s, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }
  
  public static func sub(lhs: Scalar, rhs: Value) -> Value {
    let count = rhs.count
    var s = lhs
    var c = Value(repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        nsc_sub_scalar_array_f16(s, aBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }
  
  public static func mult(lhs: Value, rhs: Scalar) -> Value {
    let count = lhs.count
    var s = rhs
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        nsc_mult_scalar(s, aBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }

  public static func mult(lhs: Scalar, rhs: Value) -> Value {
    let count = rhs.count
    var s = lhs
    var c = Value(repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        nsc_mult_scalar(s, aBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }
  
  public static func mult(lhs: Value, rhs: Value) -> Value {
    let count = lhs.count
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      rhs.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          nsc_mult(aBuf.baseAddress!, bBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
        }
      }
    }
    return c
  }
  
  public static func div(lhs: Value, rhs: Value) -> Value {
    let count = lhs.count
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      rhs.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          nsc_div(aBuf.baseAddress!, bBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
        }
      }
    }
    return c
  }

  public static func div(lhs: Value, rhs: Scalar) -> Value {
    let count = lhs.count
    var s = rhs
    var c = Value(repeating: 0, count: count)
    lhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        nsc_div_array_scalar(aBuf.baseAddress!, s, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }

  public static func div(lhs: Scalar, rhs: Value) -> Value {
    let count = rhs.count
    var s = lhs
    var c = Value(repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        nsc_div_scalar_array(s, aBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }
}
#endif
