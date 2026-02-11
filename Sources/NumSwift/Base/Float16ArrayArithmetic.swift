//
//  NumSwiftCF16Base.swift
//  NumSwift
//
//  Created by William Vabrinskas on 7/19/24.
//

import Foundation
import NumSwiftC


#if arch(arm64)
struct Float16ArrayArithmetic: FloatArithmeticBase {
  public static func sum(_ array: [Float16]) -> Float16 {
    nsc_sum(array, Int32(array.count))
  }

  public static func sumOfSquares(_ array: [Float16]) -> Float16 {
    nsc_sum_of_squares(array, Int32(array.count))
  }

  public static func indexOfMin(_ array: [Float16]) -> (UInt, Float16) {
    let out = nsc_index_of_min(array, Int32(array.count))
    return (UInt(out.index), out.value)
  }

  public static func indexOfMax(_ array: [Float16]) -> (UInt, Float16) {
    let out = nsc_index_of_max(array, Int32(array.count))
    return (UInt(out.index), out.value)
  }

  public static func max(_ array: [Float16]) -> Float16 {
    nsc_max(array, Int32(array.count))
  }

  public static func min(_ array: [Float16]) -> Float16 {
    nsc_min(array, Int32(array.count))
  }

  public static func mean(_ array: [Float16]) -> Float16 {
    nsc_mean(array, Int32(array.count))
  }

  public static func add(lhs: Float16, rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_add_scalar(lhs, rhs, Int32(rhs.count), &result)
    return result
  }

  public static func add(lhs: [Float16], rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_add(lhs, rhs, Int32(rhs.count), &result)
    return result
  }

  public static func sub(lhs: [Float16], rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_sub(lhs, rhs, Int32(rhs.count), &result)
    return result
  }

  public static func sub(lhs: [Float16], rhs: Float16) -> [Float16] {
    var result = [Float16](repeating: 0, count: lhs.count)
    nsc_sub_scalar_f16(lhs, rhs, Int32(lhs.count), &result)
    return result
  }

  public static func sub(lhs: Float16, rhs: [Float16]) -> [Float16] {
    let count = rhs.count
    var s = lhs
    var c = [Float16](repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        nsc_sub_scalar_array_f16(s, aBuf.baseAddress!, Int32(count), cBuf.baseAddress!)
      }
    }
    return c
  }

  public static func mult(lhs: [Float16], rhs: Float16) -> [Float16] {
    var result = [Float16](repeating: 0, count: lhs.count)
    nsc_mult_scalar(rhs, lhs, Int32(lhs.count), &result)
    return result
  }

  public static func mult(lhs: Float16, rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_mult_scalar(lhs, rhs, Int32(rhs.count), &result)
    return result
  }

  public static func mult(lhs: [Float16], rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_mult(lhs, rhs, Int32(rhs.count), &result)
    return result
  }

  public static func div(lhs: [Float16], rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_div(lhs, rhs, Int32(rhs.count), &result)
    return result
  }

  public static func div(lhs: [Float16], rhs: Float16) -> [Float16] {
    var result = [Float16](repeating: 0, count: lhs.count)
    nsc_div_array_scalar(lhs, rhs, Int32(lhs.count), &result)
    return result
  }

  public static func div(lhs: Float16, rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_div_scalar_array(lhs, rhs, Int32(rhs.count), &result)
    return result
  }
}

#endif
