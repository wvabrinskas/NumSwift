//
//  NumSwiftCF16Base.swift
//  NumSwift
//
//  Created by William Vabrinskas on 7/19/24.
//

import Foundation
import NumSwiftC

public protocol FloatArithmeticBase {
  associatedtype Scalar: FloatingPoint
  
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
}

public struct Float16Arithmetic: FloatArithmeticBase {
  public static func sum(_ array: [Float16]) -> Float16 {
    nsc_sum(array)
  }
  
  public static func sumOfSquares(_ array: [Float16]) -> Float16 {
    nsc_sum_of_squares(array)
  }
  
  public static func indexOfMin(_ array: [Float16]) -> (UInt, Float16) {
    let out = nsc_index_of_min(array)
    return (UInt(out.index), out.value)
  }
  
  public static func indexOfMax(_ array: [Float16]) -> (UInt, Float16) {
    let out = nsc_index_of_max(array)
    return (UInt(out.index), out.value)
  }
  
  public static func max(_ array: [Float16]) -> Float16 {
    nsc_max(array)
  }
  
  public static func min(_ array: [Float16]) -> Float16 {
    nsc_max(array)
  }
  
  public static func mean(_ array: [Float16]) -> Float16 {
    nsc_mean(array)
  }
  
  public static func add(lhs: Float16, rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_add_scalar(lhs, rhs, &result)
    return result
  }
  
  public static func add(lhs: [Float16], rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_add(lhs, rhs, &result)
    return result
  }
  
  public static func sub(lhs: [Float16], rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_sub(lhs, rhs, &result)
    return result
  }
  
  public static func mult(lhs: [Float16], rhs: Float16) -> [Float16] {
    var result = [Float16](repeating: 0, count: lhs.count)
    nsc_mult_scalar(rhs, lhs, &result)
    return result
  }
  
  public static func mult(lhs: Float16, rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_mult_scalar(lhs, rhs, &result)
    return result
  }
  
  public static func mult(lhs: [Float16], rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_mult(lhs, rhs, &result)
    return result
  }
  
  public static func div(lhs: [Float16], rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_div(lhs, rhs, &result)
    return result
  }
  
  public static func div(lhs: [Float16], rhs: Float16) -> [Float16] {
    var result = [Float16](repeating: 0, count: lhs.count)
    nsc_div_array_scalar(lhs, rhs, &result)
    return result
  }
  
  public static func div(lhs: Float16, rhs: [Float16]) -> [Float16] {
    var result = [Float16](repeating: 0, count: rhs.count)
    nsc_div_scalar_array(lhs, rhs, &result)
    return result
  }
}

