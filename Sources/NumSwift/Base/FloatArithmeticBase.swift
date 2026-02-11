//
//  FloatArithmeticBase.swift
//  NumSwift
//
//  Created by William Vabrinskas on 2/10/26.
//


public protocol FloatArithmeticBase {
  associatedtype Scalar: FloatingPoint
  associatedtype Value: Collection<Scalar>
  
  static func sum(_ array: Value) -> Scalar
  static func sumOfSquares(_ array: Value) -> Scalar
  static func indexOfMin(_ array: Value) -> (UInt, Scalar)
  static func indexOfMax(_ array: Value) -> (UInt, Scalar)
  static func max(_ array: Value) -> Scalar
  static func min(_ array: Value) -> Scalar
  static func mean(_ array: Value) -> Scalar

  static func add(lhs: Scalar, rhs: Value) -> Value
  static func add(lhs: Value, rhs: Value) -> Value
  
  static func sub(lhs: Scalar, rhs: Value) -> Value
  static func sub(lhs: Value, rhs: Value) -> Value
  static func sub(lhs: Value, rhs: Scalar) -> Value

  static func mult(lhs: Value, rhs: Scalar) -> Value
  static func mult(lhs: Scalar, rhs: Value) -> Value
  static func mult(lhs: Value, rhs: Value) -> Value
  
  static func div(lhs: Value, rhs: Value) -> Value
  static func div(lhs: Value, rhs: Scalar) -> Value
  static func div(lhs: Scalar, rhs: Value) -> Value
}

