//
//  1DFloat.swift
//  NumSwift
//
//  Created by William Vabrinskas on 2/10/26.
//

//
//  File.swift
//
//
//  Created by William Vabrinskas on 1/22/22.
//

import Foundation
import Accelerate


// MARK: 1D operations
//use accelerate
public extension Array where Element == Float {
  @inline(__always)
  var sum: Element {
    FloatArrayArithmetic.sum(self)
  }

  @inline(__always)
  var sumOfSquares: Element {
    FloatArrayArithmetic.sumOfSquares(self)
  }

  @inline(__always)
  var indexOfMin: (UInt, Element) {
    FloatArrayArithmetic.indexOfMin(self)
  }

  @inline(__always)
  var indexOfMax: (UInt, Element) {
    FloatArrayArithmetic.indexOfMax(self)
  }

  @inline(__always)
  var max: Element {
    FloatArrayArithmetic.max(self)
  }

  @inline(__always)
  var min: Element {
    FloatArrayArithmetic.min(self)
  }

  @inline(__always)
  var mean: Element {
    FloatArrayArithmetic.mean(self)
  }

  @inline(__always)
  static func +(lhs: [Element], rhs: Element) -> [Element] {
    return FloatArrayArithmetic.add(lhs: rhs, rhs: lhs)
  }

  @inline(__always)
  static func +(lhs: Element, rhs: [Element]) -> [Element] {
    return FloatArrayArithmetic.add(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func +(lhs: [Element], rhs: [Element]) -> [Element] {
    return FloatArrayArithmetic.add(lhs: rhs, rhs: lhs)
  }

  @inline(__always)
  static func -(lhs: [Element], rhs: [Element]) -> [Element] {
    return FloatArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }
  
  @inline(__always)
  static func -(lhs: Self, rhs: Element) -> Self {
    FloatArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }
  
  @inline(__always)
  static func -(lhs: Element, rhs: Self) -> Self {
    FloatArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: [Element], rhs: Element) -> [Element] {
    return FloatArrayArithmetic.mult(lhs: rhs, rhs: lhs)
  }

  @inline(__always)
  static func *(lhs: Element, rhs: [Element]) -> [Element] {
    return FloatArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: [Element], rhs: [Element]) -> [Element] {
    return FloatArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: [Element], rhs: [Element]) -> [Element] {
    return FloatArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: [Element], rhs: Element) -> [Element] {
    return FloatArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: Element, rhs: [Element]) -> [Element] {
    return FloatArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }
  
  func reverse() -> Self {
    var result = self
    vDSP.reverse(&result)
    return result
  }

}


// MARK: Contiguous Array Float

public extension ContiguousArray where Element == Float {
  var sumOfSquares: Float {
    FloatContiguousArrayArithmetic.sumOfSquares(self)
  }
  
  var mean: Float {
    FloatContiguousArrayArithmetic.mean(self)
  }
  
  var sum: Float {
    FloatContiguousArrayArithmetic.sum(self)
  }
  
  @inline(__always)
  var indexOfMin: (UInt, Element) {
    FloatContiguousArrayArithmetic.indexOfMin(self)
  }

  @inline(__always)
  var indexOfMax: (UInt, Element) {
    FloatContiguousArrayArithmetic.indexOfMax(self)
  }

  @inline(__always)
  var max: Element {
    FloatContiguousArrayArithmetic.max(self)
  }

  @inline(__always)
  var min: Element {
    FloatContiguousArrayArithmetic.min(self)
  }

  @inline(__always)
  static func +(lhs: Self, rhs: Element) -> Self {
    FloatContiguousArrayArithmetic.add(lhs: rhs, rhs: lhs)
  }

  @inline(__always)
  static func +(lhs: Element, rhs: Self) -> Self{
    let count = rhs.count
    var s = lhs
    var c = Self(repeating: 0, count: count)
    rhs.withUnsafeBufferPointer { aBuf in
      c.withUnsafeMutableBufferPointer { cBuf in
        vDSP_vsadd(aBuf.baseAddress!, 1, &s, cBuf.baseAddress!, 1, vDSP_Length(count))
      }
    }
    return c
  }

  @inline(__always)
  static func +(lhs: Self, rhs: Self) -> Self {
    FloatContiguousArrayArithmetic.add(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func -(lhs: Self, rhs: Self) -> Self {
    FloatContiguousArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }
  
  @inline(__always)
  static func -(lhs: Self, rhs: Element) -> Self {
    FloatContiguousArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }
  
  @inline(__always)
  static func -(lhs: Element, rhs: Self) -> Self {
    FloatContiguousArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: Self, rhs: Element) -> Self {
    FloatContiguousArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: Element, rhs: Self) -> Self {
    FloatContiguousArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: Self, rhs: Self) -> Self {
    FloatContiguousArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: Self, rhs: Self) -> Self {
    FloatContiguousArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: Self, rhs: Element) -> Self {
    FloatContiguousArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: Element, rhs: Self) -> Self {
    FloatContiguousArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }
}
