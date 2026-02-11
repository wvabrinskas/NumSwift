//
//  1DFloat16.swift
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

//protocol FloatingPointMath: Collection {
//  @inline(__always)
//  var sum: Element { get  }
//
//  @inline(__always)
//  var sumOfSquares: Element { get }
//
//  @inline(__always)
//  var indexOfMin: (UInt, Element) { get }
//
//  @inline(__always)
//  var indexOfMax: (UInt, Element) { get }
//
//  @inline(__always)
//  var max: Element { get }
//
//  @inline(__always)
//  var min: Element { get }
//
//  @inline(__always)
//  var mean: Element { get }
//  
//  @inline(__always)
//  static func +(lhs: Self, rhs: Element) -> Self
//
//  @inline(__always)
//  static func +(lhs: Element, rhs: Self) -> Self
//  
//  @inline(__always)
//  static func +(lhs: Self, rhs: Self) -> Self
//  
//  @inline(__always)
//  static func -(lhs: Self, rhs: Self) -> Self
//  
//  @inline(__always)
//  static func -(lhs: Self, rhs: Element) -> Self
//  
//  @inline(__always)
//  static func -(lhs: Element, rhs: Self) -> Self
//
//  @inline(__always)
//  static func *(lhs: Self, rhs: Element) -> Self
//
//  @inline(__always)
//  static func *(lhs: Element, rhs: Self) -> Self
//
//  @inline(__always)
//  static func *(lhs: Self, rhs: Self) -> Self
//
//  @inline(__always)
//  static func /(lhs: Self, rhs: Self) -> Self
//
//  @inline(__always)
//  static func /(lhs: Self, rhs: Element) -> Self
//
//  @inline(__always)
//  static func /(lhs: Element, rhs: Self) -> Self
//}

#if arch(arm64)

//use accelerate
public extension Array where Element == Float16 {
  @inline(__always)
  var sum: Element {
    Float16ArrayArithmetic.sum(self)
  }

  @inline(__always)
  var sumOfSquares: Element {
    Float16ArrayArithmetic.sumOfSquares(self)
  }

  @inline(__always)
  var indexOfMin: (UInt, Element) {
    Float16ArrayArithmetic.indexOfMin(self)
  }

  @inline(__always)
  var indexOfMax: (UInt, Element) {
    Float16ArrayArithmetic.indexOfMax(self)
  }

  @inline(__always)
  var max: Element {
    Float16ArrayArithmetic.max(self)
  }

  @inline(__always)
  var min: Element {
    Float16ArrayArithmetic.min(self)
  }

  @inline(__always)
  var mean: Element {
    Float16ArrayArithmetic.mean(self)
  }
  
  @inline(__always)
  static func +(lhs: [Element], rhs: Element) -> [Element] {
    Float16ArrayArithmetic.add(lhs: rhs, rhs: lhs)
  }

  @inline(__always)
  static func +(lhs: Element, rhs: [Element]) -> [Element] {
    Float16ArrayArithmetic.add(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func +(lhs: [Element], rhs: [Element]) -> [Element] {
    Float16ArrayArithmetic.add(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func -(lhs: [Element], rhs: [Element]) -> [Element] {
    Float16ArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }
  
  @inline(__always)
  static func -(lhs: Self, rhs: Element) -> Self {
    Float16ArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }
  
  @inline(__always)
  static func -(lhs: Element, rhs: Self) -> Self {
    Float16ArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: [Element], rhs: Element) -> [Element] {
    Float16ArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: Element, rhs: [Element]) -> [Element] {
    Float16ArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: [Element], rhs: [Element]) -> [Element] {
    Float16ArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: [Element], rhs: [Element]) -> [Element] {
    Float16ArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: [Element], rhs: Element) -> [Element] {
    Float16ArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: Element, rhs: [Element]) -> [Element] {
    Float16ArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }
}


// MARK: Contiguous Array Float 16

public extension ContiguousArray where Element == Float16 {
  var sumOfSquares: Element {
    Float16ContiguousArrayArithmetic.sumOfSquares(self)
  }
  
  var mean: Element {
    Float16ContiguousArrayArithmetic.mean(self)
  }
  
  var sum: Element {
    Float16ContiguousArrayArithmetic.sum(self)
  }
  
  @inline(__always)
  var indexOfMin: (UInt, Element) {
    Float16ContiguousArrayArithmetic.indexOfMin(self)
  }

  @inline(__always)
  var indexOfMax: (UInt, Element) {
    Float16ContiguousArrayArithmetic.indexOfMax(self)
  }

  @inline(__always)
  var max: Element {
    Float16ContiguousArrayArithmetic.max(self)
  }

  @inline(__always)
  var min: Element {
    Float16ContiguousArrayArithmetic.min(self)
  }
  
  @inline(__always)
  static func +(lhs: Self, rhs: Element) -> Self {
    Float16ContiguousArrayArithmetic.add(lhs: rhs, rhs: lhs)
  }

  @inline(__always)
  static func +(lhs: Element, rhs: Self) -> Self {
    Float16ContiguousArrayArithmetic.add(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func +(lhs: Self, rhs: Self) -> Self {
    Float16ContiguousArrayArithmetic.add(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func -(lhs: Self, rhs: Self) -> Self {
    Float16ContiguousArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }
  
  @inline(__always)
  static func -(lhs: Self, rhs: Element) -> Self {
    Float16ContiguousArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }
  
  @inline(__always)
  static func -(lhs: Element, rhs: Self) -> Self {
    Float16ContiguousArrayArithmetic.sub(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: Self, rhs: Element) -> Self {
    Float16ContiguousArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: Element, rhs: Self) -> Self {
    Float16ContiguousArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func *(lhs: Self, rhs: Self) -> Self {
    Float16ContiguousArrayArithmetic.mult(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: Self, rhs: Self) -> Self {
    Float16ContiguousArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: Self, rhs: Element) -> Self {
    Float16ContiguousArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }

  @inline(__always)
  static func /(lhs: Element, rhs: Self) -> Self {
    Float16ContiguousArrayArithmetic.div(lhs: lhs, rhs: rhs)
  }
}

#endif
