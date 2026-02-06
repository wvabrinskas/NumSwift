//
//  File.swift
//
//
//  Created by William Vabrinskas on 1/22/22.
//

import Foundation
import Accelerate

public extension Array where Element == [Double] {

  @inline(__always)
  var shape: [Int] {
    let rows = self.count
    let cols = self[safe: 0]?.count ?? 0
    return [cols, rows]
  }

  func zeroPad(filterSize: (Int, Int), stride: (Int, Int) = (1,1)) -> Self {
    guard let first = self.first else {
      return self
    }
    
    let height = Double(self.count)
    let width = Double(first.count)
    
    let outHeight = ceil(height / Double(stride.0))
    let outWidth = ceil(width / Double(stride.1))
    
    let padAlongHeight = Swift.max((outHeight - 1) * Double(stride.0) + Double(filterSize.0) - height, 0)
    let padAlongWidth = Swift.max((outWidth - 1) * Double(stride.1) + Double(filterSize.1) - width, 0)

    let paddingTop = Int(floor(padAlongHeight / 2))
    let paddingBottom = Int(padAlongHeight - Double(paddingTop))
    let paddingLeft = Int(floor(padAlongWidth / 2))
    let paddingRight = Int(padAlongWidth - Double(paddingLeft))
    
    var result: [Element] = self
    let newRow = [Double](repeating: 0, count: first.count)
    
    //top / bottom comes first so we can match row insertion
    for _ in 0..<paddingTop {
      result.insert(newRow, at: 0)
    }
    
    for _ in 0..<paddingBottom {
      result.append(newRow)
    }
    
    var paddingLeftMapped: [[Double]] = []
    result.forEach { v in
      var row = v
      for _ in 0..<paddingLeft {
        row.insert(0, at: 0)
      }
      paddingLeftMapped.append(row)
    }
    
    result = paddingLeftMapped
    
    var paddingRightMapped: [[Double]] = []
    
    result.forEach { v in
      var row = v
      for _ in 0..<paddingRight {
        row.append(0)
      }
      paddingRightMapped.append(row)
    }
    
    result = paddingRightMapped
    
    return result
  }
  
  func zeroPad() -> Self {
    guard let first = self.first else {
      return self
    }
    
    let result: [Element] = self

    let mapped = result.map { r -> [Double] in
      var newR: [Double] = [0]
      newR.append(contentsOf: r)
      newR.append(0)
      return newR
    }
    
    let zeros = [Double](repeating: 0, count: first.count + 2)
    var r = [zeros]
    r.append(contentsOf: mapped)
    r.append(zeros)
        
    return r
  }
  
  func flip180() -> Self {
    self.reversed().map { $0.reverse() }
  }
  
  /// Uses `vDSP_mtrans` to transpose each 2D array throughout the depth of the array
  /// - Returns: The transposed array
  func transpose() -> Self {
    
    let mShape = shape
    let row = mShape[safe: 1] ?? 0
    let col = mShape[safe: 0] ?? 0
    
    var d: [Double] = [Double](repeating: 0, count: row * col)
    let flat = flatMap { $0 }
    
    vDSP_mtransD(flat,
                vDSP_Stride(1),
                &d,
                vDSP_Stride(1),
                vDSP_Length(col),
                vDSP_Length(row))
    
    return d.reshape(columns: row) // because cols count will become rows count
  }

}

//use accelerate
public extension Array where Element == Double {
  @inline(__always)
  var sum: Element {
    vDSP.sum(self)
  }

  @inline(__always)
  var sumOfSquares: Element {
    let stride = vDSP_Stride(1)
    let n = vDSP_Length(self.count)

    var c: Element = .nan

    vDSP_svesqD(self,
                stride,
                &c,
                n)
    return c
  }

  @inline(__always)
  var indexOfMin: (UInt, Element) {
    vDSP.indexOfMinimum(self)
  }

  @inline(__always)
  var indexOfMax: (UInt, Element) {
    vDSP.indexOfMaximum(self)
  }

  @inline(__always)
  var mean: Element {
    vDSP.mean(self)
  }

  @inline(__always)
  var max: Element {
    vDSP.maximum(self)
  }

  @inline(__always)
  var min: Element {
    vDSP.minimum(self)
  }
  
  @inlinable mutating func clip(_ to: Element) {
    self = self.map { Swift.max(-to, Swift.min(to, $0)) }
  }
  
  @inlinable mutating func l1Normalize(limit: Element) {
    //normalize gradients
    let norm = self.sum
    if norm > limit {
      self = self / norm
    }
  }
  
  /// Will normalize the vector using the L2 norm to 1.0 if the sum of squares is greater than the limit
  /// - Parameter limit: the sumOfSquares limit that when reached it should normalize
  @inlinable mutating func l2Normalize(limit: Element) {
    //normalize gradients
    let norm = self.sumOfSquares
    if norm > limit {
      let length = sqrt(norm)
      self = self / length
    }
  }
  
  /// Normalizes input to have 0 mean and 1 unit standard deviation
  @discardableResult
  @inlinable mutating func normalize() -> (mean: Element, std: Element) {
    var mean: Element = 0
    var std: Element = 0
    var result: [Element] = [Element](repeating: 0, count: self.count)
    vDSP_normalizeD(self,
                   vDSP_Stride(1),
                   &result,
                   vDSP_Stride(1),
                   &mean,
                   &std,
                   vDSP_Length(self.count))
    self = result
    return (mean, std)
  }
  
  func reverse() -> Self {
    var result = self
    vDSP.reverse(&result)
    return result
  }

  mutating func fillZeros() {
    vDSP.fill(&self, with: .zero)
  }
  
  func dot(_ b: [Element]) -> Element {
    let n = vDSP_Length(self.count)
    var C: Element = .nan
    
    let aStride = vDSP_Stride(1)
    let bStride = vDSP_Stride(1)
    
    vDSP_dotprD(self,
                aStride,
                b,
                bStride,
                &C,
                n)
    
    return C
  }
  
  func multiply(B: [Element], columns: Int32, rows: Int32, dimensions: Int32 = 1) -> [Element] {
    let M = vDSP_Length(dimensions)
    let N = vDSP_Length(columns)
    let K = vDSP_Length(rows)
    
    var C: [Element] = [Element].init(repeating: 0, count: Int(N))
    
    let aStride = vDSP_Stride(1)
    let bStride = vDSP_Stride(1)
    let cStride = vDSP_Stride(1)
    
    vDSP_mmulD(self,
               aStride,
               B,
               bStride,
               &C,
               cStride,
               vDSP_Length(M),
               vDSP_Length(N),
               vDSP_Length(K))
    
    return C
  }
  
  func transpose(columns: Int, rows: Int) -> [Element] {
    var result: [Element] = [Element].init(repeating: 0, count: columns * rows)
    
    vDSP_mtransD(self,
                 vDSP_Stride(1),
                 &result,
                 vDSP_Stride(1),
                 vDSP_Length(columns),
                 vDSP_Length(rows))
    
    return result
  }
  
  @inline(__always)
  static func +(lhs: [Element], rhs: Element) -> [Element] {
    return vDSP.add(rhs, lhs)
  }

  @inline(__always)
  static func +(lhs: Element, rhs: [Element]) -> [Element] {
    return vDSP.add(lhs, rhs)
  }

  @inline(__always)
  static func +(lhs: [Element], rhs: [Element]) -> [Element] {
    return vDSP.add(rhs, lhs)
  }

  @inline(__always)
  static func -(lhs: [Element], rhs: [Element]) -> [Element] {
    return vDSP.subtract(lhs, rhs)
  }

  @inline(__always)
  static func *(lhs: [Element], rhs: Element) -> [Element] {
    return vDSP.multiply(rhs, lhs)
  }

  @inline(__always)
  static func *(lhs: Element, rhs: [Element]) -> [Element] {
    return vDSP.multiply(lhs, rhs)
  }

  @inline(__always)
  static func *(lhs: [Element], rhs: [Element]) -> [Element] {
    return vDSP.multiply(lhs, rhs)
  }

  @inline(__always)
  static func /(lhs: [Element], rhs: [Element]) -> [Element] {
    return vDSP.divide(lhs, rhs)
  }
  
  @inline(__always)
  static func /(lhs: [Element], rhs: Element) -> [Element] {
    return vDSP.divide(lhs, rhs)
  }

  @inline(__always)
  static func /(lhs: Element, rhs: [Element]) -> [Element] {
    return vDSP.divide(rhs, lhs)
  }
}

public extension Array where Element: Equatable & Numeric & FloatingPoint {
  
  var average: Element {
    let sum = self.sumSlow
    return sum / Element(self.count)
  }
  
  func scale(_ range: ClosedRange<Element> = 0...1) -> [Element] {
    let max = self.max() ?? 0
    let min = self.min() ?? 0
    let b = range.upperBound
    let a = range.lowerBound
    
    let new =  self.map { x -> Element in
      let ba = (b - a)
      let numerator = x - min
      let denominator = max - min
      
      return ba * (numerator / denominator) + a
    }
    return new
  }
  
  func scale(from range: ClosedRange<Element> = 0...1,
             to toRange: ClosedRange<Element> = 0...1) -> [Element] {
    
    let max = range.upperBound
    let min = range.lowerBound
    
    let b = toRange.upperBound
    let a = toRange.lowerBound
    
    let new =  self.map { x -> Element in
      let ba = (b - a)
      let numerator = x - min
      let denominator = max - min
      
      return ba * (numerator / denominator) + a
    }
    return new
  }
  
  static func -(lhs: [Element], rhs: Element) -> [Element] {
    return lhs.map({ $0 - rhs })
  }
  
  static func -(lhs: Element, rhs: [Element]) -> [Element] {
    return rhs.map({ lhs - $0 })
  }
}

public extension Array where Element == [[Double]] {
  
  var shape: [Int] {
    let depth = self.count
    
    let rows = self[safe: 0]?.count ?? 0
    let cols = self[safe: 0]?[safe: 0]?.count ?? 0
    
    return [cols, rows, depth]
  }
  
  /// Uses `vDSP_mtrans` to transpose each 2D array throughout the depth of the array
  /// - Returns: The transposed array
  func transpose() -> Self {
    var result: Self = []
    
    forEach { m in
      let mShape = m.shape
      let row = mShape[safe: 1] ?? 0
      let col = mShape[safe: 0] ?? 0
      
      var d: [Double] = [Double](repeating: 0, count: row * col)
      let flat = m.flatMap { $0 }
      
      vDSP_mtransD(flat,
                  vDSP_Stride(1),
                  &d,
                  vDSP_Stride(1),
                  vDSP_Length(col),
                  vDSP_Length(row))
      
      let dReshaped = d.reshape(columns: row) // because cols count will become rows count
      result.append(dReshaped)
    }

    return result
  }

  
  static func *(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    let leftShape = left.shape
    let rightShape = right.shape
    
    precondition(leftShape == rightShape)
    
    let depth = leftShape[safe: 2] ?? 0
    let rows = leftShape[safe: 1] ?? 0
    
    var result: Self = []
    for d in 0..<depth {
      var new2d: Element = []
      for r in 0..<rows {
        new2d.append(left[d][r] * right[d][r])
      }
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    let leftShape = left.shape
    let rightShape = right.shape
    
    precondition(leftShape == rightShape)
    
    let depth = leftShape[safe: 2] ?? 0
    let rows = leftShape[safe: 1] ?? 0
    
    var result: Self = []
    for d in 0..<depth {
      var new2d: Element = []
      for r in 0..<rows {
        new2d.append(left[d][r] / right[d][r])
      }
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    let leftShape = left.shape
    let rightShape = right.shape
    
    precondition(leftShape == rightShape)
    
    let depth = leftShape[safe: 2] ?? 0
    let rows = leftShape[safe: 1] ?? 0
    
    var result: Self = []
    for d in 0..<depth {
      var new2d: Element = []
      for r in 0..<rows {
        new2d.append(left[d][r] - right[d][r])
      }
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    let leftShape = left.shape
    let rightShape = right.shape
    
    precondition(leftShape == rightShape)
    
    let depth = leftShape[safe: 2] ?? 0
    let rows = leftShape[safe: 1] ?? 0
    
    var result: Self = []
    for d in 0..<depth {
      var new2d: Element = []
      for r in 0..<rows {
        new2d.append(left[d][r] + right[d][r])
      }
      result.append(new2d)
    }
    return result
  }
}

