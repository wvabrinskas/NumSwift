//
//  File.swift
//
//
//  Created by William Vabrinskas on 1/22/22.
//

import Foundation
import Accelerate

public extension Array where Element == [Float] {
  var shape: [Int] {
    let rows = self.count
    let cols = self[safe: 0]?.count ?? 0
    return [cols, rows]
  }
  
  func flatten() -> [Self.Element.Element] {
    flatMap { $0 }
  }
  
  func transpose2d() -> Self {
    let mShape = shape
    let row = mShape[safe: 1] ?? 0
    let col = mShape[safe: 0] ?? 0
    
    return NumSwiftC.tranpose(self, size: (row, col))
  }
  
  /// Uses `vDSP_mtrans` to transpose each 2D array throughout the depth of the array
  /// - Returns: The transposed array
  func transpose() -> Self {
    
    let mShape = shape
    let row = mShape[safe: 1] ?? 0
    let col = mShape[safe: 0] ?? 0
    
    var d: [Float] = [Float](repeating: 0, count: row * col)
    let flat = flatten()
    
    vDSP_mtrans(flat,
                vDSP_Stride(1),
                &d,
                vDSP_Stride(1),
                vDSP_Length(col),
                vDSP_Length(row))
    
    return d.reshape(columns: row) // because cols count will become rows count
  }

  func zeroPad(padding: NumSwiftPadding) -> Self {
    guard let first = self.first else {
      return self
    }
    
    let paddingTop = padding.top
    let paddingLeft = padding.left
    let paddingRight = padding.right
    let paddingBottom = padding.bottom
    
    var result: [Element] = self
    let newRow = [Float](repeating: 0, count: first.count)
    
    //top / bottom comes first so we can match row insertion
    for _ in 0..<paddingTop {
      result.insert(newRow, at: 0)
    }
    
    for _ in 0..<paddingBottom {
      result.append(newRow)
    }
    
    var paddingLeftMapped: [[Float]] = []
    result.forEach { v in
      var row = v
      for _ in 0..<paddingLeft {
        row.insert(0, at: 0)
      }
      paddingLeftMapped.append(row)
    }
    
    result = paddingLeftMapped
    
    var paddingRightMapped: [[Float]] = []
    
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
  
  func stridePad(strides: (rows: Int, columns: Int), shrink: Int = 0) -> Self {
    var result = stridePad(strides: strides)
    result = result.shrink(by: shrink)
    return result
  }
  
  func stridePad(strides: (rows: Int, columns: Int), padding: Int = 0) -> Self {
    var result = stridePad(strides: strides)
    
    for _ in 0..<padding {
      result = result.zeroPad()
    }
    
    return result
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
        
    let paddingObc = NumSwiftPadding(top: paddingTop,
                                     left: paddingLeft,
                                     right: paddingRight,
                                     bottom: paddingBottom)
    
    return self.zeroPad(padding: paddingObc)
  }

  func zeroPad() -> Self {
    guard let first = self.first else {
      return self
    }
    
    let result: [Element] = self

    let mapped = result.map { r -> [Float] in
      var newR: [Float] = [0]
      newR.append(contentsOf: r)
      newR.append(0)
      return newR
    }
    
    let zeros = [Float](repeating: 0, count: first.count + 2)
    var r = [zeros]
    r.append(contentsOf: mapped)
    r.append(zeros)
        
    return r
  }
  
  func shrink(by size: Int) -> Self {
    var results: [[Float]] = self
    
    for _ in 0..<size {
      var newResult: [[Float]] = []
        
      results.forEach { p in
        var newRow: [Float] = p
        newRow.removeFirst()
        newRow.removeLast()
        newResult.append(newRow)
      }
      
      results = newResult
      results.removeFirst()
      results.removeLast()
    }
    
    return results
  }
  
  func stridePad(strides: (rows: Int, columns: Int)) -> Self {
    guard let firstCount = self.first?.count else {
      return self
    }
    
    let numToPad = (strides.rows - 1, strides.columns - 1)
        
    let newRows = count + ((strides.rows - 1) * (count - 1))
    let newColumns = firstCount + ((strides.columns - 1) * (count - 1))
    
    var result: [[Float]] = NumSwift.zerosLike((rows: newRows, columns: newColumns))
    
    var mutableSelf: [Float] = self.flatten()
    if numToPad.0 > 0 || numToPad.1 > 0 {
      
      for r in stride(from: 0, to: newRows, by: strides.rows) {
        for c in stride(from: 0, to: newColumns, by: strides.columns) {
          result[r][c] = mutableSelf.removeFirst()
        }
      }
      
    } else {
      return self
    }
      
    return result
  }

  func flip180() -> Self {
    self.reversed().map { $0.reverse() }
  }
}

//use accelerate
public extension Array where Element == Float {
  var sum: Element {
    vDSP.sum(self)
  }
  
  var sumOfSquares: Element {
    let stride = vDSP_Stride(1)
    let n = vDSP_Length(self.count)
    
    var c: Element = .nan
    
    vDSP_svesq(self,
               stride,
               &c,
               n)
    return c
  }
  
  var indexOfMin: (UInt, Element) {
    vDSP.indexOfMinimum(self)
  }
  
  var indexOfMax: (UInt, Element) {
    vDSP.indexOfMaximum(self)
  }
  
  var max: Element {
    vDSP.maximum(self)
  }
  
  var min: Element {
    vDSP.minimum(self)
  }
  
  var mean: Element {
    vDSP.mean(self)
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
    var mean: Float = 0
    var std: Float = 0
    var result: [Float] = [Float](repeating: 0, count: self.count)
    vDSP_normalize(self,
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
  
  func dot(_ b: [Element]) -> Element {
    let n = vDSP_Length(self.count)
    var C: Element = .nan
    
    let aStride = vDSP_Stride(1)
    let bStride = vDSP_Stride(1)
    
    vDSP_dotpr(self,
               aStride,
               b,
               bStride,
               &C,
               n)
    
    return C
  }

  static func +(lhs: [Element], rhs: Element) -> [Element] {
    return vDSP.add(rhs, lhs)
  }
  
  static func +(lhs: Element, rhs: [Element]) -> [Element] {
    return vDSP.add(lhs, rhs)
  }
  
  static func +(lhs: [Element], rhs: [Element]) -> [Element] {
    return vDSP.add(rhs, lhs)
  }
  
  static func -(lhs: [Element], rhs: [Element]) -> [Element] {
    return vDSP.subtract(lhs, rhs)
  }
  
  static func *(lhs: [Element], rhs: Element) -> [Element] {
    return vDSP.multiply(rhs, lhs)
  }
  
  static func *(lhs: Element, rhs: [Element]) -> [Element] {
    return vDSP.multiply(lhs, rhs)
  }
  
  static func *(lhs: [Element], rhs: [Element]) -> [Element] {
    return vDSP.multiply(lhs, rhs)
  }
  
  static func /(lhs: [Element], rhs: [Element]) -> [Element] {
    return vDSP.divide(lhs, rhs)
  }
  
  static func /(lhs: [Element], rhs: Element) -> [Element] {
    return vDSP.divide(lhs, rhs)
  }
  
  static func /(lhs: Element, rhs: [Element]) -> [Element] {
    return vDSP.divide(lhs, rhs)
  }
  
}

public extension Array where Element == [[Float]] {
  var shape: [Int] {
    let depth = self.count
    
    let rows = self[safe: 0]?.count ?? 0
    let cols = self[safe: 0]?[safe: 0]?.count ?? 0
    
    return [cols, rows, depth]
  }
  
  var sumOfSquares: Float {
    var result: Float = 0
    self.forEach { a in
      a.forEach { b in
        let stride = vDSP_Stride(1)
        let n = vDSP_Length(b.count)
        
        var c: Float = .nan
        
        vDSP_svesq(b,
                   stride,
                   &c,
                   n)
        
        result += c
      }
    }

    return result
  }
  
  var mean: Float {
    var r: Float = 0
    var total = 0
    self.forEach { a in
      a.forEach { b in
        r += b.mean
        total += 1
      }
    }
    
    return r / Float(total)
  }
  
  var sum: Float {
    var r: Float = 0
    self.forEach { a in
      a.forEach { b in
        r += b.sum
      }
    }
    
    return r
  }
  
  func flatten() -> [Float] {
    flatMap { $0.flatMap { $0 } }
  }
  
  /// Uses `vDSP_mtrans` to transpose each 2D array throughout the depth of the array
  /// - Returns: The transposed array
  func transpose2d() -> Self {
    map {
      let mShape = $0.shape
      let row = mShape[safe: 1] ?? 0
      let col = mShape[safe: 0] ?? 0
      
      let dReshaped = NumSwiftC.tranpose($0, size: (row, col))
      return dReshaped
    }
  }
  
  func matmul(_ b: Self) -> Self {
    let bShape = b.shape
    let aShape = shape
    
    let bColumns = bShape[safe: 0] ?? 0
    let bRows = bShape[safe: 1] ?? 0
    let bDepth = bShape[safe: 2] ?? 0
    
    let aColumns = aShape[safe: 0] ?? 0
    let aRows = aShape[safe: 1] ?? 0
    let aDepth = aShape[safe: 2] ?? 0
    
    precondition(aColumns == bRows, "A matrix columns does not match B matrix rows")
    precondition(aDepth == bDepth, "A matrix depth does not match B matrix depth")
    
    var result: Self = []
    
    for d in 0..<aDepth {
      let cResult = NumSwiftC.matmul(self[d],
                                     b: b[d],
                                     aSize: (rows: aRows, columns: aColumns),
                                     bSize: (rows: bRows, columns: bColumns))
      result.append(cResult)
    }
    
    return result
  }

  static func *(lhs: Self, rhs: Float) -> Self {
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d = NumSwiftC.mult(lhs[d], scalar: rhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Float, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d = NumSwiftC.mult(rhs[d], scalar: lhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: Float) -> Self {
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d = NumSwiftC.divide(lhs[d], scalar: rhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Float, rhs: Self) -> Self {
    var result: Self = []
    result.reserveCapacity(rhs.count)
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d].map { lhs / $0 }
      result.append(new2d)
    }
    return result
  }

  static func +(lhs: Self, rhs: Float) -> Self {
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for d in 0..<lhs.count {
      result.append(NumSwiftC.add(lhs[d], scalar: rhs))
    }
    return result
  }

  static func +(lhs: Float, rhs: Self) -> Self {
    var result: Self = []
    result.reserveCapacity(rhs.count)
    for d in 0..<rhs.count {
      result.append(NumSwiftC.add(rhs[d], scalar: lhs))
    }
    return result
  }

  static func -(lhs: Float, rhs: Self) -> Self {
    var result: Self = []
    result.reserveCapacity(rhs.count)
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d].map { lhs - $0 }
      result.append(new2d)
    }
    return result
  }

  static func -(lhs: Self, rhs: Float) -> Self {
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for d in 0..<lhs.count {
      result.append(NumSwiftC.sub(lhs[d], scalar: rhs))
    }
    return result
  }
  
  static func *(lhs: Self, rhs: Self) -> Self {
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for i in 0..<lhs.count {
      result.append(NumSwiftC.mult(lhs[i], rhs[i]))
    }
    return result
  }

  static func /(lhs: Self, rhs: Self) -> Self {
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for i in 0..<lhs.count {
      result.append(NumSwiftC.divide(lhs[i], rhs[i]))
    }
    return result
  }

  static func -(lhs: Self, rhs: Self) -> Self {
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for i in 0..<lhs.count {
      result.append(NumSwiftC.sub(lhs[i], rhs[i]))
    }
    return result
  }

  static func +(lhs: Self, rhs: Self) -> Self {
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for i in 0..<lhs.count {
      result.append(NumSwiftC.add(lhs[i], rhs[i]))
    }
    return result
  }
  
  // MARK: - Array Broadcasting Operations
  
  static func +(lhs: Self, rhs: [Float]) -> Self {
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d = NumSwiftC.add(lhs[d], array: rhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: [Float], rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d = NumSwiftC.add(rhs[d], array: lhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: [Float]) -> Self {
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d = NumSwiftC.sub(lhs[d], array: rhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: [Float], rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d].enumerated().map { (index, value) in
        let scalarValue = lhs[safe: index] ?? 0.0
        return scalarValue - value
      }
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Self, rhs: [Float]) -> Self {
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d = NumSwiftC.mult(lhs[d], array: rhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: [Float], rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d = NumSwiftC.mult(rhs[d], array: lhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: [Float]) -> Self {
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d = NumSwiftC.divide(lhs[d], array: rhs)
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: [Float], rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d].enumerated().map { (index, value) in
        let scalarValue = lhs[safe: index] ?? 0.0
        return scalarValue / value
      }
      result.append(new2d)
    }
    
    return result
  }
}


public extension Array {
  func as3D() -> [[[Element]]] {
    let amount = count
    let result: [[[Element]]] = [[[Element]]].init(repeating: [[Element]].init(repeating: self,
                                                                               count: amount),
                                                   count: amount)
    return result
  }
  
  func as2D() -> [[Element]] {
    let amount = count
    let result: [[Element]] = [[Element]].init(repeating: self, count: amount)
    return result
  }
  
  subscript(safe safeIndex: Int, default: Element) -> Element {
    if safeIndex < 0 {
      return `default`
    }
    
    return self[safe: safeIndex] ?? `default`
  }
  
  subscript(_ multiple: [Int], default: Element) -> Self {
    var result: Self = []
    result.reserveCapacity(multiple.count)
    
    multiple.forEach { i in
      result.append(self[safe: i, `default`])
    }
    
    return result
  }
  
  subscript(range multiple: Range<Int>, default: Element) -> Self {
    var result: Self = []
    result.reserveCapacity(multiple.count)
    
    multiple.forEach { i in
      result.append(self[safe: i, `default`])
    }
    
    return result
  }

  subscript(range multiple: Range<Int>, dialate: Int, default: Element) -> Self {
    var result: Self = []
    result.reserveCapacity(multiple.count)
    
    var i: Int = 0
    var dialateTotal: Int = 0
    
    let range = [Int](multiple)

    while i < multiple.count {
      if i > 0 && dialateTotal < dialate {
        result.append(self[safe: -1, `default`])
        dialateTotal += 1
      } else {
        result.append(self[safe: range[i], `default`])
        i += 1
        dialateTotal = 0
      }
    }

    return result
  }
  
  subscript(range multiple: Range<Int>, padding: Int, dialate: Int, default: Element) -> Self {
    var result: Self = []
    //result.reserveCapacity(multiple.count)
    
    var i: Int = 0
    var dialateTotal: Int = 0
    var paddingTotal: Int = 0
    
    let range = [Int](multiple)
    
    while i < (multiple.count + padding * 4) {
      if (i > multiple.count || i == 0) && paddingTotal < padding {
        result.append(self[safe: -1, `default`])
        paddingTotal += 1
      } else if i > 0 && i < multiple.count && dialateTotal < dialate {
        result.append(self[safe: -1, `default`])
        dialateTotal += 1
      } else {
        if i < multiple.count {
          result.append(self[safe: range[i], `default`])
          paddingTotal = 0
          dialateTotal = 0
        }
        i += 1
      }
    }

    return result
  }
}

public extension Array where Element == [Float] {
  
  subscript(_ start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            default: Float) -> Self {
    var result: Self = []
    
    let defaultRow = [Float].init(repeating: `default`, count: count)
    result.append(contentsOf: self[range: start.row..<end.row, defaultRow])
    
    return result.map { $0[range: start.column..<end.column, `default`] }
  }

  subscript(flat start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            default: Float) -> [Float] {
    var result: [Float] = []
    
    let defaultRow = [Float].init(repeating: `default`, count: count)
    
    self[range: start.row..<end.row, defaultRow].forEach { row in
      let column = row[range: start.column..<end.column, `default`]
      result.append(contentsOf: column)
    }
        
    return result
  }

  subscript(_ start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            dialate: Int,
            default: Float) -> Self {
    
    var result: Self = []

    let defaultRow = [Float].init(repeating: `default`, count: count)
    result.append(contentsOf: self[range: start.row..<end.row, dialate, defaultRow])
    
    return result.map { $0[range: start.column..<end.column, dialate, `default`] }
  }
  
  subscript(_ start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            padding: Int,
            dialate: Int,
            default: Float) -> Self {
    
    var result: Self = []

    let defaultRow = [Float].init(repeating: `default`, count: count)
    result.append(contentsOf: self[range: start.row..<end.row, padding, dialate, defaultRow])
    
    return result.map { $0[range: start.column..<end.column, padding, dialate, `default`] }
  }
  
  subscript(flat start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            dialate: Int,
            default: Float) -> [Float] {
    
    var result: [Float] = []

    let defaultRow = [Float].init(repeating: `default`, count: count)
    
    self[range: start.row..<end.row, dialate, defaultRow].forEach { row in
      let column = row[range: start.column..<end.column, dialate, `default`]
      result.append(contentsOf: column)
    }
            
    return result
  }
  
  subscript(flat start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            padding: Int,
            dialate: Int,
            default: Float) -> [Float] {
    
    var result: [Float] = []

    let defaultRow = [Float].init(repeating: `default`, count: count)
    
    self[range: start.row..<end.row, padding, dialate, defaultRow].forEach { row in
      let column = row[range: start.column..<end.column, padding, dialate, `default`]
      result.append(contentsOf: column)
    }
            
    return result
  }
}


public extension Array where Element == [Float] {
  var sumOfSquares: Float {
    var result: Float = 0
    self.forEach { a in
      let stride = vDSP_Stride(1)
      let n = vDSP_Length(a.count)
      
      var c: Float = .nan
      
      vDSP_svesq(a,
                 stride,
                 &c,
                 n)
      
      result += c
    }

    return result
  }
  
  var mean: Float {
    var r: Float = 0
    var total = 0
    self.forEach { a in
      r += a.mean
      total += 1
    }
    
    return r / Float(total)
  }
  
  var sum: Float {
    var r: Float = 0
    self.forEach { a in
      r += a.sum
    }
    
    return r
  }
  
  static func *(lhs: Self, rhs: Float) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d] * rhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Float, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d] * lhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Float, rhs: Self) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = left / rhs[d]
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: Float) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d] / rhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Self, rhs: Float) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d] + rhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Float, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d] + lhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Float, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = lhs - rhs[d]
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: Float) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d] - rhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Self, rhs: Self) -> Self {
    precondition(lhs.shape == rhs.shape)
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for d in 0..<lhs.count {
      result.append(lhs[d] * rhs[d])
    }
    return result
  }

  static func /(lhs: Self, rhs: Self) -> Self {
    precondition(lhs.shape == rhs.shape)
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for d in 0..<lhs.count {
      result.append(lhs[d] / rhs[d])
    }
    return result
  }

  static func -(lhs: Self, rhs: Self) -> Self {
    precondition(lhs.shape == rhs.shape)
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for d in 0..<lhs.count {
      result.append(lhs[d] - rhs[d])
    }
    return result
  }

  static func +(lhs: Self, rhs: Self) -> Self {
    precondition(lhs.shape == rhs.shape)
    var result: Self = []
    result.reserveCapacity(lhs.count)
    for d in 0..<lhs.count {
      result.append(lhs[d] + rhs[d])
    }
    return result
  }
}

// MARK: - Flat Array Operations (Accelerate-backed)

/// Provides high-performance flat-array operations on `ContiguousArray<Float>` using Apple's Accelerate framework.
/// These avoid the overhead of nested `[[[Float]]]` arrays and enable direct SIMD/vectorized computation.
public enum NumSwiftFlat {
  
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
  
  /// Performs matrix multiplication on flat [Float] using Accelerate's vDSP_mmul.
  public static func matmul(_ a: [Float],
                            _ b: [Float],
                            aRows: Int,
                            aCols: Int,
                            bRows: Int,
                            bCols: Int) -> [Float] {
    precondition(aCols == bRows, "A columns (\(aCols)) must equal B rows (\(bRows))")
    var c = [Float](repeating: 0, count: aRows * bCols)
    
    vDSP_mmul(a, 1,
              b, 1,
              &c, 1,
              vDSP_Length(aRows),
              vDSP_Length(bCols),
              vDSP_Length(aCols))
    
    return c
  }
  
  /// Performs matrix multiplication on flat ContiguousArray<Float> using Accelerate's vDSP_mmul.
  public static func matmul(_ a: ContiguousArray<Float>,
                            _ b: ContiguousArray<Float>,
                            aRows: Int,
                            aCols: Int,
                            bRows: Int,
                            bCols: Int) -> ContiguousArray<Float> {
    precondition(aCols == bRows, "A columns (\(aCols)) must equal B rows (\(bRows))")
    var c = ContiguousArray<Float>(repeating: 0, count: aRows * bCols)
    
    a.withUnsafeBufferPointer { aBuf in
      b.withUnsafeBufferPointer { bBuf in
        c.withUnsafeMutableBufferPointer { cBuf in
          vDSP_mmul(aBuf.baseAddress!, 1,
                    bBuf.baseAddress!, 1,
                    cBuf.baseAddress!, 1,
                    vDSP_Length(aRows),
                    vDSP_Length(bCols),
                    vDSP_Length(aCols))
        }
      }
    }
    
    return c
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
  
  /// Returns the shape of a stride-padded result without actually padding.
  public static func stridePadShape(inputSize: (rows: Int, columns: Int),
                                    strides: (rows: Int, columns: Int)) -> (rows: Int, columns: Int) {
    let newRows = inputSize.rows + ((strides.rows - 1) * (inputSize.rows - 1))
    let newColumns = inputSize.columns + ((strides.columns - 1) * (inputSize.columns - 1))
    return (newRows, newColumns)
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
    var c = ContiguousArray<Float16>(repeating: 0, count: aRows * bCols)
    
    for i in 0..<aRows {
      for j in 0..<bCols {
        var sum: Float16 = 0
        for k in 0..<aCols {
          sum += a[i * aCols + k] * b[k * bCols + j]
        }
        c[i * bCols + j] = sum
      }
    }
    
    return c
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
