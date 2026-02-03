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
