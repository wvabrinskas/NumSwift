//
//  File.swift
//
//
//  Created by William Vabrinskas on 1/22/22.
//

import Foundation
import Accelerate

public extension Array where Element == [Float16] {
  var shape: [Int] {
    let rows = self.count
    let cols = self[safe: 0]?.count ?? 0
    return [cols, rows]
  }
  
  func flatten(inputSize: (rows: Int, columns: Int)? = nil) -> [Self.Element.Element] {
    NumSwiftC.flatten(self, inputSize: inputSize)
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
    
    var d: [Float16] = [Float16](repeating: 0, count: row * col)
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
    let newRow = [Float16](repeating: 0, count: first.count)
    
    //top / bottom comes first so we can match row insertion
    for _ in 0..<paddingTop {
      result.insert(newRow, at: 0)
    }
    
    for _ in 0..<paddingBottom {
      result.append(newRow)
    }
    
    var paddingLeftMapped: [[Float16]] = []
    result.forEach { v in
      var row = v
      for _ in 0..<paddingLeft {
        row.insert(0, at: 0)
      }
      paddingLeftMapped.append(row)
    }
    
    result = paddingLeftMapped
    
    var paddingRightMapped: [[Float16]] = []
    
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

    let mapped = result.map { r -> [Float16] in
      var newR: [Float16] = [0]
      newR.append(contentsOf: r)
      newR.append(0)
      return newR
    }
    
    let zeros = [Float16](repeating: 0, count: first.count + 2)
    var r = [zeros]
    r.append(contentsOf: mapped)
    r.append(zeros)
        
    return r
  }
  
  func shrink(by size: Int) -> Self {
    var results: [[Float16]] = self
    
    for _ in 0..<size {
      var newResult: [[Float16]] = []
        
      results.forEach { p in
        var newRow: [Float16] = p
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
    
    var result: [[Float16]] = NumSwift.zerosLike((rows: newRows, columns: newColumns))
    
    var mutableSelf: [Float16] = self.flatten()
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
public extension Array where Element == Float16 {
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
  
  mutating func fillZeros() {
    vDSP.fill(&self, with: .zero)
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
  
  func multiply(B: [Element], columns: Int32, rows: Int32, dimensions: Int32 = 1) -> [Element] {
    let M = vDSP_Length(dimensions)
    let N = vDSP_Length(columns)
    let P = vDSP_Length(rows)
    
    var C: [Element] = [Element].init(repeating: 0, count: Int(N * M))
    
    let aStride = vDSP_Stride(1)
    let bStride = vDSP_Stride(1)
    let cStride = vDSP_Stride(1)
    
    vDSP_mmul(self,
              aStride,
              B,
              bStride,
              &C,
              cStride,
              vDSP_Length(M),
              vDSP_Length(N),
              vDSP_Length(P))
    
    return C
  }
  
  func transpose(columns: Int, rows: Int) -> [Element] {
    var result: [Element] = [Element].init(repeating: 0, count: columns * rows)
    
    vDSP_mtrans(self,
                vDSP_Stride(1),
                &result,
                vDSP_Stride(1),
                vDSP_Length(columns),
                vDSP_Length(rows))
    
    return result
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

public extension Array where Element == [[Float16]] {
  var shape: [Int] {
    let depth = self.count
    
    let rows = self[safe: 0]?.count ?? 0
    let cols = self[safe: 0]?[safe: 0]?.count ?? 0
    
    return [cols, rows, depth]
  }
  
  var sumOfSquares: Float16 {
    var result: Float16 = 0
    self.forEach { a in
      a.forEach { b in
        let stride = vDSP_Stride(1)
        let n = vDSP_Length(b.count)
        
        var c: Float16 = .nan
        
        vDSP_svesq(b,
                   stride,
                   &c,
                   n)
        
        result += c
      }
    }

    return result
  }
  
  var mean: Float16 {
    var r: Float16 = 0
    var total = 0
    self.forEach { a in
      a.forEach { b in
        r += b.mean
        total += 1
      }
    }
    
    return r / Float16(total)
  }
  
  var sum: Float16 {
    var r: Float16 = 0
    self.forEach { a in
      a.forEach { b in
        r += b.sum
      }
    }
    
    return r
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
  
  /// Uses `vDSP_mtrans` to transpose each 2D array throughout the depth of the array
  /// - Returns: The transposed array
  func transpose() -> Self {
    var result: Self = []
    
    forEach { m in
      let mShape = m.shape
      let row = mShape[safe: 1] ?? 0
      let col = mShape[safe: 0] ?? 0
      
      var d: [Float16] = [Float16](repeating: 0, count: row * col)
      let flat = m.flatten()
      
      vDSP_mtrans(flat,
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
  
  static func *(lhs: Self, rhs: [Float16]) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 * rhs[d] }
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Self, rhs: [Float16]) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 + rhs[d] }
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: [Float16]) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 / rhs[d] }
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: [Float16]) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 - rhs[d] }
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Self, rhs: Float16) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 * rhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Float16, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d].map { $0 * lhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: Float16) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 / rhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Float16, rhs: Self) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d].map { left / $0 }
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Self, rhs: Float16) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 + rhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Float16, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d].map { $0 + lhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Float16, rhs: Self) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d].map { left - $0 }
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: Float16) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 - rhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    var result: Self = []
    for d in 0..<lhs.count {
      var new2d: Element = []
      for r in 0..<lhs[d].count {
        new2d.append(left[d][r] * right[d][r])
      }
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    var result: Self = []
    for d in 0..<lhs.count {
      var new2d: Element = []
      for r in 0..<lhs[d].count {
        new2d.append(left[d][r] / right[d][r])
      }
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs

    var result: Self = []
    for d in 0..<lhs.count {
      var new2d: Element = []
      for r in 0..<lhs[d].count {
        new2d.append(left[d][r] - right[d][r])
      }
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    var result: Self = []
    for d in 0..<lhs.count {
      var new2d: Element = []
      for r in 0..<lhs[d].count {
        new2d.append(left[d][r] + right[d][r])
      }
      result.append(new2d)
    }
    return result
  }
}

public extension Array where Element == [Float16] {
  
  subscript(_ start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            default: Float16) -> Self {
    var result: Self = []
    
    let defaultRow = [Float16].init(repeating: `default`, count: count)
    result.append(contentsOf: self[range: start.row..<end.row, defaultRow])
    
    return result.map { $0[range: start.column..<end.column, `default`] }
  }

  subscript(flat start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            default: Float16) -> [Float16] {
    var result: [Float16] = []
    
    let defaultRow = [Float16].init(repeating: `default`, count: count)
    
    self[range: start.row..<end.row, defaultRow].forEach { row in
      let column = row[range: start.column..<end.column, `default`]
      result.append(contentsOf: column)
    }
        
    return result
  }

  subscript(_ start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            dialate: Int,
            default: Float16) -> Self {
    
    var result: Self = []

    let defaultRow = [Float16].init(repeating: `default`, count: count)
    result.append(contentsOf: self[range: start.row..<end.row, dialate, defaultRow])
    
    return result.map { $0[range: start.column..<end.column, dialate, `default`] }
  }
  
  subscript(_ start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            padding: Int,
            dialate: Int,
            default: Float16) -> Self {
    
    var result: Self = []

    let defaultRow = [Float16].init(repeating: `default`, count: count)
    result.append(contentsOf: self[range: start.row..<end.row, padding, dialate, defaultRow])
    
    return result.map { $0[range: start.column..<end.column, padding, dialate, `default`] }
  }
  
  subscript(flat start: (row: Int, column: Int),
            end: (row: Int, column: Int),
            dialate: Int,
            default: Float16) -> [Float16] {
    
    var result: [Float16] = []

    let defaultRow = [Float16].init(repeating: `default`, count: count)
    
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
            default: Float16) -> [Float16] {
    
    var result: [Float16] = []

    let defaultRow = [Float16].init(repeating: `default`, count: count)
    
    self[range: start.row..<end.row, padding, dialate, defaultRow].forEach { row in
      let column = row[range: start.column..<end.column, padding, dialate, `default`]
      result.append(contentsOf: column)
    }
            
    return result
  }
}


public extension Array where Element == [Float16] {
  var sumOfSquares: Float16 {
    var result: Float16 = 0
    self.forEach { a in
      let stride = vDSP_Stride(1)
      let n = vDSP_Length(a.count)
      
      var c: Float16 = .nan
      
      vDSP_svesq(a,
                 stride,
                 &c,
                 n)
      
      result += c
    }

    return result
  }
  
  var mean: Float16 {
    var r: Float16 = 0
    var total = 0
    self.forEach { a in
      r += a.mean
      total += 1
    }
    
    return r / Float16(total)
  }
  
  var sum: Float16 {
    var r: Float16 = 0
    self.forEach { a in
      r += a.sum
    }
    
    return r
  }
  
  static func *(lhs: Self, rhs: Float16) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d] * rhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Float16, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d] * lhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Float16, rhs: Self) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = left / rhs[d]
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: Float16) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d] / rhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Self, rhs: Float16) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d] + rhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Float16, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = rhs[d] + lhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Float16, rhs: Self) -> Self {
    var result: Self = []
    for d in 0..<rhs.count {
      let new2d: Element = lhs - rhs[d]
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: Float16) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d] - rhs
      result.append(new2d)
    }
    
    return result
  }
  
  static func *(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    let leftShape = left.shape
    let rightShape = right.shape
    
    precondition(leftShape == rightShape)
    
    let depth = left.count
    
    var result: Self = []
    for d in 0..<depth {
      result.append(left[d] * right[d])
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    let leftShape = left.shape
    let rightShape = right.shape
    
    precondition(leftShape == rightShape)
    
    let depth = left.count
    
    var result: Self = []
    for d in 0..<depth {
      result.append(left[d] / right[d])
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    let leftShape = left.shape
    let rightShape = right.shape
    
    precondition(leftShape == rightShape)
    
    let depth = left.count
    
    var result: Self = []
    for d in 0..<depth {
      result.append(left[d] - right[d])
    }
    
    return result
  }
  
  static func +(lhs: Self, rhs: Self) -> Self {
    let left = lhs
    let right = rhs
    
    let leftShape = left.shape
    let rightShape = right.shape
    
    precondition(leftShape == rightShape)
    
    let depth = left.count
    
    var result: Self = []
    for d in 0..<depth {
      result.append(left[d] + right[d])
    }
    
    return result
  }
}
