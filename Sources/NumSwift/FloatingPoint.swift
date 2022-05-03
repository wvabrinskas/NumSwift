//
//  File.swift
//
//
//  Created by William Vabrinskas on 1/22/22.
//

import Foundation
import Accelerate

public extension Array where Element == [Double] {
  
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

  func transConv2d(_ filter: [[Double]],
                   strides: (rows: Int, columns: Int) = (1,1),
                   padding: NumSwift.ConvPadding = .valid,
                   filterSize: (rows: Int, columns: Int),
                   inputSize: (rows: Int, columns: Int)) -> Element {
    
    let result: Element = NumSwift.transConv2D(signal: self,
                                               filter: filter,
                                               strides: strides,
                                               padding: padding,
                                               filterSize: filterSize,
                                               inputSize: inputSize).flatten()
      
    return result
  }
  
  func flip180() -> Self {
    self.reversed().map { $0.reverse() }
  }
}

public extension Array where Element == [Float] {

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
  
  func transConv2d(_ filter: [[Float]],
                   strides: (rows: Int, columns: Int) = (1,1),
                   padding: NumSwift.ConvPadding = .valid,
                   filterSize: (rows: Int, columns: Int),
                   inputSize: (rows: Int, columns: Int)) -> Element {
    
    let result: Element = NumSwift.transConv2D(signal: self,
                                               filter: filter,
                                               strides: strides,
                                               padding: padding,
                                               filterSize: filterSize,
                                               inputSize: inputSize).flatten()
      
    return result
  }
  
  
  ///Performs a convolutional operation on a 2D array with the given filter and returns a 1D array with the results
  /// - Parameters:
  ///   - filter: Filter to apply
  ///   - padding: Zero padding applied to the input.
  ///   - filterSize: Size of the filter (rows, columns)
  ///   - inputSize: Input size (rows, columns)
  /// - Returns: 2D convolution result as a 1D array
  func conv2D(_ filter: [[Float]],
              padding: NumSwift.ConvPadding = .valid,
              filterSize: (rows: Int, columns: Int),
              inputSize: (rows: Int, columns: Int)) -> Element {
    
    let paddingNum = padding.extra(inputSize: inputSize,
                                   filterSize: filterSize,
                                   stride: (1,1))
    
    let newInputSize = ((inputSize.rows + paddingNum.0), (inputSize.columns + paddingNum.1))
    
    var signal = self
    
    if padding == .same {
      signal = signal.zeroPad(filterSize: filterSize)
    }
    
    let filterRows = filterSize.rows
    let filterColumns = filterSize.columns
    
    let flatKernel = filter.flatMap { $0 }
        
    let flat = signal.flatMap { $0 }
    
    /*
     (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1*/
    let rows = (((inputSize.rows + paddingNum.0) - (filterSize.rows - 1) - 1)) + 1
    let columns = (((inputSize.columns + paddingNum.1) - (filterSize.columns - 1) - 1)) + 1

    let outputSize = (rows, columns)
    
    let conv: [Float] = vDSP.convolve(flat,
                                      rowCount: newInputSize.0,
                                      columnCount: newInputSize.1,
                                      withKernel: flatKernel,
                                      kernelRowCount: filterRows,
                                      kernelColumnCount: filterColumns)

    //remove padded 0s
    let starting: Int = Int(ceil(sqrt(Double(conv.count)))) + 1
    let ending: Int = conv.count
    
    var results: [Float] = []
    
    for i in stride(from: starting, to: ending, by: starting - 1) {
      let slice = conv[i..<(i + outputSize.0)]
      results.append(contentsOf: slice)
      if results.count == rows * columns {
        break
      }
    }
    
    return results
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
  
  func multiply(B: [Element], columns: Int32, rows: Int32, dimensions: Int32 = 1) -> [Element] {
    let M = vDSP_Length(dimensions)
    let N = vDSP_Length(columns)
    let K = vDSP_Length(rows)
    
    var C: [Element] = [Element].init(repeating: 0, count: Int(N))
    
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
              vDSP_Length(K))
    
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
  
}

//use accelerate
public extension Array where Element == Double {
  var sum: Element {
    vDSP.sum(self)
  }
  
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
  
  var indexOfMin: (UInt, Element) {
    vDSP.indexOfMinimum(self)
  }
  
  var indexOfMax: (UInt, Element) {
    vDSP.indexOfMaximum(self)
  }
  
  var mean: Element {
    vDSP.mean(self)
  }
  
  var max: Element {
    vDSP.maximum(self)
  }
  
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


public extension Array where Element == [[Float]] {
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
  
  static func *(lhs: Self, rhs: Float) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 * rhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func /(lhs: Self, rhs: Float) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 / rhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func +(lhs: Self, rhs: Float) -> Self {
    let left = lhs
            
    var result: Self = []
    for d in 0..<lhs.count {
      let new2d: Element = left[d].map { $0 + rhs }
      result.append(new2d)
    }
    
    return result
  }
  
  static func -(lhs: Self, rhs: Float) -> Self {
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
