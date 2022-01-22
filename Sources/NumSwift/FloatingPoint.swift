//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/22/22.
//

import Foundation

public extension Array where Element: Equatable & Numeric & FloatingPoint {
  
  var average: Element {
    let sum = self.sum
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
  
  static func /(lhs: [Element], rhs: [Element]) -> [Element] {
    precondition(lhs.count == rhs.count)
    
    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = lhs[i]
      let right = rhs[i]
      addedArray.append(left / right)
    }
    
    return addedArray
  }
  
  static func /(lhs: [Element], rhs: Element) -> [Element] {
    return lhs.map({ $0 / rhs })
  }
}
