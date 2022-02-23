import Foundation
import Accelerate

extension Collection where Self.Iterator.Element: RandomAccessCollection {
  // PRECONDITION: `self` must be rectangular, i.e. every row has equal size.
  
  /// Transposes an array. Does not use the vDSP library for fast transpose
  /// - Returns: transposed array 
  func transposed() -> [[Self.Iterator.Element.Iterator.Element]] {
    guard let firstRow = self.first else { return [] }
    return firstRow.indices.map { index in
      self.map{ $0[index] }
    }
  }
  
  func transposed(rotate: Bool) {
    
  }
}


public extension Array {
  subscript(safe safeIndex: Int) -> Element? {
    if safeIndex < self.count {
      return self[safeIndex]
    }
    
    return nil
  }
  
  /// Returns the shape of an N-dimensional array, ex 3D array -> (Col, Row, Dep)
  var shape: [Int] {
    var results: [Int] = []
    
    var currentElement: Any = self
    
    while let current = currentElement as? Array<Any> {
      
      if let currentArray = current as? Array<Array<Any>> {
        if let first = currentArray.first,
           currentArray.allSatisfy({ $0.count == first.count }) == false {
          fatalError("ERROR: Using shape on an array where all elements arent the same length is not allowed.")
          break
        }
      }
      
      results.append(current.count)
      
      if let next = current.first {
        currentElement = next
      } else {
        break
      }
    }
    
    return results.reversed()
  }
  
  
  func reshape(columns: Int) -> [[Element]] {
    var twoDResult: [[Element]] = []
    var oneDResult: [Element] = []
      
    for i in 0..<count {
      oneDResult.append(self[i])
      if (i + 1) % columns == 0 {
        twoDResult.append(oneDResult)
        oneDResult = []
      }
    }
    
    return twoDResult
  }
  
}

public extension Array where Element: Equatable & Numeric {
  var sum: Element {
    return self.reduce(0, +)
  }
}

public extension Array where Element: Equatable {
  
  func batched(into size: Int) -> [[Element]] {
    return stride(from: 0, to: count, by: size).map {
      Array(self[$0 ..< Swift.min($0 + size, count)])
    }
  }
  /// Get a copy of self but with randomized data indexes
  /// - Returns: Returns Self but with the data randomized
  func randomize() -> Self {
    var arrayCopy = self
    var randomArray: [Element] = []
    
    for _ in 0..<self.count {
      guard let element = arrayCopy.randomElement() else {
        break
      }
      randomArray.append(element)
      
      if let index = arrayCopy.firstIndex(of: element) {
        arrayCopy.remove(at: index)
      }
    }
    
    return randomArray
  }

}

