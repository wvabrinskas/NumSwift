import Foundation
import Accelerate

//3D

public extension Collection where Self.Element: Sequence, Element.Element: Sequence {
  func flatten() -> [Self.Element.Element.Element] {
    return self.flatMap { $0.flatMap { $0 } }
  }
}

public extension Collection {
  /// Returns the shape of an N-dimensional array, ex 3D array -> (Col, Row, Dep)
  /// Slow as it does comparisons to protocol
  var shapeOf: [Int] {
    var results: [Int] = []
    
    var currentElement: Any = self
    
    while let current = currentElement as? Array<Any> {
      results.append(current.count)
      
      if let next = current.first {
        currentElement = next
      } else {
        break
      }
    }
    
    return results.reversed()
  }
  
  func fullFlatten<T>() -> [T] {
    var results: [Any] = self as? [Any] ?? []
    
    var iterator = results.makeIterator()
    
    while results.first is Array<Any> {
      results = []
      
      while let i = iterator.next() as? Array<Any> {
        results.append(contentsOf: i)
      }
      
      iterator = results.makeIterator()
    }
    
    return results as? [T] ?? []
  }
}

public extension Collection where Self.Iterator.Element: RandomAccessCollection {

  // PRECONDITION: `self` must be rectangular, i.e. every row has equal size.
  
  /// Transposes an array. Does not use the vDSP library for fast transpose
  /// - Returns: transposed array 
  func transposed() -> [[Self.Iterator.Element.Iterator.Element]] {
    guard let firstRow = self.first else { return [] }
    return firstRow.indices.map { index in
      self.map{ $0[index] }
    }
  }
}


public extension Array {
  
  var shape: [Int] {
    return shapeOf
  }
  
  func batched(into size: Int) -> [[Element]] {
    return stride(from: 0, to: count, by: size).map {
      Array(self[$0 ..< Swift.min($0 + size, count)])
    }
  }
  
  subscript(safe safeIndex: Int) -> Element? {
    if safeIndex >= 0,
       safeIndex < self.count {
      return self[safeIndex]
    }
    
    return nil
  }
  
  func concurrentBatchedForEach(workers: Int,
                                priority: DispatchQoS.QoSClass = .default,
                                _ block: @escaping (_ elements: [Element],
                                                    _ workerIndex: Int,
                                                    _ indexRange: CountableRange<Int>,
                                                    _ processingCount: Int,
                                                    _ workerId: UUID) -> ()) {
    
    DispatchQueue.concurrentBatchedPerform(units: self.count,
                                           workers: workers,
                                           priority: priority) { range, workerIndex, count, workerId in
      block(Array(self[range]), workerIndex, range, count, workerId)
    }
  }

  func concurrentForEach(workers: Int, priority: DispatchQoS.QoSClass = .default, _ block: @escaping (_ element: Element, _ index: Int, _ processingCount: Int, _ workerId: UUID) -> ()) {
    DispatchQueue.concurrentPerform(units: self.count, workers: workers, priority: priority) { i, count, workerId in
      block(self[i], i, count, workerId)
    }
  }
  
  func concurrentForEach(workers: Int, priority: DispatchQoS.QoSClass = .default, _ block: @escaping (_ element: Element, _ index: Int) -> ()) {
    DispatchQueue.concurrentPerform(units: self.count, workers: workers, priority: priority) { i in
      block(self[i], i)
    }
  }
  
  func concurrentForEach(_ block: (_ element: Element, _ index: Int) -> ()) {
    let group = DispatchGroup()

    DispatchQueue.concurrentPerform(iterations: self.count) { i in
      group.enter()
      block(self[i], i)
      group.leave()
    }
    
    group.wait()
  }
  
  func reshape(columns: Int) -> [[Element]] {
    var twoDResult: [[Element]] = []
          
    for c in stride(from: 0, through: self.count, by: columns) {
      if c + columns <= self.count {
        let row = Array(self[c..<c + columns]) // copying to array is slow
        twoDResult.append(row)
      }
    }
    
    return twoDResult
  }
}

public extension Array where Element: Equatable & Numeric {
  var sumSlow: Element {
    return self.reduce(0, +)
  }
}

public extension Array where Element: Equatable {

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

