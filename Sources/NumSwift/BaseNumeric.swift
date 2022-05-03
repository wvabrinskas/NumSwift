import Foundation
import Accelerate

@propertyWrapper
public struct Atomic<Value> {
  private let lock = NSLock()
  private var value: Value
  
  public init(default: Value) {
    self.value = `default`
  }
  
  public var wrappedValue: Value {
    get {
      lock.lock()
      defer { lock.unlock() }
      return value
    }
    set {
      lock.lock()
      value = newValue
      lock.unlock()
    }
  }
}

public extension Collection {
  /// Returns the shape of an N-dimensional array, ex 3D array -> (Col, Row, Dep)
  var shape: [Int] {
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
  
  func flatten<T>() -> [T] {
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
  
  func batched(into size: Int) -> [[Element]] {
    return stride(from: 0, to: count, by: size).map {
      Array(self[$0 ..< Swift.min($0 + size, count)])
    }
  }
  
  subscript(safe safeIndex: Int) -> Element? {
    if safeIndex < self.count {
      return self[safeIndex]
    }
    
    return nil
  }

  func concurrentForEach(workers: Int, _ block: @escaping (_ element: Element, _ index: Int) -> ()) {
    DispatchQueue.concurrentPerform(units: self.count, workers: workers) { i in
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
        let row = Array(self[c..<c + columns])
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

