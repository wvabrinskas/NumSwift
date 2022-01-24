import Foundation

public extension Numeric where Self: FloatingPoint {
  static func *(lhs: Self, rhs: [Self]) -> [Self] {
    return rhs.map({ $0 * lhs })
  }
  
  static func +(lhs: Self, rhs: [Self]) -> [Self] {
    return rhs.map({ $0 + lhs })
  }
  
  static func -(lhs: Self, rhs: [Self]) -> [Self] {
    return rhs.map({ $0 - lhs })
  }
  
  static func /(lhs: Self, rhs: [Self]) -> [Self] {
    return rhs.map({ $0 / lhs })
  }
}

public extension Array {
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
    
    return results
  }
}

public extension Array where Element: Equatable & Numeric {
  
  var sum: Element {
    return self.reduce(0, +)
  }
  
  static func +=(lhs: inout [Element], rhs: [Element]) {
    precondition(lhs.count == rhs.count)

    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = rhs[i]
      let right = lhs[i]
      addedArray.append(left + right)
    }
      
    lhs = addedArray
  }
  
  static func +(lhs: [Element], rhs: Element) -> [Element] {
    return lhs.map({ $0 + rhs })
  }
  
  static func +(lhs: [Element], rhs: [Element]) -> [Element] {
    precondition(lhs.count == rhs.count)

    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = lhs[i]
      let right = rhs[i]
      addedArray.append(left + right)
    }
    
    return addedArray
  }
  
  static func -(lhs: [Element], rhs: [Element]) -> [Element] {
    precondition(lhs.count == rhs.count)

    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = lhs[i]
      let right = rhs[i]
      addedArray.append(left - right)
    }
    
    return addedArray
  }
  
  static func -(lhs: [Element], rhs: Element) -> [Element] {
    return lhs.map({ $0 - rhs })
  }
  
  static func *(lhs: [Element], rhs: Element) -> [Element] {
    return lhs.map({ $0 * rhs })
  }
  
  static func *(lhs: [Element], rhs: [Element]) -> [Element] {
    precondition(lhs.count == rhs.count)

    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = lhs[i]
      let right = rhs[i]
      addedArray.append(left * right)
    }
    
    return addedArray
  }
  
  static func *=(lhs: inout [Element], rhs: [Element]) {
    precondition(lhs.count == rhs.count)

    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = rhs[i]
      let right = lhs[i]
      addedArray.append(left * right)
    }
      
    lhs = addedArray
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

