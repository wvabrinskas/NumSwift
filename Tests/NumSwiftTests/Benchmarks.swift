//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/12/24.
//

import Foundation
@testable import NumSwift
@testable import NumSwiftC
import XCTest

extension XCTestCase {
  var isGithubCI: Bool {
    if let value = ProcessInfo.processInfo.environment["CI"] {
      return value == "true"
    }
    return false
  }
}

final class Benchmarks: XCTestCase {
  override func setUp() {
    super.setUp()
    continueAfterFailure = false
  }
  
  func test_speedZerosLike() {
    guard isGithubCI == false else { return }
    
    measure {
      let size = (5000, 5000)
      let _: [[Float]] = NumSwift.zerosLike(size)
    }
  }
  
}
