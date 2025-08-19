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
  
  func test_speedConv2D() {
    guard isGithubCI == false else { return }
    
    let signalShape = (256,256)

    let filter: [[Float]] = [[0, 0, 1],
                             [0, 0, 1],
                             [0, 0, 1]]
    
    let signal: [[Float]] = NumSwift.onesLike(signalShape)
    
    measure {
      let _ = NumSwiftC.conv2d(signal: signal,
                                  filter: filter,
                                  strides: (1,1),
                                  padding: .same,
                                  filterSize: (3,3),
                                  inputSize: signalShape)
    }
  }
  
  func test_speedMatmul() {
    guard isGithubCI == false else { return }
    
    // Test with 128x128 matrices to trigger Winograd optimization
    let a: [[Float]] = NumSwift.onesLike((128, 128))
    let b: [[Float]] = NumSwift.onesLike((128, 128))
    
    measure {
      let _ = NumSwiftC.matmul(a, b: b, aSize: (128, 128), bSize: (128, 128))
    }
  }
  
  func test_speedMatmulSmall() {
    guard isGithubCI == false else { return }
    
    // Test with 3x3 matrices - should use standard multiplication
    let a: [[Float]] = NumSwift.onesLike((3, 3))
    let b: [[Float]] = NumSwift.onesLike((3, 3))
    
    measure {
      let _ = NumSwiftC.matmul(a, b: b, aSize: (3, 3), bSize: (3, 3))
    }
  }
  
}
