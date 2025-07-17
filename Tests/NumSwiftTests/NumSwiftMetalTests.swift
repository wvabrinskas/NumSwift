//
//  NumSwiftMetalTest.swift
//  NumSwift
//
//  Created by William Vabrinskas on 7/17/25.
//

import XCTest
@testable import NumSwift
@testable import NumSwiftMetal

final class NumSwiftMetalTests: XCTestCase {
  private let numSwiftMetal = NumSwiftMetal()
  
  func testMetalConv2D() {
    let signalShape = (5,5)

    let filter: [[Float]] = [[0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0]]

    let signal: [[Float]] = [[Float]](repeating: [0,0,1,0,0], count: signalShape.0)

    let result = numSwiftMetal?.conv2d(signal, filter, stride: (1,1))
//    let result = NumSwiftC.conv2d(signal: signal,
//                                  filter: filter,
//                                  strides: (1,1),
//                                  padding: .same,
//                                  filterSize: (rows: 3, columns: 3),
//                                  inputSize: signalShape)
    
    let expected: [[Float]] = [[0.0, 0.0, 2.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 2.0, 0.0, 0.0]]
    
    XCTAssertEqual(result, expected)
  }
}


