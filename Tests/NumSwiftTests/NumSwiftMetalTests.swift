//
//  NumSwiftMetalTest.swift
//  NumSwift
//
//  Created by William Vabrinskas on 7/17/25.
//

import XCTest
@testable import NumSwift

final class NumSwiftMetalTests: XCTestCase {
  private let numSwiftMetal = NumSwiftMetal()
  
  override func setUp() {
    super.setUp()
    numSwiftMetal?.setBackend(.metal)
  }
  
  func testMetalConv2D() {
    let signalShape = (5,5)
    

    let filter: [[Float]] = [[0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0]]

    let signal: [[Float]] = [[Float]](repeating: [0,0,1,0,0], count: signalShape.0)

    let result = numSwiftMetal?.conv2d(signal,
                                       filter,
                                       stride: (1,1),
                                       padding: .same)

    let expected: [[Float]] = [[0.0, 0.0, 2.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 2.0, 0.0, 0.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalConv2DValidPadding() {
    let signal: [[Float]] = [[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]]
    
    let filter: [[Float]] = [[1, 0],
                             [0, 1]]
    
    let result = numSwiftMetal?.conv2d(signal, filter, stride: (1,1), padding: .valid)
    
    let expected: [[Float]] = [[6, 8],
                               [12, 14]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalConv2DSamePadding() {
    let signal: [[Float]] = [[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]]
    
    let filter: [[Float]] = [[1, 0],
                             [0, 1]]
    
    let result = numSwiftMetal?.conv2d(signal, filter, stride: (1,1), padding: .same)
    
    let expected: [[Float]] = [[6.0, 8.0, 3.0],
                               [12.0, 14.0, 6.0],
                               [7.0, 8.0, 9.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalConv2DFloat16ValidPadding() {
    let signal: [[Float16]] = [[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]]
    
    let filter: [[Float16]] = [[1, 0],
                               [0, 1]]
    
    let result = numSwiftMetal?.conv2d(signal, filter, stride: (1,1), padding: .valid)
    
    let expected: [[Float16]] = [[6, 8],
                                 [12, 14]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalConv2DFloat16SamePadding() {
    let signal: [[Float16]] = [[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]]
    
    let filter: [[Float16]] = [[1, 0],
                               [0, 1]]
    
    let result = numSwiftMetal?.conv2d(signal, filter, stride: (1,1), padding: .same)
    
    let expected: [[Float16]] = [[6.0, 8.0, 3.0],
                                 [12.0, 14.0, 6.0],
                                 [7.0, 8.0, 9.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Sum Tests
  
  func testMetalSumFloat() {
    let array: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
    let result = numSwiftMetal?.sum(array)
    let expected: Float = 15.0
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalSumFloat16() {
    let array: [Float16] = [1.0, 2.0, 3.0, 4.0, 5.0]
    let result = numSwiftMetal?.sum(array)
    let expected: Float16 = 15.0
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalSumEmptyArray() {
    let array: [Float] = []
    let result = numSwiftMetal?.sum(array)
    let expected: Float = 0.0
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalSumSingleElement() {
    let array: [Float] = [42.0]
    let result = numSwiftMetal?.sum(array)
    let expected: Float = 42.0
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Max Tests
  
  func testMetalMaxFloat() {
    let array: [Float] = [1.0, 5.0, 3.0, 2.0, 4.0]
    let result = numSwiftMetal?.max(array)
    let expected: Float = 5.0
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMaxFloat16() {
    let array: [Float16] = [1.0, 5.0, 3.0, 2.0, 4.0]
    let result = numSwiftMetal?.max(array)
    let expected: Float16 = 5.0
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMaxNegativeNumbers() {
    let array: [Float] = [-5.0, -1.0, -3.0, -2.0, -4.0]
    let result = numSwiftMetal?.max(array)
    let expected: Float = -1.0
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMaxSingleElement() {
    let array: [Float] = [42.0]
    let result = numSwiftMetal?.max(array)
    let expected: Float = 42.0
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Min Tests
  
  func testMetalMinFloat() {
    let array: [Float] = [1.0, 5.0, 3.0, 2.0, 4.0]
    let result = numSwiftMetal?.min(array)
    let expected: Float = 1.0
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMinNegativeNumbers() {
    let array: [Float] = [-5.0, -1.0, -3.0, -2.0, -4.0]
    let result = numSwiftMetal?.min(array)
    let expected: Float = -5.0
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMinSingleElement() {
    let array: [Float] = [42.0]
    let result = numSwiftMetal?.min(array)
    let expected: Float = 42.0
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Element-wise Addition Tests
  
  func testMetalAddFloat() {
    let lhs: [Float] = [1.0, 2.0, 3.0, 4.0]
    let rhs: [Float] = [5.0, 6.0, 7.0, 8.0]
    let result = numSwiftMetal?.add(lhs, rhs)
    let expected: [Float] = [6.0, 8.0, 10.0, 12.0]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalAddFloat16() {
    let lhs: [Float16] = [1.0, 2.0, 3.0, 4.0]
    let rhs: [Float16] = [5.0, 6.0, 7.0, 8.0]
    let result = numSwiftMetal?.add(lhs, rhs)
    let expected: [Float16] = [6.0, 8.0, 10.0, 12.0]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalAddEmptyArrays() {
    let lhs: [Float] = []
    let rhs: [Float] = []
    let result = numSwiftMetal?.add(lhs, rhs)
    let expected: [Float] = []
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalAddSingleElement() {
    let lhs: [Float] = [10.0]
    let rhs: [Float] = [5.0]
    let result = numSwiftMetal?.add(lhs, rhs)
    let expected: [Float] = [15.0]
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Element-wise Subtraction Tests
  
  func testMetalSubtractFloat() {
    let lhs: [Float] = [10.0, 8.0, 6.0, 4.0]
    let rhs: [Float] = [5.0, 3.0, 2.0, 1.0]
    let result = numSwiftMetal?.subtract(lhs, rhs)
    let expected: [Float] = [5.0, 5.0, 4.0, 3.0]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalSubtractNegativeResult() {
    let lhs: [Float] = [1.0, 2.0, 3.0]
    let rhs: [Float] = [5.0, 6.0, 7.0]
    let result = numSwiftMetal?.subtract(lhs, rhs)
    let expected: [Float] = [-4.0, -4.0, -4.0]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalSubtractSingleElement() {
    let lhs: [Float] = [10.0]
    let rhs: [Float] = [3.0]
    let result = numSwiftMetal?.subtract(lhs, rhs)
    let expected: [Float] = [7.0]
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Element-wise Multiplication Tests
  
  func testMetalMultiplyFloat() {
    let lhs: [Float] = [2.0, 3.0, 4.0, 5.0]
    let rhs: [Float] = [6.0, 7.0, 8.0, 9.0]
    let result = numSwiftMetal?.multiply(lhs, rhs)
    let expected: [Float] = [12.0, 21.0, 32.0, 45.0]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMultiplyFloat16() {
    let lhs: [Float16] = [2.0, 3.0, 4.0, 5.0]
    let rhs: [Float16] = [6.0, 7.0, 8.0, 9.0]
    let result = numSwiftMetal?.multiply(lhs, rhs)
    let expected: [Float16] = [12.0, 21.0, 32.0, 45.0]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMultiplyByZero() {
    let lhs: [Float] = [2.0, 3.0, 4.0, 5.0]
    let rhs: [Float] = [0.0, 0.0, 0.0, 0.0]
    let result = numSwiftMetal?.multiply(lhs, rhs)
    let expected: [Float] = [0.0, 0.0, 0.0, 0.0]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMultiplySingleElement() {
    let lhs: [Float] = [7.0]
    let rhs: [Float] = [3.0]
    let result = numSwiftMetal?.multiply(lhs, rhs)
    let expected: [Float] = [21.0]
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Element-wise Division Tests
  
  func testMetalDivideFloat() {
    let lhs: [Float] = [12.0, 21.0, 32.0, 45.0]
    let rhs: [Float] = [3.0, 7.0, 8.0, 9.0]
    let result = numSwiftMetal?.divide(lhs, rhs)
    let expected: [Float] = [4.0, 3.0, 4.0, 5.0]
    
    XCTAssertNotNil(result)
    
    for i in result!.enumerated() {
      XCTAssertEqual(abs(result![i.0] - expected[i.0]), 0.0, accuracy: 0.0001)
    }
  }
  
  func testMetalDivideByOne() {
    let lhs: [Float] = [5.0, 10.0, 15.0, 20.0]
    let rhs: [Float] = [1.0, 1.0, 1.0, 1.0]
    let result = numSwiftMetal?.divide(lhs, rhs)
    let expected: [Float] = [5.0, 10.0, 15.0, 20.0]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalDivideSingleElement() {
    let lhs: [Float] = [15.0]
    let rhs: [Float] = [3.0]
    let result = numSwiftMetal?.divide(lhs, rhs)
    let expected: [Float] = [5.0]
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Matrix Multiplication Tests
  
  func testMetalMatmulFloat() {
    let a: [[Float]] = [[1.0, 2.0],
                        [3.0, 4.0]]
    let b: [[Float]] = [[5.0, 6.0],
                        [7.0, 8.0]]
    let result = numSwiftMetal?.matmul(a, b)
    let expected: [[Float]] = [[19.0, 22.0],
                               [43.0, 50.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMatmulFloat16() {
    let a: [[Float16]] = [[1.0, 2.0],
                          [3.0, 4.0]]
    let b: [[Float16]] = [[5.0, 6.0],
                          [7.0, 8.0]]
    let result = numSwiftMetal?.matmul(a, b)
    let expected: [[Float16]] = [[19.0, 22.0],
                                 [43.0, 50.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMatmulNonSquareMatrices() {
    let a: [[Float]] = [[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]
    let b: [[Float]] = [[7.0, 8.0],
                        [9.0, 10.0],
                        [11.0, 12.0]]
    let result = numSwiftMetal?.matmul(a, b)
    let expected: [[Float]] = [[58.0, 64.0],
                               [139.0, 154.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMatmulIdentityMatrix() {
    let a: [[Float]] = [[1.0, 2.0],
                        [3.0, 4.0]]
    let identity: [[Float]] = [[1.0, 0.0],
                               [0.0, 1.0]]
    let result = numSwiftMetal?.matmul(a, identity)
    let expected: [[Float]] = [[1.0, 2.0],
                               [3.0, 4.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalMatmulSingleElement() {
    let a: [[Float]] = [[5.0]]
    let b: [[Float]] = [[3.0]]
    let result = numSwiftMetal?.matmul(a, b)
    let expected: [[Float]] = [[15.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Backend Selection Tests
  
  func testSetBackendCPU() {
    numSwiftMetal?.setBackend(.cpu)
    
    // Test that small arrays use CPU backend
    let smallArray: [Float] = [1.0, 2.0, 3.0]
    let result = numSwiftMetal?.sum(smallArray)
    XCTAssertEqual(result, 6.0)
  }
    
  func testLargeArrays() {
    let size = 5000
    let lhs: [Float] = Array(repeating: 2.0, count: size)
    let rhs: [Float] = Array(repeating: 3.0, count: size)
    
    let result = numSwiftMetal?.add(lhs, rhs)
    let expected: [Float] = Array(repeating: 5.0, count: size)
    
    XCTAssertEqual(result, expected)
  }
  
  func testPerformanceWithLargeArrays() {
    let size = 10000
    let array: [Float] = (1...size).map { Float($0) }
    
    measure {
      let _ = numSwiftMetal?.sum(array)
    }
  }
  
  // MARK: - Conv2D Additional Tests
  
  func testMetalConv2DStride2() {
    let signal: [[Float]] = [[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]
    
    let filter: [[Float]] = [[1, 0],
                             [0, 1]]
    
    let result = numSwiftMetal?.conv2d(signal, filter, stride: (2, 2), padding: .valid)
    
    let expected: [[Float]] = [[7, 11],
                               [23, 27]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalConv2DLargeFilter() {
    let signal: [[Float]] = [[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15],
                             [16, 17, 18, 19, 20],
                             [21, 22, 23, 24, 25]]
    
    let filter: [[Float]] = [[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]]
    
    let result = numSwiftMetal?.conv2d(signal, filter, stride: (1, 1), padding: .valid)
    
    let expected: [[Float]] = [[63, 72, 81],
                               [108, 117, 126],
                               [153, 162, 171]]
    
    XCTAssertEqual(result, expected)
  }
  
  // MARK: - Transposed Convolution (Deconvolution) Tests
  
  func testMetalTransConv2DFloat() {
    let signal: [[Float]] = [[1, 2],
                             [3, 4]]
    
    let filter: [[Float]] = [[1, 0],
                             [0, 1]]
    
    let result = numSwiftMetal?.transconv2d(signal, filter, stride: (1, 1), padding: .valid)
    
    // For transposed convolution with stride 1 and 2x2 filter on 2x2 input, output should be 3x3
    let expected: [[Float]] = [[1, 2, 0],
                               [3, 5, 2],
                               [0, 3, 4]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalTransConv2DFloat16() {
    let signal: [[Float16]] = [[1, 2],
                               [3, 4]]
    
    let filter: [[Float16]] = [[1, 0],
                               [0, 1]]
    
    let result = numSwiftMetal?.transconv2d(signal, filter, stride: (1, 1), padding: .valid)
    
    let expected: [[Float16]] = [[1, 2, 0],
                                 [3, 5, 2],
                                 [0, 3, 4]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalTransConv2DStride2() {
    let signal: [[Float]] = [[1, 2],
                             [3, 4]]
    
    let filter: [[Float]] = [[1, 1],
                             [1, 1]]
    
    let result = numSwiftMetal?.transconv2d(signal, filter, stride: (2, 2), padding: .valid)
    
    let expected: [[Float]] = [[1, 1, 2, 2],
                               [1, 1, 2, 2],
                               [3, 3, 4, 4],
                               [3, 3, 4, 4]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalTransConv2DSamePadding() {
    let signal: [[Float]] = [[1, 2],
                             [3, 4]]
    
    let filter: [[Float]] = [[1, 1],
                             [1, 1]]
    
    let result = numSwiftMetal?.transconv2d(signal, filter, stride: (1, 1), padding: .same)
    
    let expected: [[Float]] = [[1, 3],
                               [4, 10]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testMetalTransConv2DCompareWithCPU() {
    let signalShape = (3, 3)
    let filterShape = (2, 2)
    
    let filter: [[Float]] = [[1, 0],
                             [0, 1]]
    let signal: [[Float]] = [[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]]
    
    let metalResult = numSwiftMetal?.transconv2d(signal, filter, stride: (1, 1), padding: .valid)
    
    let cpuResult = NumSwiftC.transConv2d(signal: signal,
                                          filter: filter,
                                          strides: (1, 1),
                                          padding: .valid,
                                          filterSize: filterShape,
                                          inputSize: signalShape)
    
    XCTAssertEqual(metalResult, cpuResult)
  }
}


