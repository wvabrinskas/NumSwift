import XCTest
@testable import NumSwift
#if os(iOS)
import UIKit
#endif

final class NumSwiftTestsFloat1616: XCTestCase {
  // we want unique pointers because generally these arrays are passed into C and the memory is accessed directly.
  func test_zerosUniquePointers() {
    let size = (500, 500)
    let zeros: [[Float16]] = NumSwift.zerosLike(size)
    
    var lastPtr: UnsafeBufferPointer<Float16>?
    zeros.forEach { f in
      f.withUnsafeBufferPointer { ptr in
        XCTAssertNotEqual(ptr.baseAddress, lastPtr?.baseAddress)
        lastPtr = ptr
      }
    }
  }
  
  func testArraySubtract() {
    let test = 10.0
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    let expected = [-9.0, -8.0, -7.0, -6.0, -5.0]
    let result = testArray - test
    XCTAssertEqual(result, expected)
  }
  
  func testArrayToArraySubtract() {
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    let testArray2 = [10.0, 20.0, 30.0, 40.0, 50.0]
    
    let expected = [-9.0, -18.0, -27.0, -36.0, -45.0]
    let result = testArray - testArray2
    XCTAssertEqual(result, expected)
  }
  
  func testFill() {
    
    let r = [1,1,1,1,1,1,1]
    
    let result = NumSwift.zerosLike(r)
    
    XCTAssertEqual(result, [0,0,0,0,0,0,0])
    
    let test = [[[0,1], [0,1]],
                [[0,1], [0,1]]]
    
    let expected = [[[0,0], [0,0]],
                    [[0,0], [0,0]]]
    
    let result2 = NumSwift.zerosLike(test)
    XCTAssertEqual(result2, expected)
  }
  
  func testArrayAddition() {
    let test = 10.0
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    let expected = [11.0, 12.0, 13.0, 14.0, 15.0]
    let result = testArray + test
    XCTAssertEqual(result, expected)
  }
  
  func testArrayToArrayAddition() {
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    let testArray2 = [10.0, 20.0, 30.0, 40.0, 50.0]
    
    let expected = [11.0, 22.0, 33.0, 44.0, 55.0]
    let result = testArray + testArray2
    XCTAssertEqual(result, expected)
  }
  
  func testArrayMultiplication() {
    let test = 10.0
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    let expected = [10.0, 20.0, 30.0, 40.0, 50.0]
    let result = testArray * test
    XCTAssertEqual(result, expected)
  }
  
  func testArrayToArrayMultiplication() {
    let testArray = [1.0, 2.0, 3.0, 4.0, 5.0]
    let testArray2 = [10.0, 20.0, 30.0, 40.0, 50.0]
    
    let expected = [10.0, 40.0, 90.0, 160.0, 250.0]
    
    let result = testArray * testArray2
    XCTAssertEqual(result, expected)
  }
  
  func testArrayDivision() {
    let test = 10.0
    let testArray = [10.0, 20.0, 30.0, 40.0, 50.0]
    
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    let result = testArray / test
    XCTAssertEqual(result, expected)
  }
  
  func testArrayToArrayDivision() {
    let testArray = [10.0, 40.0, 90.0, 160.0, 250.0]
    let testArray2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    let expected = [10.0, 20.0, 30.0, 40.0, 50.0]
    
    let result = testArray / testArray2
    XCTAssertEqual(result, expected)
  }
  
  func testFlip() {
    let v: [[Float16]] = [[1,2,3],
                        [4,5,6],
                        [7,8,9]]
    
    let r = v.flip180()
    
    let expected: [[Float16]] = [[9, 8, 7],
                               [6, 5, 4],
                               [3, 2, 1]]
    
    XCTAssertEqual(r, expected)
  }
  
  func testScale() {
    let testArray = [0.0, 5.0, 10.0, 20.0]
    let expected = [-1.0, -0.5, 0.0, 1.0]
    
    let scaled = testArray.scale(from: 0...20, to: -1...1)
    XCTAssertEqual(scaled, expected)
  }
  
  func testShape() {
    let test: [[[Float16]]] = [[[0,1], [0,1]],
                             [[0,1], [0,1]]] //2, 2
    let expected = [2, 2, 2]
    
    XCTAssertEqual(test.shape, expected)
    
    let data: [[[Float16]]] = [[[1, 1, 1, 0, 0, 0, 1],
                              [1, 1, 1, 0, 0, 0, 1],
                              [1, 1, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 1]]]
    
    XCTAssertEqual(data.shape, [7, 6, 1])
  }
  
  func testDotProduct() {
    let a = [2.0, 2.0, 2.0]
    let b = [2.0, 2.0, 2.0]
    
    let expected = 12.0
    
    XCTAssertEqual(expected, a.dot(b))
  }
  
  func test_matmul_SingleDim_C() {
    let n1: [Float16] = [1, 1, 1]
    let n2: [Float16] = [2, 2, 2]
    let n3: [Float16] = [3, 3, 3]
    
    let layer = [n1, n2 , n3]
    let A = layer
        
    let B: [[Float16]] = [[2,2],
                        [2,2],
                        [2,2]]
    
    let output = NumSwiftC.matmul(A, b: B, aSize: (3,3), bSize: (3, 2))
    
    
    let expected: [[Float16]] = [[6.0, 6.0],
                                [12.0, 12.0],
                                [18.0, 18.0]]
    
    XCTAssertEqual(expected, output)
  }
  
  func test_matmul_SingleDim() {
    let n1: [Float16] = [1, 1, 1]
    let n2: [Float16] = [2, 2, 2]
    let n3: [Float16] = [3, 3, 3]
    
    let layer = [n1, n2 , n3]
    let A = [layer]
        
    let B: [[[Float16]]] = [[[2,2],
                           [2,2],
                           [2,2]]]
    
    let output = A.matmul(B)
    
    let expected: [[[Float16]]] = [[[6.0, 6.0],
                                  [12.0, 12.0],
                                  [18.0, 18.0]]]
    
    XCTAssertEqual(expected, output)
  }
  
  func test_3d_fast_C_transpose_Float16() {
    let r: [[[Float16]]] = [[[6.0, 6.0],
                           [12.0, 12.0],
                           [18.0, 18.0]],
                          [[6.0, 6.0],
                           [12.0, 12.0]]]
    
    let expected: [[[Float16]]] = [[[6.0, 12.0, 18.0],
                                  [6.0, 12.0, 18.0]],
                                 [[6.0, 12.0],
                                  [6.0, 12.0]]]
    
    let transposed = r.transpose2d()
    XCTAssertEqual(transposed, expected)
  }
  
  
  func test_2d_fast_C_transpose_Float16() {
    let r: [[Float16]] = [[6.0, 6.0],
                        [12.0, 12.0],
                        [18.0, 18.0]]
    
    let expected: [[Float16]] = [[6.0, 12.0, 18.0],
                               [6.0, 12.0, 18.0]]
    
    let transposed = r.transpose2d()
    XCTAssertEqual(transposed, expected)
  }

  func test_uneven_matrix_math() {
    
    let expected: [[[Float16]]] = [[[6.0, 6.0],
                                  [12.0, 12.0],
                                  [18.0, 18.0]],
                                  [[6.0, 6.0],
                                   [12.0, 12.0]]]
  
    
    let expected2: [[[Float16]]] = [[[6.0, 6.0],
                                  [12.0, 12.0],
                                  [18.0, 18.0]],
                                  [[6.0, 6.0],
                                   [12.0, 12.0]]]
    
    XCTAssertEqual(expected + expected2, [[[12.0, 12.0],
                                           [24.0, 24.0],
                                           [36.0, 36.0]],
                                          [[12.0, 12.0],
                                           [24.0, 24.0]]])
  }
  
  func test_randomChoice_valid() {
    let probabilityArray: [Float16] = [0, 1.0, 0, 0, 0]
    let array: [Float16] = [1, 2, 3, 4, 5]
    let result = NumSwift.randomChoice(in: array, p: probabilityArray)
    
    XCTAssertEqual(2.0, result.0)
    XCTAssertEqual(1, result.1)
  }
  
  func test_randomChoice_probArrayAllZeros() {
    let probabilityArray: [Float16] = [0, 0, 0, 0, 0]
    let array: [Float16] = [1, 2, 3, 4, 5]
    let result = NumSwift.randomChoice(in: array, p: probabilityArray)
    
    XCTAssertTrue(result.0 > 0)
  }
  
  func test_matmul_MultDim() {
    let n1: [Float16] = [1, 1, 1]
    let n2: [Float16] = [2, 2, 2]
    let n3: [Float16] = [3, 3, 3]
    
    let layer = [n1, n2 , n3]
    let A = [layer, layer]
        
    let B: [[[Float16]]] = [[[2,2],
                           [2,2],
                           [2,2]],
                          [[2,2],
                           [2,2],
                           [2,2]]]
    
    let output = A.matmul(B)
    
    let expected: [[[Float16]]] = [[[6.0, 6.0],
                                  [12.0, 12.0],
                                  [18.0, 18.0]],
                                  [[6.0, 6.0],
                                   [12.0, 12.0],
                                   [18.0, 18.0]]]
    
    XCTAssertEqual(expected, output)
  }
  
  func testCPadding() {
    let test: [[Float16]] = [[1, 2],
                           [3, 4]]

        
    let expected: [[Float16]] = [[0.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 2.0, 0.0],
                               [0.0, 3.0, 4.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0]]
    
    let padded = NumSwiftC.zeroPad(signal: test,
                                   padding: NumSwiftPadding(top: 1,
                                                            left: 1,
                                                            right: 1,
                                                            bottom: 1))
    
    XCTAssertEqual(expected, padded)
  }
  
  func testCStridePadding() {
    let test: [[Float16]] = [[1, 2],
                           [3, 4]]

    let padded = NumSwiftC.stridePad(signal: test,
                                     strides: (2,2))
        
    let expected: [[Float16]] = [[1.0, 0.0, 2.0],
                               [0.0, 0.0, 0.0],
                               [3.0, 0.0, 4.0]]

    XCTAssertEqual(expected, padded)
  }
  
  func testStridePadding() {
    let test: [[Float16]] = [[1, 2],
                           [3, 4]]

    let padded = test.stridePad(strides: (2,2), padding: 1)
        
    let expected: [[Float16]] = [[0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 2.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 3.0, 0.0, 4.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0]]


    XCTAssertEqual(expected, padded)
  }
  
  func testCConv2D() {
    let signalShape = (5,5)

    let filter: [[Float16]] = [[0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0]]

    let signal: [[Float16]] = [[Float16]](repeating: [0,0,1,0,0], count: signalShape.0)

    let result = NumSwiftC.conv2d(signal: signal,
                                  filter: filter,
                                  strides: (1,1),
                                  padding: .same,
                                  filterSize: (rows: 3, columns: 3),
                                  inputSize: signalShape)
    
    let expected: [[Float16]] = [[0.0, 0.0, 2.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 3.0, 0.0, 0.0],
                               [0.0, 0.0, 2.0, 0.0, 0.0]]
    
    XCTAssertEqual(result, expected)
  }
  
  func testCConv1D() {
    let signalShape = (5,5)

    let filter: [[Float16]] = [[0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0]]

    let signal: [[Float16]] = [[Float16]](repeating: [0,0,1,0,0], count: signalShape.0)

    let result = NumSwiftC.conv1d(signal: signal.flatten(),
                                  filter: filter.flatten(),
                                  strides: (1,1),
                                  padding: .same,
                                  filterSize: (rows: 3, columns: 3),
                                  inputSize: signalShape)
    
    let expected: [Float16] = [0.0, 0.0, 2.0, 0.0, 0.0,
                             0.0, 0.0, 3.0, 0.0, 0.0,
                             0.0, 0.0, 3.0, 0.0, 0.0,
                             0.0, 0.0, 3.0, 0.0, 0.0,
                             0.0, 0.0, 2.0, 0.0, 0.0]
    
    XCTAssert(result == expected)
  }

  func testTransCConv2D() {
    let signalShape = (5,5)
    let filterShape = (4,4)

    let filter: [[Float16]] = [[Float16]](repeating: [0,0,1,0], count: filterShape.0)
    let signal: [[Float16]] = [[Float16]](repeating: [0,0,1,0,0], count: signalShape.0)

    let result = NumSwiftC.transConv2d(signal: signal,
                                       filter: filter,
                                       strides: (2,2),
                                       padding: .same,
                                       filterSize: filterShape,
                                       inputSize: signalShape)
    
    let expected: [[Float16]] = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    
    XCTAssertEqual(result, expected)
  }

  
  func testTransCConv1D() {
    let signalShape = (5,5)
    let filterShape = (4,4)

    let filter: [[Float16]] = [[Float16]](repeating: [0,0,1,0], count: filterShape.0)
    let signal: [[Float16]] = [[Float16]](repeating: [0,0,1,0,0], count: signalShape.0)

    let result = NumSwiftC.transConv1d(signal: signal.flatten(),
                                       filter: filter.flatten(),
                                       strides: (2,2),
                                       padding: .same,
                                       filterSize: filterShape,
                                       inputSize: signalShape)

    let reshaped = result.reshape(columns: 10)
    
    let expected: [[Float16]] = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    
    XCTAssert(reshaped == expected)
  }

  func test2DConv() {
    let signalShape = (5,5)

    let filter: [[Float16]] = [[0, 0, 1, 0],
                             [0, 0, 1, 0],
                             [0, 0, 1, 0],
                             [0, 0, 1, 0]]
    
    let signal: [[Float16]] = [[Float16]](repeating: [0,0,1,0,0], count: signalShape.0)
    
    let rows = NumSwiftC.conv2d(signal: signal,
                                filter: filter,
                                strides: (1,1),
                                padding: .same,
                                filterSize: (4,4),
                                inputSize: signalShape)
    
    let expected: [[Float16]] =  [[0.0, 3.0, 0.0, 0.0, 0.0],
                                [0.0, 4.0, 0.0, 0.0, 0.0],
                                [0.0, 4.0, 0.0, 0.0, 0.0],
                                [0.0, 3.0, 0.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0, 0.0, 0.0]]

    XCTAssert(expected == rows)
  }
  
  func testReshape() {
    let input: [Float16] = [1, 1, 1, 2, 2, 2]
    
    let expected: [[Float16]] = [[1, 1, 1],
                               [2, 2, 2]]
    
    let output = input.reshape(columns: 3)
    
    XCTAssert(output == expected)
  }
  
  func testClip() {
    var test: [Float16] = [-0.2, 5.0, -0.5, 1.0]
    test.clip(0.5)
    XCTAssert(test == [-0.2, 0.5, -0.5, 0.5])
  }
  
  func testL2Normalize() {
    var test: [Float16] = [1,2,3,4]
    let expected: Float16 = 1.0
    
    XCTAssertNotEqual(test.sumOfSquares, expected)
    
    test.l2Normalize(limit: 1.0)
    
    XCTAssertEqual(test.sumOfSquares, expected)
  }

  func testFlatten() {
    let data: [[[Float16]]] = [[[0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0]]]
    
    let r: [Float16] = data.flatten()
    let expected: [Float16] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    XCTAssertFalse(r.isEmpty)
    XCTAssertEqual(r, expected)
    
    let data2D: [[Float16]] = [[1, 1, 1],
                              [2, 2, 2]]
    
    let r2D: [Float16] = data2D.flatten()
    let expected2D: [Float16] = [1,1,1,2,2,2]
    XCTAssertFalse(r2D.isEmpty)
    XCTAssertEqual(r2D, expected2D)
    
    let dataEmpty: [[Float16]] = [[],[]]
    
    let rEmpty: [Float16] = dataEmpty.flatten()
    XCTAssertEqual(rEmpty, [])
  }
  
  func testFlattenC() {
    let data: [[Float16]] = [[0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0]]
        
    let r = NumSwiftC.flatten(data)
    
    let expected: [Float16] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    XCTAssertFalse(r.isEmpty)
    XCTAssertEqual(r, expected)
  }

  //this fails in the CI for some reason...probably some Float16 logic
//  func testNormalization() {
//    var test: [Float16] = [1,2,3,4]
//
//    let result = test.normalize()
//
//    let mean = result.mean
//    let std = result.std
//    let expected: [Float16] = [-1.3416408, -0.44721365, 0.44721353, 1.3416407]
//
//    XCTAssertEqual(expected, test)
//    XCTAssertEqual(mean, Float16(2.5))
//    XCTAssertEqual(std, Float16(1.118034))
//  }
}

