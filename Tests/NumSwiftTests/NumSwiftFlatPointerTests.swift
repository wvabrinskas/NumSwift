import XCTest
@testable import NumSwift

final class NumSwiftFlatPointerTests: XCTestCase {

  // MARK: - Float Element-wise Arithmetic

  func testAddFloat() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let b: [Float] = [10, 20, 30, 40, 50]
    let expected = ContiguousArray<Float>(a) + ContiguousArray<Float>(b)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.add(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
        }
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testSubFloat() {
    let a: [Float] = [10, 20, 30, 40, 50]
    let b: [Float] = [1, 2, 3, 4, 5]
    let expected = ContiguousArray<Float>(a) - ContiguousArray<Float>(b)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.sub(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
        }
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testMulFloat() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let b: [Float] = [2, 3, 4, 5, 6]
    let expected = ContiguousArray<Float>(a) * ContiguousArray<Float>(b)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.mul(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
        }
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testDivFloat() {
    let a: [Float] = [10, 20, 30, 40, 50]
    let b: [Float] = [2, 4, 5, 8, 10]
    let expected = ContiguousArray<Float>(a) / ContiguousArray<Float>(b)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.div(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
        }
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  // MARK: - Float Scalar Arithmetic

  func testAddScalarFloat() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let scalar: Float = 10
    let expected = ContiguousArray<Float>(a) + scalar

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.add(aPtr.baseAddress!, scalar: scalar, result: rPtr.baseAddress!, count: a.count)
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testSubScalarFloat() {
    let a: [Float] = [10, 20, 30, 40, 50]
    let scalar: Float = 5
    let expected = ContiguousArray<Float>(a) - scalar

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.sub(aPtr.baseAddress!, scalar: scalar, result: rPtr.baseAddress!, count: a.count)
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testSubScalarFromArrayFloat() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let scalar: Float = 10
    let expected = scalar - ContiguousArray<Float>(a)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.sub(scalar: scalar, aPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testMulScalarFloat() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let scalar: Float = 3
    let expected = ContiguousArray<Float>(a) * scalar

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.mul(aPtr.baseAddress!, scalar: scalar, result: rPtr.baseAddress!, count: a.count)
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testDivScalarFloat() {
    let a: [Float] = [10, 20, 30, 40, 50]
    let scalar: Float = 5
    let expected = ContiguousArray<Float>(a) / scalar

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.div(aPtr.baseAddress!, scalar: scalar, result: rPtr.baseAddress!, count: a.count)
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testDivScalarByArrayFloat() {
    let a: [Float] = [2, 4, 5, 8, 10]
    let scalar: Float = 100
    let expected = scalar / ContiguousArray<Float>(a)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.div(scalar: scalar, aPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  // MARK: - Float Reductions

  func testSumFloat() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let expected = ContiguousArray<Float>(a).sum

    let result = a.withUnsafeBufferPointer { aPtr in
      NumSwiftFlat.sum(aPtr.baseAddress!, count: a.count)
    }
    XCTAssertEqual(result, expected, accuracy: 1e-6)
  }

  func testMeanFloat() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let expected = ContiguousArray<Float>(a).mean

    let result = a.withUnsafeBufferPointer { aPtr in
      NumSwiftFlat.mean(aPtr.baseAddress!, count: a.count)
    }
    XCTAssertEqual(result, expected, accuracy: 1e-6)
  }

  func testSumOfSquaresFloat() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let expected = ContiguousArray<Float>(a).sumOfSquares

    let result = a.withUnsafeBufferPointer { aPtr in
      NumSwiftFlat.sumOfSquares(aPtr.baseAddress!, count: a.count)
    }
    XCTAssertEqual(result, expected, accuracy: 1e-6)
  }

  // MARK: - Float Unary Operations

  func testNegateFloat() {
    let a: [Float] = [1, -2, 3, -4, 5]
    let expected = NumSwiftFlat.negate(a)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.negate(aPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
      }
    }
    XCTAssertEqual(result, expected)
  }

  func testSqrtFloat() {
    let a: [Float] = [1, 4, 9, 16, 25]
    let expected = NumSwiftFlat.sqrt(a)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.sqrt(aPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
      }
    }
    XCTAssertEqual(result, expected)
  }

  func testClipFloat() {
    let a: [Float] = [-5, -1, 0, 1, 5]
    let expected = NumSwiftFlat.clip(a, to: 2)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.clip(aPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count, limit: 2)
      }
    }
    XCTAssertEqual(result, expected)
  }

  // MARK: - Float Matrix Operations

  func testTransposeFloat() {
    let a: [Float] = [1, 2, 3, 4, 5, 6]
    let expected = NumSwiftFlat.transpose(a, rows: 2, columns: 3)

    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.transpose(aPtr.baseAddress!, result: rPtr.baseAddress!, rows: 2, columns: 3)
      }
    }
    XCTAssertEqual(result, expected)
  }

  func testMatmulFloat() {
    let a: [Float] = [1, 2, 3, 4, 5, 6]
    let b: [Float] = [7, 8, 9, 10, 11, 12]
    let expected = NumSwiftFlat.matmul(a, b, aRows: 2, aCols: 3, bRows: 3, bCols: 2)

    var result = [Float](repeating: 0, count: 4)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.matmul(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!,
                              aRows: 2, aCols: 3, bRows: 3, bCols: 2)
        }
      }
    }
    XCTAssertEqual(result, expected)
  }

  // MARK: - Float16 Tests

  #if arch(arm64)
  func testAddFloat16() {
    let a: [Float16] = [1, 2, 3, 4, 5]
    let b: [Float16] = [10, 20, 30, 40, 50]
    let expected = ContiguousArray<Float16>(a) + ContiguousArray<Float16>(b)

    var result = [Float16](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.add(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
        }
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testSubFloat16() {
    let a: [Float16] = [10, 20, 30, 40, 50]
    let b: [Float16] = [1, 2, 3, 4, 5]
    let expected = ContiguousArray<Float16>(a) - ContiguousArray<Float16>(b)

    var result = [Float16](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.sub(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
        }
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testMulFloat16() {
    let a: [Float16] = [1, 2, 3, 4, 5]
    let b: [Float16] = [2, 3, 4, 5, 6]
    let expected = ContiguousArray<Float16>(a) * ContiguousArray<Float16>(b)

    var result = [Float16](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.mul(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
        }
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testDivFloat16() {
    let a: [Float16] = [10, 20, 30, 40, 50]
    let b: [Float16] = [2, 4, 5, 8, 10]
    let expected = ContiguousArray<Float16>(a) / ContiguousArray<Float16>(b)

    var result = [Float16](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.div(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!, count: a.count)
        }
      }
    }
    XCTAssertEqual(result, Array(expected))
  }

  func testSumFloat16() {
    let a: [Float16] = [1, 2, 3, 4, 5]
    let expected = ContiguousArray<Float16>(a).sum

    let result = a.withUnsafeBufferPointer { aPtr in
      NumSwiftFlat.sum(aPtr.baseAddress!, count: a.count)
    }
    XCTAssertEqual(result, expected)
  }

  func testMeanFloat16() {
    let a: [Float16] = [1, 2, 3, 4, 5]
    let expected = ContiguousArray<Float16>(a).mean

    let result = a.withUnsafeBufferPointer { aPtr in
      NumSwiftFlat.mean(aPtr.baseAddress!, count: a.count)
    }
    XCTAssertEqual(result, expected)
  }

  func testMatmulFloat16() {
    let a: [Float16] = [1, 2, 3, 4, 5, 6]
    let b: [Float16] = [7, 8, 9, 10, 11, 12]
    let expected = NumSwiftFlat.matmul(a, b, aRows: 2, aCols: 3, bRows: 3, bCols: 2)

    var result = [Float16](repeating: 0, count: 4)
    a.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.matmul(aPtr.baseAddress!, bPtr.baseAddress!, result: rPtr.baseAddress!,
                              aRows: 2, aCols: 3, bRows: 3, bCols: 2)
        }
      }
    }
    XCTAssertEqual(result, expected)
  }

  func testTransposeFloat16() {
    let a: [Float16] = [1, 2, 3, 4, 5, 6]
    let expected = NumSwiftFlat.transpose(a, rows: 2, columns: 3)

    var result = [Float16](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
      result.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.transpose(aPtr.baseAddress!, result: rPtr.baseAddress!, rows: 2, columns: 3)
      }
    }
    XCTAssertEqual(result, expected)
  }

  func testScalarArithmeticFloat16() {
    let a: [Float16] = [1, 2, 3, 4, 5]
    let scalar: Float16 = 10

    var addResult = [Float16](repeating: 0, count: a.count)
    var mulResult = [Float16](repeating: 0, count: a.count)
    var divResult = [Float16](repeating: 0, count: a.count)

    a.withUnsafeBufferPointer { aPtr in
      addResult.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.add(aPtr.baseAddress!, scalar: scalar, result: rPtr.baseAddress!, count: a.count)
      }
      mulResult.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.mul(aPtr.baseAddress!, scalar: scalar, result: rPtr.baseAddress!, count: a.count)
      }
      divResult.withUnsafeMutableBufferPointer { rPtr in
        NumSwiftFlat.div(aPtr.baseAddress!, scalar: scalar, result: rPtr.baseAddress!, count: a.count)
      }
    }

    XCTAssertEqual(addResult, Array(ContiguousArray<Float16>(a) + scalar))
    XCTAssertEqual(mulResult, Array(ContiguousArray<Float16>(a) * scalar))
    XCTAssertEqual(divResult, Array(ContiguousArray<Float16>(a) / scalar))
  }
  #endif
}
