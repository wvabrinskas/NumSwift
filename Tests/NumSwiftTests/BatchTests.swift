import XCTest
@testable import NumSwift

final class BatchTests: XCTestCase {

  // MARK: - NumSwiftC conv1dBatch (flat Float)

  func test_conv1dBatch_matchesSingleCalls() {
    let inputSize = (rows: 5, columns: 5)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [0, 1, 0,
                           0, 1, 0,
                           0, 1, 0]

    let signal1: [Float] = [[Float]](repeating: [0,0,1,0,0], count: 5).flatMap { $0 }
    let signal2: [Float] = [[Float]](repeating: [1,0,0,0,1], count: 5).flatMap { $0 }

    let single1 = NumSwiftC.conv1d(signal: signal1, filter: filter, strides: (1,1),
                                    padding: .same, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.conv1d(signal: signal2, filter: filter, strides: (1,1),
                                    padding: .same, filterSize: filterSize, inputSize: inputSize)

    let batchedSignal = [signal1, signal2].flatMap { $0 }
    let batchResult = NumSwiftC.conv1dBatch(signal: batchedSignal, filter: filter,
                                             strides: (1,1), padding: .same,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: 2)

    let expectedCount = single1.count + single2.count
    XCTAssertEqual(batchResult.count, expectedCount)
    XCTAssertEqual(Array(batchResult[0..<single1.count]), single1)
    XCTAssertEqual(Array(batchResult[single1.count..<expectedCount]), single2)
  }

  func test_conv1dBatch_validPadding() {
    let inputSize = (rows: 5, columns: 5)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [1, 0, 0,
                           0, 1, 0,
                           0, 0, 1]

    let signal: [Float] = [[Float]](repeating: [1,0,0,0,1], count: 5).flatMap { $0 }
    let batchedSignal = [signal, signal, signal].flatMap { $0 }

    let single = NumSwiftC.conv1d(signal: signal, filter: filter, strides: (1,1),
                                   padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.conv1dBatch(signal: batchedSignal, filter: filter,
                                             strides: (1,1), padding: .valid,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: 3)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 3)
    for b in 0..<3 {
      XCTAssertEqual(Array(batchResult[b * outputSize..<(b + 1) * outputSize]), single,
                     "Batch item \(b) mismatch")
    }
  }

  // MARK: - NumSwiftC conv2dBatch (2D Float)

  func test_conv2dBatch_matchesSingleCalls() {
    let inputSize = (rows: 5, columns: 5)
    let filterSize = (rows: 3, columns: 3)

    let filter: [[Float]] = [[0, 1, 0],
                              [0, 1, 0],
                              [0, 1, 0]]

    let signal1: [[Float]] = [[Float]](repeating: [0,0,1,0,0], count: 5)
    let signal2: [[Float]] = [[Float]](repeating: [1,0,0,0,1], count: 5)

    let single1 = NumSwiftC.conv2d(signal: signal1, filter: filter, strides: (1,1),
                                    padding: .same, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.conv2d(signal: signal2, filter: filter, strides: (1,1),
                                    padding: .same, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.conv2dBatch(signals: [signal1, signal2], filter: filter,
                                             strides: (1,1), padding: .same,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: 2)

    XCTAssertEqual(batchResult.count, 2)
    XCTAssertEqual(batchResult[0], single1)
    XCTAssertEqual(batchResult[1], single2)
  }

  // MARK: - NumSwiftC transConv1dBatch (flat Float)

  func test_transConv1dBatch_matchesSingleCalls() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [1, 0, 0,
                           0, 1, 0,
                           0, 0, 1]

    let signal1: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    let signal2: [Float] = [0, 1, 0, 1, 0, 1, 0, 1, 0]

    let single1 = NumSwiftC.transConv1d(signal: signal1, filter: filter, strides: (1,1),
                                         padding: .valid, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.transConv1d(signal: signal2, filter: filter, strides: (1,1),
                                         padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchedSignal = [signal1, signal2].flatMap { $0 }
    let batchResult = NumSwiftC.transConv1dBatch(signal: batchedSignal, filter: filter,
                                                  strides: (1,1), padding: .valid,
                                                  filterSize: filterSize, inputSize: inputSize,
                                                  batchCount: 2)

    let outputSize = single1.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single1)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single2)
  }

  func test_transConv1dBatch_samePadding() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [1, 1, 1,
                           1, 1, 1,
                           1, 1, 1]

    let signal: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    let single = NumSwiftC.transConv1d(signal: signal, filter: filter, strides: (1,1),
                                        padding: .same, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.transConv1dBatch(signal: [signal, signal].flatMap { $0 }, filter: filter,
                                                  strides: (1,1), padding: .same,
                                                  filterSize: filterSize, inputSize: inputSize,
                                                  batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single)
  }

  // MARK: - NumSwiftC transConv2dBatch (2D Float)

  func test_transConv2dBatch_matchesSingleCalls() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 3, columns: 3)

    let filter: [[Float]] = [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]

    let signal1: [[Float]] = [[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]
    let signal2: [[Float]] = [[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]]

    let single1 = NumSwiftC.transConv2d(signal: signal1, filter: filter, strides: (1,1),
                                         padding: .valid, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.transConv2d(signal: signal2, filter: filter, strides: (1,1),
                                         padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.transConv2dBatch(signals: [signal1, signal2], filter: filter,
                                                  strides: (1,1), padding: .valid,
                                                  filterSize: filterSize, inputSize: inputSize,
                                                  batchCount: 2)

    XCTAssertEqual(batchResult.count, 2)
    XCTAssertEqual(batchResult[0], single1)
    XCTAssertEqual(batchResult[1], single2)
  }

  // MARK: - NumSwiftC matmul1dBatch (flat Float)

  func test_matmul1dBatch_matchesSingleCalls() {
    let aSize = (rows: 3, columns: 3)
    let bSize = (rows: 3, columns: 2)

    let b: [Float] = [2, 2,
                      2, 2,
                      2, 2]

    let a1: [Float] = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    let a2: [Float] = [4, 4, 4, 5, 5, 5, 6, 6, 6]

    let single1 = NumSwiftC.matmul1d(a1, b: b, aSize: aSize, bSize: bSize)
    let single2 = NumSwiftC.matmul1d(a2, b: b, aSize: aSize, bSize: bSize)

    let batchedA = [a1, a2].flatMap { $0 }
    let batchResult = NumSwiftC.matmul1dBatch(batchedA, b: b, aSize: aSize, bSize: bSize, batchCount: 2)

    let outputSize = single1.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single1)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single2)
  }

  func test_matmul1dBatch_singleBatch() {
    let aSize = (rows: 2, columns: 3)
    let bSize = (rows: 3, columns: 2)

    let a: [Float] = [1, 2, 3, 4, 5, 6]
    let b: [Float] = [7, 8, 9, 10, 11, 12]

    let single = NumSwiftC.matmul1d(a, b: b, aSize: aSize, bSize: bSize)
    let batchResult = NumSwiftC.matmul1dBatch(a, b: b, aSize: aSize, bSize: bSize, batchCount: 1)

    XCTAssertEqual(batchResult, single)
  }

  // MARK: - NumSwiftC matmulBatch (2D Float)

  func test_matmulBatch_matchesSingleCalls() {
    let aSize = (rows: 3, columns: 3)
    let bSize = (rows: 3, columns: 2)

    let b: [[Float]] = [[2, 2],
                         [2, 2],
                         [2, 2]]

    let a1: [[Float]] = [[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3]]
    let a2: [[Float]] = [[4, 4, 4],
                          [5, 5, 5],
                          [6, 6, 6]]

    let single1 = NumSwiftC.matmul(a1, b: b, aSize: aSize, bSize: bSize)
    let single2 = NumSwiftC.matmul(a2, b: b, aSize: aSize, bSize: bSize)

    let batchResult = NumSwiftC.matmulBatch([a1, a2], b: b, aSize: aSize, bSize: bSize, batchCount: 2)

    XCTAssertEqual(batchResult.count, 2)
    XCTAssertEqual(batchResult[0], single1)
    XCTAssertEqual(batchResult[1], single2)
  }

  // MARK: - NumSwiftFlat batch (Array<Float>)

  func test_flat_conv2dBatch_matchesSingleCalls() {
    let inputSize = (rows: 5, columns: 5)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [0, 1, 0,
                           0, 1, 0,
                           0, 1, 0]

    let signal1: [Float] = [[Float]](repeating: [0,0,1,0,0], count: 5).flatMap { $0 }
    let signal2: [Float] = [[Float]](repeating: [1,0,0,0,1], count: 5).flatMap { $0 }

    let single1 = NumSwiftFlat.conv2d(signal: signal1, filter: filter, strides: (1,1),
                                       padding: .same, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftFlat.conv2d(signal: signal2, filter: filter, strides: (1,1),
                                       padding: .same, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftFlat.conv2dBatch(signal: [signal1, signal2].flatMap { $0 }, filter: filter,
                                                strides: (1,1), padding: .same,
                                                filterSize: filterSize, inputSize: inputSize,
                                                batchCount: 2)

    let outputSize = single1.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single1)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single2)
  }

  func test_flat_transConv2dBatch_matchesSingleCalls() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [1, 0, 0,
                           0, 1, 0,
                           0, 0, 1]

    let signal: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    let single = NumSwiftFlat.transConv2d(signal: signal, filter: filter, strides: (1,1),
                                           padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftFlat.transConv2dBatch(signal: [signal, signal].flatMap { $0 }, filter: filter,
                                                     strides: (1,1), padding: .valid,
                                                     filterSize: filterSize, inputSize: inputSize,
                                                     batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single)
  }

  func test_flat_matmulBatch_matchesSingleCalls() {
    let a1: [Float] = [1, 2, 3, 4, 5, 6]
    let a2: [Float] = [7, 8, 9, 10, 11, 12]
    let b: [Float] = [1, 0, 0, 1, 1, 1]

    let single1 = NumSwiftFlat.matmul(a1, b, aRows: 2, aCols: 3, bRows: 3, bCols: 2)
    let single2 = NumSwiftFlat.matmul(a2, b, aRows: 2, aCols: 3, bRows: 3, bCols: 2)

    let batchResult = NumSwiftFlat.matmulBatch([a1, a2].flatMap { $0 }, b, aRows: 2, aCols: 3,
                                                bRows: 3, bCols: 2, batchCount: 2)

    let outputSize = single1.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single1)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single2)
  }

  // MARK: - NumSwiftFlat batch (ContiguousArray<Float>)

  func test_contiguous_conv2dBatch() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 2, columns: 2)

    let filter: ContiguousArray<Float> = [1, 0, 0, 1]
    let signal: ContiguousArray<Float> = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    let single = NumSwiftFlat.conv2d(signal: signal, filter: filter, strides: (1,1),
                                      padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchSignal = ContiguousArray([signal, signal].flatMap { $0 })
    let batchResult = NumSwiftFlat.conv2dBatch(signal: batchSignal, filter: filter,
                                                strides: (1,1), padding: .valid,
                                                filterSize: filterSize, inputSize: inputSize,
                                                batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(ContiguousArray(batchResult[0..<outputSize]), single)
    XCTAssertEqual(ContiguousArray(batchResult[outputSize..<outputSize * 2]), single)
  }

  func test_contiguous_transConv2dBatch() {
    let inputSize = (rows: 2, columns: 2)
    let filterSize = (rows: 2, columns: 2)

    let filter: ContiguousArray<Float> = [1, 1, 1, 1]
    let signal: ContiguousArray<Float> = [1, 2, 3, 4]

    let single = NumSwiftFlat.transConv2d(signal: signal, filter: filter, strides: (1,1),
                                           padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftFlat.transConv2dBatch(signal: ContiguousArray([signal, signal].flatMap { $0 }), filter: filter,
                                                     strides: (1,1), padding: .valid,
                                                     filterSize: filterSize, inputSize: inputSize,
                                                     batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(ContiguousArray(batchResult[0..<outputSize]), single)
    XCTAssertEqual(ContiguousArray(batchResult[outputSize..<outputSize * 2]), single)
  }

  func test_contiguous_matmulBatch() {
    let a: ContiguousArray<Float> = [1, 2, 3, 4, 5, 6]
    let b: ContiguousArray<Float> = [7, 8, 9, 10, 11, 12]

    let single = NumSwiftFlat.matmul(a, b, aRows: 2, aCols: 3, bRows: 3, bCols: 2)

    let batchResult = NumSwiftFlat.matmulBatch(ContiguousArray([a, a].flatMap { $0 }), b, aRows: 2, aCols: 3,
                                                bRows: 3, bCols: 2, batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(ContiguousArray(batchResult[0..<outputSize]), single)
    XCTAssertEqual(ContiguousArray(batchResult[outputSize..<outputSize * 2]), single)
  }

  // MARK: - NumSwiftFlatPointer batch (Float pointers)

  func test_pointer_conv2dBatch() {
    let inputSize = (rows: 5, columns: 5)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [0, 1, 0,
                           0, 1, 0,
                           0, 1, 0]

    let signal1: [Float] = [[Float]](repeating: [0,0,1,0,0], count: 5).flatMap { $0 }
    let signal2: [Float] = [[Float]](repeating: [1,0,0,0,1], count: 5).flatMap { $0 }

    let single1 = NumSwiftC.conv1d(signal: signal1, filter: filter, strides: (1,1),
                                    padding: .same, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.conv1d(signal: signal2, filter: filter, strides: (1,1),
                                    padding: .same, filterSize: filterSize, inputSize: inputSize)

    let batchedSignal = [signal1, signal2].flatMap { $0 }
    let outputSize = single1.count
    var result = [Float](repeating: 0, count: outputSize * 2)

    batchedSignal.withUnsafeBufferPointer { sPtr in
      filter.withUnsafeBufferPointer { fPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.conv2dBatch(signal: sPtr.baseAddress!, filter: fPtr.baseAddress!,
                                   result: rPtr.baseAddress!, strides: (1,1), padding: .same,
                                   filterSize: filterSize, inputSize: inputSize, batchCount: 2)
        }
      }
    }

    XCTAssertEqual(Array(result[0..<outputSize]), single1)
    XCTAssertEqual(Array(result[outputSize..<outputSize * 2]), single2)
  }

  func test_pointer_transConv2dBatch() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [1, 0, 0,
                           0, 1, 0,
                           0, 0, 1]

    let signal: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    let single = NumSwiftC.transConv1d(signal: signal, filter: filter, strides: (1,1),
                                        padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchedSignal = [signal, signal].flatMap { $0 }
    let outputSize = single.count
    var result = [Float](repeating: 0, count: outputSize * 2)

    batchedSignal.withUnsafeBufferPointer { sPtr in
      filter.withUnsafeBufferPointer { fPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.transConv2dBatch(signal: sPtr.baseAddress!, filter: fPtr.baseAddress!,
                                        result: rPtr.baseAddress!, strides: (1,1), padding: .valid,
                                        filterSize: filterSize, inputSize: inputSize, batchCount: 2)
        }
      }
    }

    XCTAssertEqual(Array(result[0..<outputSize]), single)
    XCTAssertEqual(Array(result[outputSize..<outputSize * 2]), single)
  }

  func test_pointer_matmulBatch() {
    let a1: [Float] = [1, 2, 3, 4, 5, 6]
    let a2: [Float] = [7, 8, 9, 10, 11, 12]
    let b: [Float] = [1, 0, 0, 1, 1, 1]

    let single1 = NumSwiftC.matmul1d(a1, b: b, aSize: (2, 3), bSize: (3, 2))
    let single2 = NumSwiftC.matmul1d(a2, b: b, aSize: (2, 3), bSize: (3, 2))

    let batchedA = [a1, a2].flatMap { $0 }
    let outputSize = single1.count
    var result = [Float](repeating: 0, count: outputSize * 2)

    batchedA.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.matmulBatch(aPtr.baseAddress!, bPtr.baseAddress!,
                                   result: rPtr.baseAddress!,
                                   aRows: 2, aCols: 3, bRows: 3, bCols: 2, batchCount: 2)
        }
      }
    }

    XCTAssertEqual(Array(result[0..<outputSize]), single1)
    XCTAssertEqual(Array(result[outputSize..<outputSize * 2]), single2)
  }

  // MARK: - Edge cases

  func test_conv1dBatch_singleBatch() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 2, columns: 2)

    let filter: [Float] = [1, 1, 1, 1]
    let signal: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    let single = NumSwiftC.conv1d(signal: signal, filter: filter, strides: (1,1),
                                   padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.conv1dBatch(signal: signal, filter: filter,
                                             strides: (1,1), padding: .valid,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: 1)

    XCTAssertEqual(batchResult, single)
  }

  func test_conv2dBatch_singleBatch() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 2, columns: 2)

    let filter: [[Float]] = [[1, 1],
                              [1, 1]]
    let signal: [[Float]] = [[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]]

    let single = NumSwiftC.conv2d(signal: signal, filter: filter, strides: (1,1),
                                   padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.conv2dBatch(signals: [signal], filter: filter,
                                             strides: (1,1), padding: .valid,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: 1)

    XCTAssertEqual(batchResult.count, 1)
    XCTAssertEqual(batchResult[0], single)
  }

  func test_batch_withStrides() {
    let inputSize = (rows: 4, columns: 4)
    let filterSize = (rows: 2, columns: 2)

    let filter: [Float] = [1, 0, 0, 1]
    let signal: [Float] = [1, 2, 3, 4,
                           5, 6, 7, 8,
                           9, 10, 11, 12,
                           13, 14, 15, 16]

    let single = NumSwiftC.conv1d(signal: signal, filter: filter, strides: (2,2),
                                   padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.conv1dBatch(signal: [signal, signal].flatMap { $0 }, filter: filter,
                                             strides: (2,2), padding: .valid,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single)
  }

  func test_matmulBatch_differentInputs() {
    let aSize = (rows: 2, columns: 2)
    let bSize = (rows: 2, columns: 2)

    let b: [[Float]] = [[1, 0],
                         [0, 1]]

    let a1: [[Float]] = [[1, 2],
                          [3, 4]]
    let a2: [[Float]] = [[5, 6],
                          [7, 8]]

    let batchResult = NumSwiftC.matmulBatch([a1, a2], b: b, aSize: aSize, bSize: bSize, batchCount: 2)

    XCTAssertEqual(batchResult.count, 2)
    XCTAssertEqual(batchResult[0], a1)
    XCTAssertEqual(batchResult[1], a2)
  }

  func test_transConv2dBatch_withStrides() {
    let inputSize = (rows: 2, columns: 2)
    let filterSize = (rows: 3, columns: 3)

    let filter: [[Float]] = [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]

    let signal1: [[Float]] = [[1, 2],
                               [3, 4]]
    let signal2: [[Float]] = [[5, 6],
                               [7, 8]]

    let single1 = NumSwiftC.transConv2d(signal: signal1, filter: filter, strides: (2,2),
                                         padding: .valid, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.transConv2d(signal: signal2, filter: filter, strides: (2,2),
                                         padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.transConv2dBatch(signals: [signal1, signal2], filter: filter,
                                                  strides: (2,2), padding: .valid,
                                                  filterSize: filterSize, inputSize: inputSize,
                                                  batchCount: 2)

    XCTAssertEqual(batchResult.count, 2)
    XCTAssertEqual(batchResult[0], single1)
    XCTAssertEqual(batchResult[1], single2)
  }

  func test_largeBatch_conv1d() {
    let inputSize = (rows: 4, columns: 4)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float] = [Float](repeating: 1.0, count: 9)
    let signal: [Float] = [Float](repeating: 1.0, count: 16)

    let single = NumSwiftC.conv1d(signal: signal, filter: filter, strides: (1,1),
                                   padding: .same, filterSize: filterSize, inputSize: inputSize)

    var batchedSignal: [Float] = []
    let batchCount = 8
    for _ in 0..<batchCount {
      batchedSignal.append(contentsOf: signal)
    }

    let batchResult = NumSwiftC.conv1dBatch(signal: batchedSignal, filter: filter,
                                             strides: (1,1), padding: .same,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: batchCount)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * batchCount)
    for b in 0..<batchCount {
      XCTAssertEqual(Array(batchResult[b * outputSize..<(b + 1) * outputSize]), single,
                     "Batch item \(b) mismatch")
    }
  }

  // MARK: - Float16 Batch Tests

  #if arch(arm64)

  func test_conv1dBatch_float16_matchesSingleCalls() {
    let inputSize = (rows: 5, columns: 5)
    let filterSize = (rows: 3, columns: 3)

    let filter: [Float16] = [0, 1, 0,
                              0, 1, 0,
                              0, 1, 0]

    let signal1: [Float16] = [[Float16]](repeating: [0,0,1,0,0], count: 5).flatMap { $0 }
    let signal2: [Float16] = [[Float16]](repeating: [1,0,0,0,1], count: 5).flatMap { $0 }

    let single1 = NumSwiftC.conv1d(signal: signal1, filter: filter, strides: (1,1),
                                    padding: .same, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.conv1d(signal: signal2, filter: filter, strides: (1,1),
                                    padding: .same, filterSize: filterSize, inputSize: inputSize)

    let batchedSignal: [Float16] = [signal1, signal2].flatMap { $0 }
    let batchResult = NumSwiftC.conv1dBatch(signal: batchedSignal, filter: filter,
                                             strides: (1,1), padding: .same,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: 2)

    let outputSize = single1.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single1)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single2)
  }

  func test_conv2dBatch_float16_matchesSingleCalls() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 2, columns: 2)

    let filter: [[Float16]] = [[1, 0],
                                [0, 1]]
    let signal1: [[Float16]] = [[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]
    let signal2: [[Float16]] = [[9, 8, 7],
                                 [6, 5, 4],
                                 [3, 2, 1]]

    let single1 = NumSwiftC.conv2d(signal: signal1, filter: filter, strides: (1,1),
                                    padding: .valid, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.conv2d(signal: signal2, filter: filter, strides: (1,1),
                                    padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.conv2dBatch(signals: [signal1, signal2], filter: filter,
                                             strides: (1,1), padding: .valid,
                                             filterSize: filterSize, inputSize: inputSize,
                                             batchCount: 2)

    XCTAssertEqual(batchResult.count, 2)
    XCTAssertEqual(batchResult[0], single1)
    XCTAssertEqual(batchResult[1], single2)
  }

  func test_transConv1dBatch_float16() {
    let inputSize = (rows: 2, columns: 2)
    let filterSize = (rows: 2, columns: 2)

    let filter: [Float16] = [1, 1, 1, 1]
    let signal: [Float16] = [1, 2, 3, 4]

    let single = NumSwiftC.transConv1d(signal: signal, filter: filter, strides: (1,1),
                                        padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.transConv1dBatch(signal: [signal, signal].flatMap { $0 }, filter: filter,
                                                  strides: (1,1), padding: .valid,
                                                  filterSize: filterSize, inputSize: inputSize,
                                                  batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single)
  }

  func test_transConv2dBatch_float16() {
    let inputSize = (rows: 2, columns: 2)
    let filterSize = (rows: 2, columns: 2)

    let filter: [[Float16]] = [[1, 1],
                                [1, 1]]

    let signal1: [[Float16]] = [[1, 2],
                                 [3, 4]]
    let signal2: [[Float16]] = [[5, 6],
                                 [7, 8]]

    let single1 = NumSwiftC.transConv2d(signal: signal1, filter: filter, strides: (1,1),
                                         padding: .valid, filterSize: filterSize, inputSize: inputSize)
    let single2 = NumSwiftC.transConv2d(signal: signal2, filter: filter, strides: (1,1),
                                         padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftC.transConv2dBatch(signals: [signal1, signal2], filter: filter,
                                                  strides: (1,1), padding: .valid,
                                                  filterSize: filterSize, inputSize: inputSize,
                                                  batchCount: 2)

    XCTAssertEqual(batchResult.count, 2)
    XCTAssertEqual(batchResult[0], single1)
    XCTAssertEqual(batchResult[1], single2)
  }

  func test_matmul1dBatch_float16() {
    let aSize = (rows: 2, columns: 3)
    let bSize = (rows: 3, columns: 2)

    let b: [Float16] = [1, 0, 0, 1, 1, 1]
    let a1: [Float16] = [1, 2, 3, 4, 5, 6]
    let a2: [Float16] = [7, 8, 9, 10, 11, 12]

    let single1 = NumSwiftC.matmul1d(a1, b: b, aSize: aSize, bSize: bSize)
    let single2 = NumSwiftC.matmul1d(a2, b: b, aSize: aSize, bSize: bSize)

    let batchResult = NumSwiftC.matmul1dBatch([a1, a2].flatMap { $0 }, b: b, aSize: aSize, bSize: bSize, batchCount: 2)

    let outputSize = single1.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single1)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single2)
  }

  func test_matmulBatch_float16() {
    let aSize = (rows: 2, columns: 2)
    let bSize = (rows: 2, columns: 2)

    let b: [[Float16]] = [[1, 0],
                           [0, 1]]
    let a1: [[Float16]] = [[1, 2],
                            [3, 4]]
    let a2: [[Float16]] = [[5, 6],
                            [7, 8]]

    let batchResult = NumSwiftC.matmulBatch([a1, a2], b: b, aSize: aSize, bSize: bSize, batchCount: 2)

    XCTAssertEqual(batchResult.count, 2)
    XCTAssertEqual(batchResult[0], a1)
    XCTAssertEqual(batchResult[1], a2)
  }

  func test_flat_conv2dBatch_float16() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 2, columns: 2)

    let filter: [Float16] = [1, 1, 1, 1]
    let signal: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    let single = NumSwiftFlat.conv2d(signal: signal, filter: filter, strides: (1,1),
                                      padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchResult = NumSwiftFlat.conv2dBatch(signal: [signal, signal].flatMap { $0 }, filter: filter,
                                                strides: (1,1), padding: .valid,
                                                filterSize: filterSize, inputSize: inputSize,
                                                batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single)
  }

  func test_flat_matmulBatch_float16() {
    let a: [Float16] = [1, 2, 3, 4, 5, 6]
    let b: [Float16] = [7, 8, 9, 10, 11, 12]

    let single = NumSwiftFlat.matmul(a, b, aRows: 2, aCols: 3, bRows: 3, bCols: 2)

    let batchResult = NumSwiftFlat.matmulBatch([a, a].flatMap { $0 }, b, aRows: 2, aCols: 3,
                                                bRows: 3, bCols: 2, batchCount: 2)

    let outputSize = single.count
    XCTAssertEqual(batchResult.count, outputSize * 2)
    XCTAssertEqual(Array(batchResult[0..<outputSize]), single)
    XCTAssertEqual(Array(batchResult[outputSize..<outputSize * 2]), single)
  }

  func test_pointer_conv2dBatch_float16() {
    let inputSize = (rows: 3, columns: 3)
    let filterSize = (rows: 2, columns: 2)

    let filter: [Float16] = [1, 0, 0, 1]
    let signal: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    let single = NumSwiftC.conv1d(signal: signal, filter: filter, strides: (1,1),
                                   padding: .valid, filterSize: filterSize, inputSize: inputSize)

    let batchedSignal: [Float16] = [signal, signal].flatMap { $0 }
    let outputSize = single.count
    var result = [Float16](repeating: 0, count: outputSize * 2)

    batchedSignal.withUnsafeBufferPointer { sPtr in
      filter.withUnsafeBufferPointer { fPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.conv2dBatch(signal: sPtr.baseAddress!, filter: fPtr.baseAddress!,
                                   result: rPtr.baseAddress!, strides: (1,1), padding: .valid,
                                   filterSize: filterSize, inputSize: inputSize, batchCount: 2)
        }
      }
    }

    XCTAssertEqual(Array(result[0..<outputSize]), single)
    XCTAssertEqual(Array(result[outputSize..<outputSize * 2]), single)
  }

  func test_pointer_matmulBatch_float16() {
    let a: [Float16] = [1, 2, 3, 4, 5, 6]
    let b: [Float16] = [7, 8, 9, 10, 11, 12]

    let single = NumSwiftC.matmul1d(a, b: b, aSize: (2, 3), bSize: (3, 2))

    let batchedA: [Float16] = [a, a].flatMap { $0 }
    let outputSize = single.count
    var result = [Float16](repeating: 0, count: outputSize * 2)

    batchedA.withUnsafeBufferPointer { aPtr in
      b.withUnsafeBufferPointer { bPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
          NumSwiftFlat.matmulBatch(aPtr.baseAddress!, bPtr.baseAddress!,
                                   result: rPtr.baseAddress!,
                                   aRows: 2, aCols: 3, bRows: 3, bCols: 2, batchCount: 2)
        }
      }
    }

    XCTAssertEqual(Array(result[0..<outputSize]), single)
    XCTAssertEqual(Array(result[outputSize..<outputSize * 2]), single)
  }

  #endif
}
