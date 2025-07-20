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
  
  func test_speedConv2D_Optimized() {
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
  
  func test_speedConv2D() {
    guard isGithubCI == false else { return }
    
    let signalShape = (256,256)

    let filter: [[Float]] = [[0, 0, 1],
                             [0, 0, 1],
                             [0, 0, 1]]
    
    let signal: [[Float]] = NumSwift.onesLike(signalShape)
    
    measure {
      let _ = NumSwiftC.conv2d_ORIG(signal: signal,
                                  filter: filter,
                                  strides: (1,1),
                                  padding: .same,
                                  filterSize: (3,3),
                                  inputSize: signalShape)
    }
  }
}

extension NumSwiftC {
  public static func conv2d_ORIG(signal: [[Float]],
                            filter: [[Float]],
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> [[Float]] {
    
    let outputRows = ((inputSize.rows - filterSize.rows + (padding == .same ? filterSize.rows - 1 : 0)) / strides.0) + 1
    let outputCols = ((inputSize.columns - filterSize.columns + (padding == .same ? filterSize.columns - 1 : 0)) / strides.1) + 1
    
    let results: [[Float]] = NumSwift.zerosLike((rows: outputRows, columns: outputCols))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      signal.withUnsafeBufferPointer { sBuff in
        filter.withUnsafeBufferPointer { fBuff in
          let sPoint: [UnsafeMutablePointer<Float>?] = sBuff.map { UnsafeMutablePointer(mutating: $0) }
          let fPoint: [UnsafeMutablePointer<Float>?] = fBuff.map { UnsafeMutablePointer(mutating: $0) }
          
          let nscPadding: NSC_Padding = padding == .same ? same : valid
          
          nsc_conv2d(sPoint,
                               fPoint,
                               &rPoint,
                               NSC_Size(rows: Int32(strides.0), columns: Int32(strides.1)),
                               nscPadding,
                               NSC_Size(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns)),
                               NSC_Size(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns)))
        }
      }
    }
    
    return results
  }
}
