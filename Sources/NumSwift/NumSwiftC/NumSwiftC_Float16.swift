//
//  File.swift
//
//
//  Created by William Vabrinskas on 3/28/22.
//

import Foundation
import NumSwiftC

#if arch(arm64)
public extension NumSwiftC {
  
  public static func add(_ a: [[Float16]], _ b: [[Float16]]) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        b.withUnsafeBufferPointer { bBuff in
          let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
          let bPoint: [UnsafeMutablePointer<Float16>?] = bBuff.map { UnsafeMutablePointer(mutating: $0) }
          nsc_add2d_f16(NSC_Size(rows: Int32(rows),
                                 columns: Int32(columns)),
                        aPoint,
                        bPoint,
                        &rPoint)
        }
      }
    }
    
    return results
  }
  
  public static func sub(_ a: [[Float16]], _ b: [[Float16]]) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        b.withUnsafeBufferPointer { bBuff in
          let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
          let bPoint: [UnsafeMutablePointer<Float16>?] = bBuff.map { UnsafeMutablePointer(mutating: $0) }
          nsc_sub2d_f16(NSC_Size(rows: Int32(rows),
                                 columns: Int32(columns)),
                        aPoint,
                        bPoint,
                        &rPoint)
        }
      }
    }
    
    return results
  }
  
  public static func divide(_ a: [[Float16]], _ b: [[Float16]]) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        b.withUnsafeBufferPointer { bBuff in
          let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
          let bPoint: [UnsafeMutablePointer<Float16>?] = bBuff.map { UnsafeMutablePointer(mutating: $0) }
          nsc_divide2d_f16(NSC_Size(rows: Int32(rows),
                                    columns: Int32(columns)),
                           aPoint,
                           bPoint,
                           &rPoint)
        }
      }
    }
    
    return results
  }
  
  public static func mult(_ a: [[Float16]], _ b: [[Float16]]) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        b.withUnsafeBufferPointer { bBuff in
          let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
          let bPoint: [UnsafeMutablePointer<Float16>?] = bBuff.map { UnsafeMutablePointer(mutating: $0) }
          nsc_mult2d_f16(NSC_Size(rows: Int32(rows),
                                  columns: Int32(columns)),
                         aPoint,
                         bPoint,
                         &rPoint)
        }
      }
    }
    
    return results
  }
  
  // MARK: - 2D Arithmetic with Scalar Operations
  
  public static func add(_ a: [[Float16]], scalar: Float16) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_add2d_scalar_f16(NSC_Size(rows: Int32(rows),
                                      columns: Int32(columns)),
                             aPoint,
                             scalar,
                             &rPoint)
      }
    }
    
    return results
  }
  
  public static func add(_ a: [[Float16]], array: [Float16]) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_add2d_array_scalar_f16(NSC_Size(rows: Int32(rows),
                                            columns: Int32(columns)),
                                   aPoint,
                                   array,
                                   &rPoint)
      }
    }
    
    return results
  }
  
  public static func sub(_ a: [[Float16]], scalar: Float16) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_sub2d_scalar_f16(NSC_Size(rows: Int32(rows),
                                      columns: Int32(columns)),
                             aPoint,
                             scalar,
                             &rPoint)
      }
    }
    
    return results
  }
  
  public static func sub(_ a: [[Float16]], array: [Float16]) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_sub2d_array_scalar_f16(NSC_Size(rows: Int32(rows),
                                            columns: Int32(columns)),
                                   aPoint,
                                   array,
                                   &rPoint)
      }
    }
    
    return results
  }
  
  public static func mult(_ a: [[Float16]], scalar: Float16) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_mult2d_scalar_f16(NSC_Size(rows: Int32(rows),
                                       columns: Int32(columns)),
                              aPoint,
                              scalar,
                              &rPoint)
      }
    }
    
    return results
  }
  
  public static func mult(_ a: [[Float16]], array: [Float16]) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_mult2d_array_scalar_f16(NSC_Size(rows: Int32(rows),
                                             columns: Int32(columns)),
                                    aPoint,
                                    array,
                                    &rPoint)
      }
    }
    
    return results
  }
  
  public static func divide(_ a: [[Float16]], scalar: Float16) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_divide2d_scalar_f16(NSC_Size(rows: Int32(rows),
                                         columns: Int32(columns)),
                                aPoint,
                                scalar,
                                &rPoint)
      }
    }
    
    return results
  }
  
  public static func divide(_ a: [[Float16]], array: [Float16]) -> [[Float16]] {
    let shape = a.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: rows,
                                                   columns: columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_divide2d_array_scalar_f16(NSC_Size(rows: Int32(rows),
                                               columns: Int32(columns)),
                                      aPoint,
                                      array,
                                      &rPoint)
      }
    }
    
    return results
  }
  
  public static func tranpose(_ a: [[Float16]], size: (rows: Int, columns: Int)) -> [[Float16]] {
    let result: [[Float16]] = NumSwift.zerosLike((rows: size.columns, columns: size.rows))
    
    result.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_transpose_2d_16(aPoint,
                            &rPoint,
                            .init(rows: Int32(size.rows),
                                  columns: Int32(size.columns)))
      }
      
    }
    
    return result
  }
  
  public static func matmul(_ a: [[Float16]],
                            b: [[Float16]],
                            aSize: (rows: Int, columns: Int),
                            bSize: (rows: Int, columns: Int)) -> [[Float16]] {
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: aSize.rows,
                                                   columns: bSize.columns))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      a.withUnsafeBufferPointer { aBuff in
        b.withUnsafeBufferPointer { bBuff in
          let aPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
          let bPoint: [UnsafeMutablePointer<Float16>?] = bBuff.map { UnsafeMutablePointer(mutating: $0) }
          nsc_matmul_16(NSC_Size(rows: Int32(aSize.rows), columns: Int32(aSize.columns)),
                        NSC_Size(rows: Int32(bSize.rows), columns: Int32(bSize.columns)),
                        aPoint,
                        bPoint,
                        &rPoint)
        }
      }
    }
    
    return results
  }
  
  public static func flatten(_ input: [[Float16]], inputSize: (rows: Int, columns: Int)? = nil) -> [Float16] {
    
    let shape = input.shape
    var rows = shape[safe: 1, 0]
    var columns = shape[safe: 0, 0]
    
    if let inputSize = inputSize {
      rows = inputSize.rows
      columns = inputSize.columns
    }
    
    var results: [Float16] = [Float16](repeating: 0, count: rows * columns)
    
    input.withUnsafeBufferPointer { (inputsBuffer) in
      let inPuts: [UnsafeMutablePointer<Float16>?] = inputsBuffer.map { UnsafeMutablePointer(mutating: $0) }
      
      nsc_flatten2d_16(NSC_Size(rows: Int32(rows), columns: Int32(columns)), inPuts, &results)
    }
    
    return results
  }
  
  public static func stridePad(signal: [[Float16]],
                               strides: (rows: Int, columns: Int)) -> [[Float16]] {
    
    guard strides.rows - 1 > 0 || strides.columns - 1 > 0 else {
      return signal
    }
    
    let shape = signal.shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    
    let newRows = rows + ((strides.rows - 1) * (rows - 1))
    let newColumns = columns + ((strides.columns - 1) * (columns - 1))
    
    var results: [[Float16]] = NumSwift.zerosLike((rows: newRows, columns: newColumns))
    
    results.withUnsafeBufferPointer { rBuff in
      signal.withUnsafeBufferPointer { sBuff in
        var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
        let sPoint: [UnsafeMutablePointer<Float16>?] = sBuff.map { UnsafeMutablePointer(mutating: $0) }
        nsc_stride_pad_2D_f16(sPoint,
                              &rPoint,
                              NSC_Size(rows: Int32(rows),
                                       columns: Int32(columns)),
                              NSC_Size(rows: Int32(strides.rows),
                                       columns: Int32(strides.columns)))
      }
    }
    
    
    return results
  }
  
  public static func stridePad1D(signal: [Float16],
                                 strides: (rows: Int, columns: Int)) -> [Float16] {
    
    guard strides.rows - 1 > 0 || strides.columns - 1 > 0 else {
      return signal
    }
    
    let shape = signal.shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    
    let newRows = rows + ((strides.rows - 1) * (rows - 1))
    let newColumns = columns + ((strides.columns - 1) * (columns - 1))
    
    var results: [Float16] = [Float16](repeating: 0, count: newRows * newColumns)
    
    let flatSignal: [Float16] = signal
    
    nsc_stride_pad_f16(flatSignal,
                       &results, NSC_Size(rows: Int32(rows),
                                          columns: Int32(columns)),
                       NSC_Size(rows: Int32(strides.rows),
                                columns: Int32(strides.columns)))
    
    return results
  }
  
  public static func zeroPad(signal: [[Float16]],
                             padding: NumSwiftPadding) -> [[Float16]] {
    
    guard padding.right > 0 || padding.left > 0 || padding.top > 0 || padding.bottom > 0 else {
      return signal
    }
    
    let shape = signal.shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    
    let expectedRows = rows + padding.top + padding.bottom
    let expectedColumns = columns + padding.left + padding.right
    
    let results: [[Float16]] = NumSwift.zerosLike((expectedRows, expectedColumns))
    
    results.withUnsafeBufferPointer { rBuff in
      signal.withUnsafeBufferPointer { sBuff in
        var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
        let sPoint: [UnsafeMutablePointer<Float16>?] = sBuff.map { UnsafeMutablePointer(mutating: $0) }
        
        nsc_specific_zero_pad_2d_f16(sPoint,
                                     &rPoint,
                                     NSC_Size(rows: Int32(rows),
                                              columns: Int32(columns)),
                                     Int32(padding.top),
                                     Int32(padding.bottom),
                                     Int32(padding.left),
                                     Int32(padding.right))
      }
    }
    
    return results
  }
  
  public static func zeroPad(signal: [[Float16]],
                             filterSize: (rows: Int, columns: Int),
                             inputSize: (rows: Int, columns: Int),
                             stride: (Int, Int) = (1,1)) -> [[Float16]] {
    
    let padding = NumSwiftC.paddingCalculation(strides: stride,
                                               padding: .same,
                                               filterSize: filterSize,
                                               inputSize: inputSize)
    
    guard padding.right > 0 || padding.left > 0 || padding.top > 0 || padding.bottom > 0 else {
      return signal
    }
    
    let shape = signal.shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    
    let expectedRows = rows + padding.top + padding.bottom
    let expectedColumns = columns + padding.left + padding.right
    
    let results: [[Float16]] = NumSwift.zerosLike((expectedRows, expectedColumns))
    
    results.withUnsafeBufferPointer { rBuff in
      signal.withUnsafeBufferPointer { sBuff in
        var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
        let sPoint: [UnsafeMutablePointer<Float16>?] = sBuff.map { UnsafeMutablePointer(mutating: $0) }
        
        nsc_specific_zero_pad_2d_f16(sPoint,
                                     &rPoint,
                                     NSC_Size(rows: Int32(rows),
                                              columns: Int32(columns)),
                                     Int32(padding.top),
                                     Int32(padding.bottom),
                                     Int32(padding.left),
                                     Int32(padding.right))
      }
    }
    
    return results
  }
  
  public static func conv2d(signal: [[Float16]],
                            filter: [[Float16]],
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> [[Float16]] {
    
    let paddingResult = padding.extra(inputSize: inputSize, filterSize: filterSize, stride: strides)
    let expectedRows = ((inputSize.rows - filterSize.rows + paddingResult.top + paddingResult.bottom) / strides.0) + 1
    let expectedColumns = ((inputSize.columns - filterSize.columns + paddingResult.left + paddingResult.right) / strides.1) + 1
    
    let paddingInt: UInt32 = padding == .valid ? 0 : 1
    let results: [[Float16]] = NumSwift.zerosLike((expectedRows, expectedColumns))
    
    results.withUnsafeBufferPointer { rBuff in
      signal.withUnsafeBufferPointer { sBuff in
        filter.withUnsafeBufferPointer { fBuff in
          var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
          var sPoint: [UnsafeMutablePointer<Float16>?] = sBuff.map { UnsafeMutablePointer(mutating: $0) }
          var fPoint: [UnsafeMutablePointer<Float16>?] = fBuff.map { UnsafeMutablePointer(mutating: $0) }
          nsc_conv2d_f16(sPoint,
                         fPoint,
                         &rPoint,
                         NSC_Size(rows: Int32(strides.0), columns: Int32(strides.1)),
                         NSC_Padding(rawValue: paddingInt),
                         NSC_Size(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns)),
                         NSC_Size(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns)))
        }
      }
    }
    
    return results
  }
  
  public static func conv1d(signal: [Float16],
                            filter: [Float16],
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> [Float16] {
    
    let paddingResult = padding.extra(inputSize: inputSize, filterSize: filterSize, stride: strides)
    let expectedRows = ((inputSize.rows - filterSize.rows + paddingResult.top + paddingResult.bottom) / strides.0) + 1
    let expectedColumns = ((inputSize.columns - filterSize.columns + paddingResult.left + paddingResult.right) / strides.1) + 1
    
    let paddingInt: UInt32 = padding == .valid ? 0 : 1
    var results: [Float16] = [Float16](repeating: 0, count: expectedRows * expectedColumns)
    
    nsc_conv1d_f16(signal,
                   filter,
                   &results,
                   NSC_Size(rows: Int32(strides.0), columns: Int32(strides.1)),
                   NSC_Padding(rawValue: paddingInt),
                   NSC_Size(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns)),
                   NSC_Size(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns)))
    
    return results
  }
  
  public static func transConv2d(signal: [[Float16]],
                                 filter: [[Float16]],
                                 strides: (Int, Int) = (1,1),
                                 padding: NumSwift.ConvPadding = .valid,
                                 filterSize: (rows: Int, columns: Int),
                                 inputSize: (rows: Int, columns: Int)) -> [[Float16]] {
    
    let paddingInt: UInt32 = padding == .valid ? 0 : 1
    var padLeft = 0
    var padRight = 0
    var padTop = 0
    var padBottom = 0
    
    switch padding {
    case .same:
      padLeft = Int(floor(Double(filterSize.rows - strides.0) / Double(2)))
      padRight = filterSize.rows - strides.0 - padLeft
      padTop = Int(floor(Double(filterSize.columns - strides.1) / Double(2)))
      padBottom = filterSize.columns - strides.1 - padTop
      
    case .valid:
      break
    }
    
    let rows = (inputSize.rows - 1) * strides.0 + filterSize.rows
    let columns = (inputSize.columns - 1) * strides.1 + filterSize.columns
    
    let results: [[Float16]] = NumSwift.zerosLike((rows: (rows - (padTop + padBottom)), columns: (columns - (padLeft + padRight))))
    
    results.withUnsafeBufferPointer { rBuff in
      var rPoint: [UnsafeMutablePointer<Float16>?] = rBuff.map { UnsafeMutablePointer(mutating: $0) }
      
      signal.withUnsafeBufferPointer { aBuff in
        filter.withUnsafeBufferPointer { bBuff in
          let signalPoint: [UnsafeMutablePointer<Float16>?] = aBuff.map { UnsafeMutablePointer(mutating: $0) }
          let filterPoint: [UnsafeMutablePointer<Float16>?] = bBuff.map { UnsafeMutablePointer(mutating: $0) }
          nsc_transConv2d_f16(signalPoint,
                              filterPoint,
                              &rPoint,
                              NSC_Size(rows: Int32(strides.0), columns: Int32(strides.1)),
                              NSC_Padding(rawValue: paddingInt),
                              NSC_Size(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns)),
                              NSC_Size(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns)))
        }
      }
    }
    
    
    return results
  }
  
  
  public static func transConv1d(signal: [Float16],
                                 filter: [Float16],
                                 strides: (Int, Int) = (1,1),
                                 padding: NumSwift.ConvPadding = .valid,
                                 filterSize: (rows: Int, columns: Int),
                                 inputSize: (rows: Int, columns: Int)) -> [Float16] {
    
    let paddingInt: UInt32 = padding == .valid ? 0 : 1
    var padLeft = 0
    var padRight = 0
    var padTop = 0
    var padBottom = 0
    
    switch padding {
    case .same:
      padLeft = Int(floor(Double(filterSize.rows - strides.0) / Double(2)))
      padRight = filterSize.rows - strides.0 - padLeft
      padTop = Int(floor(Double(filterSize.columns - strides.1) / Double(2)))
      padBottom = filterSize.columns - strides.1 - padTop
      
    case .valid:
      break
    }
    
    let rows = (inputSize.rows - 1) * strides.0 + filterSize.rows
    let columns = (inputSize.columns - 1) * strides.1 + filterSize.columns
    var results: [Float16] = [Float16](repeating: 0,
                                       count: (rows - (padTop + padBottom)) * (columns - (padLeft + padRight)))
    
    nsc_transConv1d_f16(signal,
                        filter,
                        &results,
                        NSC_Size(rows: Int32(strides.0), columns: Int32(strides.1)),
                        NSC_Padding(rawValue: paddingInt),
                        NSC_Size(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns)),
                        NSC_Size(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns)))
    return results
  }
  
  public static func zeroPad(signal: [Float16],
                             filterSize: (rows: Int, columns: Int),
                             inputSize: (rows: Int, columns: Int),
                             stride: (Int, Int) = (1,1)) -> [Float16] {
    
    
    let padding = NumSwiftC.paddingCalculation(strides: stride,
                                               padding: .same,
                                               filterSize: filterSize,
                                               inputSize: inputSize)
    
    let count = (inputSize.rows + padding.top + padding.bottom) * (inputSize.columns + padding.left + padding.right)
    
    var results: [Float16] = [Float16](repeating: 0,
                                       count: count)
    
    nsc_zero_pad_f16(signal,
                     &results,
                     NSC_Size(rows: Int32(filterSize.rows),
                              columns: Int32(filterSize.columns)),
                     NSC_Size(rows: Int32(inputSize.rows),
                              columns: Int32(inputSize.columns)),
                     NSC_Size(rows: Int32(stride.0),
                              columns: Int32(stride.1)))
    
    return results
  }
}
#endif
