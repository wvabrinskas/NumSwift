//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/28/22.
//

import Foundation
import NumSwiftC

public struct NumSwiftC {
  
  public static func zeroPad(signal: [[Float]],
                             padding: NumSwiftPadding) -> [[Float]] {
    
    let shape = signal.shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    
    let expectedRows = rows + padding.top + padding.bottom
    let expectedColumns = columns + padding.left + padding.right
    var results: [Float] = [Float](repeating: 0, count: expectedRows * expectedColumns)

    let flatSignal: [Float] = signal.flatten()
    nsc_specific_zero_pad(flatSignal,
                          &results,
                          NSC_Size(rows: Int32(rows),
                                   columns: Int32(columns)),
                          Int32(padding.top),
                          Int32(padding.bottom),
                          Int32(padding.left),
                          Int32(padding.right))
    
    return results.reshape(columns: expectedColumns)
  }
  
  public static func conv2d(signal: [Float],
                            filter: [Float],
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> [Float] {
    
    let paddingResult = padding.extra(inputSize: inputSize, filterSize: filterSize, stride: strides)
    let expectedRows = ((inputSize.rows - filterSize.rows + paddingResult.top + paddingResult.bottom) / strides.0) + 1
    let expectedColumns = ((inputSize.columns - filterSize.columns + paddingResult.left + paddingResult.right) / strides.1) + 1
    
    let paddingInt: UInt32 = padding == .valid ? 0 : 1
    var results: [Float] = [Float](repeating: 0, count: expectedRows * expectedColumns)
    
    nsc_conv2d(signal,
               filter,
               &results,
               NSC_Size(rows: Int32(strides.0), columns: Int32(strides.1)),
               NSC_Padding(rawValue: paddingInt),
               NSC_Size(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns)),
               NSC_Size(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns)))
    
    return results
  }
  
  public static func transConv2d(signal: [Float],
                                 filter: [Float],
                                 strides: (Int, Int) = (1,1),
                                 padding: NumSwift.ConvPadding = .valid,
                                 filterSize: (rows: Int, columns: Int),
                                 inputSize: (rows: Int, columns: Int)) -> [Float] {
    
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
    var results: [Float] = [Float](repeating: 0,
                                   count: (rows - (padTop + padBottom)) * (columns - (padLeft + padRight)))
    
    nsc_transConv2d(signal,
                    filter,
                    &results,
                    NSC_Size(rows: Int32(strides.0), columns: Int32(strides.1)),
                    NSC_Padding(rawValue: paddingInt),
                    NSC_Size(rows: Int32(filterSize.rows), columns: Int32(filterSize.columns)),
                    NSC_Size(rows: Int32(inputSize.rows), columns: Int32(inputSize.columns)))
    return results
  }
  
  public static func paddingCalculation(strides: (Int, Int) = (1,1),
                                        padding: NumSwift.ConvPadding = .valid,
                                        filterSize: (rows: Int, columns: Int),
                                        inputSize: (rows: Int, columns: Int)) -> (top: Int, bottom: Int, left: Int, right: Int) {
    let paddingInt: UInt32 = padding == .valid ? 0 : 1

    var left: Int32 = 0
    var right: Int32 = 0
    var top: Int32 = 0
    var bottom: Int32 = 0
    
    nsc_padding_calculation(NSC_Size(rows: Int32(strides.0),
                                     columns: Int32(strides.1)),
                            NSC_Padding(paddingInt),
                            NSC_Size(rows: Int32(filterSize.rows),
                                     columns: Int32(filterSize.columns)),
                            NSC_Size(rows: Int32(inputSize.rows),
                                     columns: Int32(inputSize.columns)),
                            &top,
                            &bottom,
                            &left,
                            &right)
    
    return (Int(top), Int(bottom), Int(left), Int(right))
  }
  
  public static func zeroPad(signal: [Float],
                             filterSize: (rows: Int, columns: Int),
                             inputSize: (rows: Int, columns: Int),
                             stride: (Int, Int) = (1,1)) -> [Float] {
    
    
    let padding = NumSwiftC.paddingCalculation(strides: stride,
                                               padding: .same,
                                               filterSize: filterSize,
                                               inputSize: inputSize)
    
    let count = (inputSize.rows + padding.top + padding.bottom) * (inputSize.columns + padding.left + padding.right)

    var results: [Float] = [Float](repeating: 0,
                                   count: count)
    
    nsc_zero_pad(signal,
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
