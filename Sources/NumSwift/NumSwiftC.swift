//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/28/22.
//

import Foundation
import NumSwiftC

public struct NumSwiftC {
  
  public static func conv2d(signal: [Float],
                            filter: [Float],
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> [Float] {
    
    let paddingResult = padding.extra(inputSize: inputSize, filterSize: filterSize, stride: strides)
    let expectedRows = ((inputSize.rows - filterSize.rows + paddingResult.0) / strides.0) + 1
    let expectedColumns = ((inputSize.columns - filterSize.columns + paddingResult.1) / strides.1) + 1
    
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
}
