//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/18/22.
//

import Foundation
import Accelerate

public struct NumSwift {
  
  static func conv2dValid(signal: [[Float]], filter: [[Float]]) -> [[Float]] {
    let filterShape = filter.shape

    guard let rf = filterShape[safe: 0],
          let cf = filterShape[safe: 1] else {
            return []
          }
    
    let shape = signal.shape

    if let rd = shape[safe: 0],
        let cd = shape[safe: 1] {
      
      var results: [[Float]] = []
      
      for r in 0..<rd - rf {
        var result: [Float] = []
        
        for c in 0..<cd - cf {
          
          var sum: Float = 0
          for fr in 0..<rf {
            let dataRow = Array(signal[r + fr][c..<c + cf])
            let filterRow = filter[fr]
            let mult = (filterRow * dataRow).sum
            sum += mult
          }
          result.append(sum)
        }
        
        results.append(result)
      }
      
      return results
    }
    
    return []
  }
  
  static func conv2dValidD(signal: [[Double]], filter: [[Double]]) -> [[Double]] {
    let filterShape = filter.shape

    guard let rf = filterShape[safe: 0],
          let cf = filterShape[safe: 1] else {
            return []
          }
    
    let shape = signal.shape

    if let rd = shape[safe: 0],
        let cd = shape[safe: 1] {
      
      var results: [[Double]] = []
      
      for r in 0..<rd - rf {
        var result: [Double] = []
        
        for c in 0..<cd - cf {
          
          var sum: Double = 0
          for fr in 0..<rf {
            let dataRow = Array(signal[r + fr][c..<c + cf])
            let filterRow = filter[fr]
            let mult = (filterRow * dataRow).sum
            sum += mult
          }
          result.append(sum)
        }
        
        results.append(result)
      }
      
      return results
    }
    
    return []
  }
  
}
