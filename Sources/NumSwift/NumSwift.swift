//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/18/22.
//

import Foundation
import Accelerate
import Metal
import MetalPerformanceShaders
#if os(iOS)
import UIKit
#endif

public class NumSwift {
  
  public static func zerosLike(_ size: (rows: Int, columns: Int, depth: Int)) -> [[[Float]]] {
    let shape = [size.columns, size.rows, size.depth]
    
    var result: [Any]  = []
    var previous: Any = Float.zero
    
    shape.forEach { s in
      result = Array(repeatElement(previous, count: s))
      previous = result
    }
    
    return result as? [[[Float]]] ?? []
  }
  
  public static func zerosLike(_ size: (rows: Int, columns: Int)) -> [[Double]] {
    let shape = [size.columns, size.rows]
    
    var result: [Any]  = []
    var previous: Any = Double.zero
    
    shape.forEach { s in
      result = Array(repeatElement(previous, count: s))
      previous = result
    }
    
    return result as? [[Double]] ?? []
  }
  
  public static func zerosLike(_ size: (rows: Int, columns: Int)) -> [[Float]] {
    let shape = [size.columns, size.rows]
    
    var result: [Any]  = []
    var previous: Any = Float.zero
    
    shape.forEach { s in
      result = Array(repeatElement(previous, count: s))
      previous = result
    }
    
    return result as? [[Float]] ?? []
  }
  
  public static func zerosLike<T: Collection>(_ array: T) -> Array<AnyHashable> {
    let shape = array.shape
    
    var result: [AnyHashable]  = []
    var previous: AnyHashable = Float.zero
    
    shape.forEach { s in
      result = Array(repeatElement(previous, count: s))
      previous = result
    }
    
    return result
  }
  
  public static func conv2dValid(signal: [[Float]], filter: [[Float]]) -> [[Float]] {
    let filterShape = filter.shape
    
    guard let rf = filterShape[safe: 0],
          let cf = filterShape[safe: 1] else {
      return []
    }
    
    let shape = signal.shape
    
    guard let rd = shape[safe: 0],
          let cd = shape[safe: 1] else {
      return []
    }
    
    var results: [[Float]] = []
    
    for r in 0...rd - rf {
      var result: [Float] = []
      
      for c in 0...cd - cf {
        
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
  
  public static func conv2dValidD(signal: [[Double]], filter: [[Double]]) -> [[Double]] {
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


#if os(iOS)
public extension UIImage {
  
  /// Returns the various color layers as a 3D array
  /// - Parameter alpha: if the result should include the alpha layer
  /// - Returns: the colors mapped to their own 2D array in a 3D array as (R, G, B) and optionall (R, G, B, A)
  func layers(alpha: Bool = false) -> [[[Float]]] {
    guard let cgImage = self.cgImage,
          let provider = cgImage.dataProvider else {
      return []
    }
    
    let providerData = provider.data
    
    if let data = CFDataGetBytePtr(providerData) {
      
      var rVals: [[Float]] = []
      var gVals: [[Float]] = []
      var bVals: [[Float]] = []
      var aVals: [[Float]] = []
      
      for y in 0..<Int(size.height) {
        var rowR: [Float] = []
        var rowG: [Float] = []
        var rowB: [Float] = []
        var rowA: [Float] = []
        
        for x in 0..<Int(size.width) {
          let numberOfComponents = 4
          let pixelData = ((Int(size.width) * y) + x) * numberOfComponents
          
          let r = Float(data[pixelData]) / 255.0
          let g = Float(data[pixelData + 1]) / 255.0
          let b = Float(data[pixelData + 2]) / 255.0
          let a = Float(data[pixelData + 3]) / 255.0
          
          rowR.append(r)
          rowG.append(g)
          rowB.append(b)
          rowA.append(a)
        }
        
        rVals.append(rowR)
        gVals.append(rowG)
        bVals.append(rowB)
        aVals.append(rowA)
      }
      
      var results: [[[Float]]] = [rVals, gVals, bVals]
      if alpha {
        results.append(aVals)
      }
      
      return results
    }
    
    return []
  }
}
#endif


//MARK: Metal
public extension NumSwift {
  
  typealias GPUData = (data: [Float], size: (rows: Int, columns: Int))
  
  class GPU {
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    
    
    public init() {
      device = MTLCreateSystemDefaultDevice()
      commandQueue = device?.makeCommandQueue()
    }
    
    public func matrixMult(a: GPUData, b: GPUData) -> [Float] {
      guard let device = device else {
        return []
      }
      
      let arrayA = a.data
      let arrayB = b.data
      
      let rowsA = a.size.rows
      let columnsA = a.size.columns
      
      let rowsB = b.size.rows
      let columnsB = b.size.columns
      
      let rowsC = rowsA
      let columnsC = columnsB
      
      let aSize = rowsA * columnsA * MemoryLayout<Float>.stride
      let bSize = rowsB * columnsB * MemoryLayout<Float>.stride
      let cSize = rowsC * columnsC * MemoryLayout<Float>.stride
      
      guard let bufferA = device.makeBuffer(bytes: arrayA,
                                            length: aSize,
                                            options: []),
              let bufferB = device.makeBuffer(bytes: arrayB,
                                              length: bSize,
                                              options: []),
            let bufferC = device.makeBuffer(length: cSize,
                                            options: []) else {
        return []
      }
      
      
      let descA = MPSMatrixDescriptor(rows: rowsA,
                                      columns: columnsA,
                                      rowBytes: aSize / rowsA,
                                      dataType: .float32)
      
      let descB = MPSMatrixDescriptor(rows: rowsB,
                                      columns: columnsB,
                                      rowBytes: bSize / rowsB,
                                      dataType: .float32)
      
      let descC =  MPSMatrixDescriptor(rows: rowsC,
                                       columns: columnsC,
                                       rowBytes: cSize / rowsC,
                                       dataType: .float32)

      let matrixA = MPSMatrix(buffer: bufferA, descriptor: descA)
      let matrixB = MPSMatrix(buffer: bufferB, descriptor: descB)
      let matrixC = MPSMatrix(buffer: bufferC, descriptor: descC)
      
      let matrixMultiplication = MPSMatrixMultiplication(device: device,
                                                         transposeLeft: true,
                                                         transposeRight: false,
                                                         resultRows: rowsC,
                                                         resultColumns: columnsC,
                                                         interiorColumns: columnsA,
                                                         alpha: 1,
                                                         beta: 0)
      
      guard let commandBuffer = commandQueue?.makeCommandBuffer() else {
        return []
      }
      
      matrixMultiplication.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
      
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      
      var output: [Float] = []
      let rawPointer = matrixC.data.contents()
      let typePointer = rawPointer.bindMemory(to: Float.self, capacity: rowsC * columnsC)
      let bufferPointer = UnsafeBufferPointer(start: typePointer, count: rowsC * columnsC)
      
      let _ = bufferPointer.map { value in 
        output += [value]
      }
        
      return output
    
    }
    
  }
  
}
