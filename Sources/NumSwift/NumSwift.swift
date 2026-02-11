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

public struct NumSwiftPadding {
  public var top: Int
  public var left: Int
  public var right: Int
  public var bottom: Int
  
  public init(top: Int, left: Int, right: Int, bottom: Int) {
    self.top = top
    self.left = left
    self.right = right
    self.bottom = bottom
  }
}

public class NumSwift {
  
  public enum ConvPadding {
    case same, valid
    
    public func extra(inputSize: (Int, Int),
                      filterSize: (Int, Int),
                      stride: (Int, Int) = (1,1)) -> (top: Int, bottom: Int, left: Int, right: Int) {
      switch self {
      case .same:
        //((S-1)*W-S+F)/2
        let height = Double(inputSize.0)
        let width = Double(inputSize.1)
        
        let outHeight = ceil(height / Double(stride.0))
        let outWidth = ceil(width / Double(stride.1))
        
        let padAlongHeight = Swift.max((outHeight - 1) * Double(stride.0) + Double(filterSize.0) - height, 0)
        let padAlongWidth = Swift.max((outWidth - 1) * Double(stride.1) + Double(filterSize.1) - width, 0)
        
        let paddingTop = Int(floor(padAlongHeight / 2))
        let paddingBottom = Int(padAlongHeight - Double(paddingTop))
        let paddingLeft = Int(floor(padAlongWidth / 2))
        let paddingRight = Int(padAlongWidth - Double(paddingLeft))
        
        return (paddingTop, paddingBottom, paddingLeft, paddingRight)
      case .valid:
        return (0,0,0,0)
      }
    }
  }

  public static func randomChoice<T: Equatable>(in array: [T], p: [Float] = []) -> (T, Int) {
    guard p.isEmpty == false, p.count == array.count else {
      let index = Int.random(in: 0..<array.count)
      return (array[index], index)
    }
    
    var distributionArray: [T] = []
    
    for i in 0..<p.count {
      let prob = Int(ceil(p[i] * 100))
      let a = array[i]
      
      distributionArray.append(contentsOf: [T](repeating: a, count: prob))
    }
    
    guard distributionArray.isEmpty == false else {
      let index = Int.random(in: 0..<array.count)
      return (array[index], index)
    }
    
    let random = Int.random(in: 0..<distributionArray.count)
    let arrayIndex = array.firstIndex(where: { distributionArray[random] == $0 })
    return (distributionArray[random], arrayIndex ?? 0)
  }
  
  public static func onesLike(_ size: (rows: Int, columns: Int, depth: Int)) -> [[[Float]]] {
    let totalElements = size.rows * size.columns * size.depth
    
    if totalElements > 10000 {
      var result = ContiguousArray<ContiguousArray<ContiguousArray<Float>>>()
      result.reserveCapacity(size.depth)
      
      for _ in 0..<size.depth {
        var depthSlice = ContiguousArray<ContiguousArray<Float>>()
        depthSlice.reserveCapacity(size.rows)
        
        for _ in 0..<size.rows {
          let row = ContiguousArray<Float>(unsafeUninitializedCapacity: size.columns) { buffer, initializedCount in
            var one: Float = 1.0
            vDSP_vfill(&one, buffer.baseAddress!, 1, vDSP_Length(size.columns))
            initializedCount = size.columns
          }
          depthSlice.append(row)
        }
        result.append(depthSlice)
      }
      
      return Array(result.map { depthSlice in Array(depthSlice.map { row in Array(row) }) })
    } else {
      var result: [[[Float]]]  = []
      
      for _ in 0..<size.depth {
        var row: [[Float]] = []
        for _ in 0..<size.rows {
          row.append([Float](repeating: 1.0, count: size.columns))
        }
        result.append(row)
      }

      return result
    }
  }

  public static func onesLike(_ size: (rows: Int, columns: Int)) -> [[Double]] {
    let totalElements = size.rows * size.columns
    
    if totalElements > 1000 {
      var result = ContiguousArray<ContiguousArray<Double>>()
      result.reserveCapacity(size.rows)
      
      for _ in 0..<size.rows {
        let row = ContiguousArray<Double>(unsafeUninitializedCapacity: size.columns) { buffer, initializedCount in
          var one: Double = 1.0
          vDSP_vfillD(&one, buffer.baseAddress!, 1, vDSP_Length(size.columns))
          initializedCount = size.columns
        }
        result.append(row)
      }
      
      return Array(result.map { Array($0) })
    } else {
      return Array((0..<size.rows).map { _ in [Double](repeating: 1.0, count: size.columns) })
    }
  }
  
  public static func onesLike(_ size: (rows: Int, columns: Int)) -> [[Float]] {
    let totalElements = size.rows * size.columns
    
    if totalElements > 1000 {
      var result = ContiguousArray<ContiguousArray<Float>>()
      result.reserveCapacity(size.rows)
      
      for _ in 0..<size.rows {
        let row = ContiguousArray<Float>(unsafeUninitializedCapacity: size.columns) { buffer, initializedCount in
          var one: Float = 1.0
          vDSP_vfill(&one, buffer.baseAddress!, 1, vDSP_Length(size.columns))
          initializedCount = size.columns
        }
        result.append(row)
      }
      
      return Array(result.map { Array($0) })
    } else {
      return Array((0..<size.rows).map { _ in [Float](repeating: 1.0, count: size.columns) })
    }
  }
  
  public static func onesLike<T: Collection>(_ array: T) -> Array<AnyHashable> {
    let shape = array.shapeOf
    
    var result: [AnyHashable]  = []
    var previous: AnyHashable = 1.0
    
    shape.forEach { s in
      result = Array(repeatElement(previous, count: s))
      previous = result
    }
    
    return result
  }
  
  public static func zerosLike(_ size: (rows: Int, columns: Int, depth: Int)) -> [[[Float]]] {
    let totalElements = size.rows * size.columns * size.depth
    
    if totalElements > 10000 {
      var result = ContiguousArray<ContiguousArray<ContiguousArray<Float>>>()
      result.reserveCapacity(size.depth)
      
      for _ in 0..<size.depth {
        var depthSlice = ContiguousArray<ContiguousArray<Float>>()
        depthSlice.reserveCapacity(size.rows)
        
        for _ in 0..<size.rows {
          var row = ContiguousArray<Float>(unsafeUninitializedCapacity: size.columns) { buffer, initializedCount in
            vDSP_vclr(buffer.baseAddress!, 1, vDSP_Length(size.columns))
            initializedCount = size.columns
          }
          depthSlice.append(row)
        }
        result.append(depthSlice)
      }
      
      return Array(result.map { depthSlice in Array(depthSlice.map { row in Array(row) }) })
    } else {
      var result: [[[Float]]]  = []
      
      for _ in 0..<size.depth {
        var row: [[Float]] = []
        for _ in 0..<size.rows {
          row.append([Float](repeating: 0, count: size.columns))
        }
        result.append(row)
      }

      return result
    }
  }
  
  public static func zerosLike(_ size: (rows: Int, columns: Int)) -> [[Double]] {
    let totalElements = size.rows * size.columns
    
    if totalElements > 1000 {
      var result = ContiguousArray<ContiguousArray<Double>>()
      result.reserveCapacity(size.rows)
      
      for _ in 0..<size.rows {
        let row = ContiguousArray<Double>(unsafeUninitializedCapacity: size.columns) { buffer, initializedCount in
          vDSP_vclrD(buffer.baseAddress!, 1, vDSP_Length(size.columns))
          initializedCount = size.columns
        }
        result.append(row)
      }
      
      return Array(result.map { Array($0) })
    } else {
      return Array((0..<size.rows).map { _ in [Double](repeating: 0.0, count: size.columns) })
    }
  }
  
  public static func zerosLike(_ size: (rows: Int, columns: Int)) -> [[Float]] {
    let totalElements = size.rows * size.columns
    
    if totalElements > 1000 {
      var result = ContiguousArray<ContiguousArray<Float>>()
      result.reserveCapacity(size.rows)
      
      for _ in 0..<size.rows {
        let row = ContiguousArray<Float>(unsafeUninitializedCapacity: size.columns) { buffer, initializedCount in
          vDSP_vclr(buffer.baseAddress!, 1, vDSP_Length(size.columns))
          initializedCount = size.columns
        }
        result.append(row)
      }
      
      return Array(result.map { Array($0) })
    } else {
      return Array((0..<size.rows).map { _ in [Float](repeating: 0.0, count: size.columns) })
    }
  }
  
  
  public static func zerosLike<T: Collection>(_ array: T) -> Array<AnyHashable> {
    let shape = array.shapeOf
    
    var result: [AnyHashable]  = []
    var previous: AnyHashable = Float.zero
    
    shape.forEach { s in
      result = Array(repeatElement(previous, count: s))
      previous = result
    }
    
    return result
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



#if arch(arm64)
// MARK: Float16
extension NumSwift {
  
  public static func zerosLike(_ size: (rows: Int, columns: Int, depth: Int)) -> [[[Float16]]] {
    let totalElements = size.rows * size.columns * size.depth
    
    if totalElements > 10000 {
      var result = ContiguousArray<ContiguousArray<ContiguousArray<Float16>>>()
      result.reserveCapacity(size.depth)
      
      for _ in 0..<size.depth {
        var depthSlice = ContiguousArray<ContiguousArray<Float16>>()
        depthSlice.reserveCapacity(size.rows)
        
        for _ in 0..<size.rows {
          let row = ContiguousArray<Float16>(repeating: 0.0, count: size.columns)
          depthSlice.append(row)
        }
        result.append(depthSlice)
      }
      
      return Array(result.map { depthSlice in Array(depthSlice.map { row in Array(row) }) })
    } else {
      var result: [[[Float16]]]  = []
      
      for _ in 0..<size.depth {
        var row: [[Float16]] = []
        for _ in 0..<size.rows {
          row.append([Float16](repeating: 0, count: size.columns))
        }
        result.append(row)
      }

      return result
    }
  }
  
  public static func onesLike(_ size: (rows: Int, columns: Int, depth: Int)) -> [[[Float16]]] {
    let totalElements = size.rows * size.columns * size.depth
    
    if totalElements > 10000 {
      var result = ContiguousArray<ContiguousArray<ContiguousArray<Float16>>>()
      result.reserveCapacity(size.depth)
      
      for _ in 0..<size.depth {
        var depthSlice = ContiguousArray<ContiguousArray<Float16>>()
        depthSlice.reserveCapacity(size.rows)
        
        for _ in 0..<size.rows {
          let row = ContiguousArray<Float16>(repeating: 1.0, count: size.columns)
          depthSlice.append(row)
        }
        result.append(depthSlice)
      }
      
      return Array(result.map { depthSlice in Array(depthSlice.map { row in Array(row) }) })
    } else {
      var result: [[[Float16]]]  = []
      
      for _ in 0..<size.depth {
        var row: [[Float16]] = []
        for _ in 0..<size.rows {
          row.append([Float16](repeating: 1.0, count: size.columns))
        }
        result.append(row)
      }

      return result
    }
  }
  
  
  public static func randomChoice<T: Equatable>(in array: [T], p: [Float16] = []) -> (T, Int) {
    guard p.isEmpty == false, p.count == array.count else {
      let index = Int.random(in: 0..<array.count)
      return (array[index], index)
    }
    
    var distributionArray: [T] = []
    
    for i in 0..<p.count {
      let prob = Int(ceil(p[i] * 100))
      let a = array[i]
      
      distributionArray.append(contentsOf: [T](repeating: a, count: prob))
    }
    
    guard distributionArray.isEmpty == false else {
      let index = Int.random(in: 0..<array.count)
      return (array[index], index)
    }
    
    let random = Int.random(in: 0..<distributionArray.count)
    let arrayIndex = array.firstIndex(where: { distributionArray[random] == $0 })
    return (distributionArray[random], arrayIndex ?? 0)
  }
  
  
  public static func zerosLike(_ size: (rows: Int, columns: Int)) -> [[Float16]] {
    let totalElements = size.rows * size.columns
    
    if totalElements > 1000 {
      var result = ContiguousArray<ContiguousArray<Float16>>()
      result.reserveCapacity(size.rows)
      
      for _ in 0..<size.rows {
        let row = ContiguousArray<Float16>(repeating: 0.0, count: size.columns)
        result.append(row)
      }
      
      return Array(result.map { Array($0) })
    } else {
      return Array((0..<size.rows).map { _ in [Float16](repeating: 0.0, count: size.columns) })
    }
  }
  
  public static func onesLike(_ size: (rows: Int, columns: Int)) -> [[Float16]] {
    let totalElements = size.rows * size.columns
    
    if totalElements > 1000 {
      var result = ContiguousArray<ContiguousArray<Float16>>()
      result.reserveCapacity(size.rows)
      
      for _ in 0..<size.rows {
        let row = ContiguousArray<Float16>(repeating: 1.0, count: size.columns)
        result.append(row)
      }
      
      return Array(result.map { Array($0) })
    } else {
      return Array((0..<size.rows).map { _ in [Float16](repeating: 1.0, count: size.columns) })
    }
  }
  
}

#endif
