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
  
  public static func onesLike(_ size: (rows: Int, columns: Int)) -> [[Double]] {
    Array((0..<size.rows).map { _ in [Double](repeating: 1.0, count: size.columns) })
  }
  
  public static func onesLike(_ size: (rows: Int, columns: Int)) -> [[Float]] {
    Array((0..<size.rows).map { _ in [Float](repeating: 1.0, count: size.columns) })
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
  
  public static func zerosLike(_ size: (rows: Int, columns: Int)) -> [[Double]] {
    Array((0..<size.rows).map { _ in [Double](repeating: 0.0, count: size.columns) })
  }
  
  public static func zerosLike(_ size: (rows: Int, columns: Int)) -> [[Float]] {
    Array((0..<size.rows).map { _ in [Float](repeating: 0.0, count: size.columns) })
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
  
  @available(*, deprecated, renamed: "NumSwiftC.conv2d", message: "This function is slow. Use the NumSwiftC equivalent")
  public static func conv2d(signal: [[Float]],
                            filter: [[Float]],
                            strides: (Int, Int) = (1,1),
                            padding: NumSwift.ConvPadding = .valid,
                            filterSize: (rows: Int, columns: Int),
                            inputSize: (rows: Int, columns: Int)) -> [[Float]] {
    
    let paddingValue = padding.extra(inputSize: (inputSize.rows, inputSize.columns), filterSize: filterSize)
    
    let rows = (((inputSize.rows + (paddingValue.top + paddingValue.bottom)) - (filterSize.rows - 1) - 1) / strides.1) + 1
    let columns = (((inputSize.columns + (paddingValue.left + paddingValue.right)) - (filterSize.columns - 1) - 1) / strides.0) + 1
    
    let flatFilter: [Float] = filter.flatten()
    
    var results: [[Float]] = NumSwift.zerosLike((rows,columns))
    
    var currentRowIndex = 0
    var currentColIndex = 0
    
    let columnRange = -paddingValue.left...(inputSize.columns + paddingValue.right) - filterSize.columns
    let rowRange = -paddingValue.top...(inputSize.rows + paddingValue.bottom) - filterSize.rows
    
    let strideColumnRange = [Int](stride(from: columnRange.lowerBound, through: columnRange.upperBound, by: strides.0))
    let strideRowRange = [Int](stride(from: rowRange.lowerBound, through: rowRange.upperBound, by: strides.1))
    
    strideColumnRange.forEach { c in
      currentRowIndex = 0
      
      strideRowRange.forEach { r in
        let sig = signal[flat: (r,c), (r + filterSize.rows, c + filterSize.columns), 0]
        let dot = sig.dot(flatFilter)
        
        results[currentRowIndex][currentColIndex] = dot
        currentRowIndex += 1
      }
      
      currentColIndex += 1
    }
    
    return results
  }
  
  @available(*, deprecated, renamed: "NumSwiftC.transConv2D", message: "This function is slow. Use the NumSwiftC equivalent")
  public static func transConv2D(signal: [[Double]],
                                 filter: [[Double]],
                                 strides: (Int, Int) = (1,1),
                                 padding: ConvPadding = .valid,
                                 filterSize: (Int, Int),
                                 inputSize: (Int, Int)) -> [[Double]] {
    let rF = filterSize.0
    let cF = filterSize.1
    let rS = inputSize.0
    let cS = inputSize.1
    
    let rows = (rS - 1) * strides.0 + rF
    let columns = (cS - 1) * strides.1 + cF
    
    var result: [[Double]] = NumSwift.zerosLike((rows: rows,
                                                 columns: columns))
    
    for i in 0..<rS {
      let iPrime = i * strides.0
      
      for j in 0..<cS {
        
        let jPrime = j * strides.1
        
        for r in 0..<rF {
          for c in 0..<cF {
            
            result[iPrime + r][jPrime + c] += signal[i][j] * filter[r][c]
          }
        }
      }
    }
    
    var padLeft = 0
    var padRight = 0
    var padTop = 0
    var padBottom = 0
    
    switch padding {
    case .same:
      padLeft = Int(floor(Double(rF - strides.0) / Double(2)))
      padRight = rF - strides.0 - padLeft
      padTop = Int(floor(Double(cF - strides.1) / Double(2)))
      padBottom = cF - strides.1 - padTop
      
    case .valid:
      break
    }
    
    let padded = Array(result[padLeft..<rows - padRight].map { Array($0[padTop..<columns - padBottom]) })
    return padded
  }
  
  @available(*, deprecated, renamed: "NumSwiftC.transConv2D", message: "This function is slow. Use the NumSwiftC equivalent")
  public static func transConv2D(signal: [[Float]],
                                 filter: [[Float]],
                                 strides: (Int, Int) = (1,1),
                                 padding: ConvPadding = .valid,
                                 filterSize: (Int, Int),
                                 inputSize: (Int, Int)) -> [[Float]] {
    let rF = filterSize.0
    let cF = filterSize.1
    let rS = inputSize.0
    let cS = inputSize.1
    
    let rows = (rS - 1) * strides.0 + rF
    let columns = (cS - 1) * strides.1 + cF
    
    var result: [[Float]] = NumSwift.zerosLike((rows: rows,
                                                columns: columns))
    
    for i in 0..<rS {
      let iPrime = i * strides.0
      
      for j in 0..<cS {
        
        let jPrime = j * strides.1
        
        for r in 0..<rF {
          for c in 0..<cF {
            
            result[iPrime + r][jPrime + c] += signal[i][j] * filter[r][c]
          }
        }
      }
    }
    
    var padLeft = 0
    var padRight = 0
    var padTop = 0
    var padBottom = 0
    
    switch padding {
    case .same:
      padLeft = Int(floor(Double(rF - strides.0) / Double(2)))
      padRight = rF - strides.0 - padLeft
      padTop = Int(floor(Double(cF - strides.1) / Double(2)))
      padBottom = cF - strides.1 - padTop
      
    case .valid:
      break
    }
    
    let padded = Array(result[padTop..<rows - padBottom].map { Array($0[padLeft..<columns - padRight]) })
    return padded
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

