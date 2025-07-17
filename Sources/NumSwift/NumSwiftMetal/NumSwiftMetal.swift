import Foundation
import Metal
import MetalKit
import NumSwiftC
import Numerics

// MARK: - Metal Backend Configuration

public enum ComputeBackend {
  case cpu
  case metal
  case auto // Automatically chooses based on problem size
}

public enum ActivationType: UInt32 {
  case relu = 0
  case leakyRelu = 1
  case sigmoid = 2
  case swish = 3
  case tanh = 4
  case none = 5
  case selu = 6
  case gelu = 7
  case softmax = 8
}

public struct MetalConfiguration {
  public let device: MTLDevice
  public let commandQueue: MTLCommandQueue
  public let library: MTLLibrary
  
  public init?(device: MTLDevice? = nil) {
    guard let device = device ?? MTLCreateSystemDefaultDevice() else {
      return nil
    }
    
    self.device = device
    
    guard let commandQueue = self.device.makeCommandQueue() else {
      return nil
    }
    
    self.commandQueue = commandQueue
    
    var library: MTLLibrary?
    let bundle = Bundle.module
    
    library = try? device.makeDefaultLibrary(bundle: bundle)

    if library == nil {
      library = device.makeDefaultLibrary()
    }
    
    guard let metalLibrary = library else {
      print("Failed to load Metal library")
      return nil
    }
    
    self.library = metalLibrary
  }
}

// MARK: - NumSwift Metal Backend

public class NumSwiftMetal {
  private let config: MetalConfiguration
  private var backend: ComputeBackend = .metal
  private let update = NSRecursiveLock()
  
  // Cache for frequently used compute pipelines
  private var pipelineCache: [String: MTLComputePipelineState] = [:]
  
  public init?(configuration: MetalConfiguration? = nil) {
    if let config = configuration {
      self.config = config
    } else {
      guard let config = MetalConfiguration() else { return nil }
      self.config = config
    }
  }
  
  public func setBackend(_ backend: ComputeBackend) {
    self.backend = backend
  }
  
  // MARK: - Pipeline Management
  
  private func computePipeline(for functionName: String) -> MTLComputePipelineState? {
    defer { update.unlock() }
    update.lock()
    if let cached = pipelineCache[functionName] {
      return cached
    }
    
    guard let function = config.library.makeFunction(name: functionName) else {
      print("Failed to create function: \(functionName)")
      print("Available functions in library: \(config.library.functionNames)")
      return nil
    }
    
    do {
      let pipeline = try config.device.makeComputePipelineState(function: function)
      pipelineCache[functionName] = pipeline
      return pipeline
    } catch {
      print("Failed to create pipeline for \(functionName): \(error)")
      return nil
    }
  }
  
  // MARK: - Backend Selection Logic
  
  private func shouldUseMetal(for elementCount: Int) -> Bool {
    switch backend {
    case .cpu:
      return false
    case .metal:
      return true
    case .auto:
      // Use Metal for larger problems where parallelization benefits outweigh overhead
      return elementCount > 1000
    }
  }
  
  // MARK: - Buffer Management
  
  private func createBuffer<T>(from data: [T], type: T.Type) -> MTLBuffer? {
    let size = data.count * MemoryLayout<T>.stride
    return config.device.makeBuffer(bytes: data, length: size, options: .storageModeShared)
  }
  
  private func createBuffer<T>(count: Int, type: T.Type) -> MTLBuffer? {
    let size = count * MemoryLayout<T>.stride
    return config.device.makeBuffer(length: size, options: .storageModeShared)
  }
  
  // MARK: - Basic Array Operations
  
  public func sum(_ array: [Float]) -> Float {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.reduce(0, +)
    }
    
    guard let pipeline = computePipeline(for: "nsc_sum_float_kernel"),
          let inputBuffer = createBuffer(from: array, type: Float.self),
          let resultBuffer = createBuffer(count: 1, type: Float.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.reduce(0, +)
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return 0
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(sizeBuffer, offset: 0, index: 2)
    
    let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1)
    return resultPointer[0]
  }
  
  public func sum(_ array: [Float16]) -> Float16 {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_sum(ptr.baseAddress!)
      }
    }
    
    guard let pipeline = computePipeline(for: "nsc_sum_kernel"),
          let inputBuffer = createBuffer(from: array, type: Float16.self),
          let resultBuffer = createBuffer(count: 1, type: Float16.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      // Fallback to CPU implementation
      return array.withUnsafeBufferPointer { ptr in
        return nsc_sum(ptr.baseAddress!)
      }
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return 0
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(sizeBuffer, offset: 0, index: 2)
    
    let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: 1)
    return resultPointer[0]
  }
  
  public func max(_ array: [Float]) -> Float {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.max() ?? 0
    }
    
    guard let pipeline = computePipeline(for: "nsc_max_float_kernel"),
          let inputBuffer = createBuffer(from: array, type: Float.self),
          let resultBuffer = createBuffer(count: 1, type: Float.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.max() ?? 0
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return 0
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(sizeBuffer, offset: 0, index: 2)
    
    let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1)
    return resultPointer[0]
  }
  
  public func min(_ array: [Float]) -> Float {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.min() ?? 0
    }
    
    guard let pipeline = computePipeline(for: "nsc_min_float_kernel"),
          let inputBuffer = createBuffer(from: array, type: Float.self),
          let resultBuffer = createBuffer(count: 1, type: Float.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.min() ?? 0
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return 0
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(sizeBuffer, offset: 0, index: 2)
    
    let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1)
    return resultPointer[0]
  }
  
  public func max(_ array: [Float16]) -> Float16 {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_max(ptr.baseAddress!)
      }
    }
    
    guard let pipeline = computePipeline(for: "nsc_max_kernel"),
          let inputBuffer = createBuffer(from: array, type: Float16.self),
          let resultBuffer = createBuffer(count: 1, type: Float16.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_max(ptr.baseAddress!)
      }
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return 0
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(sizeBuffer, offset: 0, index: 2)
    
    let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: 1)
    return resultPointer[0]
  }
  
  // MARK: - Element-wise Operations
  
  public func add(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
    let elementCount = lhs.count
    guard elementCount == rhs.count else {
      fatalError("Array sizes must match")
    }
    
    if !shouldUseMetal(for: elementCount) {
      return zip(lhs, rhs).map(+)
    }
    
    guard let pipeline = computePipeline(for: "nsc_add_float_kernel"),
          let lhsBuffer = createBuffer(from: lhs, type: Float.self),
          let rhsBuffer = createBuffer(from: rhs, type: Float.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float.self) else {
      return zip(lhs, rhs).map(+)
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
    encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  public func subtract(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
    let elementCount = lhs.count
    guard elementCount == rhs.count else {
      fatalError("Array sizes must match")
    }
    
    if !shouldUseMetal(for: elementCount) {
      return zip(lhs, rhs).map(-)
    }
    
    guard let pipeline = computePipeline(for: "nsc_sub_float_kernel"),
          let lhsBuffer = createBuffer(from: lhs, type: Float.self),
          let rhsBuffer = createBuffer(from: rhs, type: Float.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float.self) else {
      return zip(lhs, rhs).map(-)
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
    encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  public func add(_ lhs: [Float16], _ rhs: [Float16]) -> [Float16] {
    let elementCount = lhs.count
    guard elementCount == rhs.count else {
      fatalError("Array sizes must match")
    }
    
    if !shouldUseMetal(for: elementCount) {
      var result = [Float16](repeating: 0, count: elementCount)
      lhs.withUnsafeBufferPointer { lhsPtr in
        rhs.withUnsafeBufferPointer { rhsPtr in
          result.withUnsafeMutableBufferPointer { resultPtr in
            nsc_add(lhsPtr.baseAddress!, rhsPtr.baseAddress!, resultPtr.baseAddress!)
          }
        }
      }
      return result
    }
    
    guard let pipeline = computePipeline(for: "nsc_add_kernel"),
          let lhsBuffer = createBuffer(from: lhs, type: Float16.self),
          let rhsBuffer = createBuffer(from: rhs, type: Float16.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float16.self) else {
      // Fallback to CPU
      var result = [Float16](repeating: 0, count: elementCount)
      lhs.withUnsafeBufferPointer { lhsPtr in
        rhs.withUnsafeBufferPointer { rhsPtr in
          result.withUnsafeMutableBufferPointer { resultPtr in
            nsc_add(lhsPtr.baseAddress!, rhsPtr.baseAddress!, resultPtr.baseAddress!)
          }
        }
      }
      return result
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
    encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  public func multiply(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
    let elementCount = lhs.count
    guard elementCount == rhs.count else {
      fatalError("Array sizes must match")
    }
    
    if !shouldUseMetal(for: elementCount) {
      return zip(lhs, rhs).map(*)
    }
    
    guard let pipeline = computePipeline(for: "nsc_mult_float_kernel"),
          let lhsBuffer = createBuffer(from: lhs, type: Float.self),
          let rhsBuffer = createBuffer(from: rhs, type: Float.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float.self) else {
      return zip(lhs, rhs).map(*)
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
    encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  public func divide(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
    let elementCount = lhs.count
    guard elementCount == rhs.count else {
      fatalError("Array sizes must match")
    }
    
    if !shouldUseMetal(for: elementCount) {
      return zip(lhs, rhs).map(/)
    }
    
    guard let pipeline = computePipeline(for: "nsc_div_float_kernel"),
          let lhsBuffer = createBuffer(from: lhs, type: Float.self),
          let rhsBuffer = createBuffer(from: rhs, type: Float.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float.self) else {
      return zip(lhs, rhs).map(/)
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
    encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  public func multiply(_ lhs: [Float16], _ rhs: [Float16]) -> [Float16] {
    let elementCount = lhs.count
    guard elementCount == rhs.count else {
      fatalError("Array sizes must match")
    }
    
    if !shouldUseMetal(for: elementCount) {
      var result = [Float16](repeating: 0, count: elementCount)
      lhs.withUnsafeBufferPointer { lhsPtr in
        rhs.withUnsafeBufferPointer { rhsPtr in
          result.withUnsafeMutableBufferPointer { resultPtr in
            nsc_mult(lhsPtr.baseAddress!, rhsPtr.baseAddress!, resultPtr.baseAddress!)
          }
        }
      }
      return result
    }
    
    guard let pipeline = computePipeline(for: "nsc_mult_kernel"),
          let lhsBuffer = createBuffer(from: lhs, type: Float16.self),
          let rhsBuffer = createBuffer(from: rhs, type: Float16.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float16.self) else {
      // Fallback to CPU
      var result = [Float16](repeating: 0, count: elementCount)
      lhs.withUnsafeBufferPointer { lhsPtr in
        rhs.withUnsafeBufferPointer { rhsPtr in
          result.withUnsafeMutableBufferPointer { resultPtr in
            nsc_mult(lhsPtr.baseAddress!, rhsPtr.baseAddress!, resultPtr.baseAddress!)
          }
        }
      }
      return result
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
    encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  // MARK: - Matrix Operations
  
  public func matmul(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
    let aRows = a.count
    let aCols = a[0].count
    let bRows = b.count
    let bCols = b[0].count
    
    guard aCols == bRows else {
      fatalError("Matrix dimensions don't match for multiplication")
    }
    
    let elementCount = aRows * bCols
    
    if !shouldUseMetal(for: elementCount) {
      var result = [[Float]]()
      for i in 0..<aRows {
        var row = [Float]()
        for j in 0..<bCols {
          var sum: Float = 0
          for k in 0..<aCols {
            sum += a[i][k] * b[k][j]
          }
          row.append(sum)
        }
        result.append(row)
      }
      return result
    }
    
    guard let pipeline = computePipeline(for: "nsc_matmul_float_kernel") else {
      fatalError("Failed to create matmul pipeline")
    }
    
    let aFlat = a.flatMap { $0 }
    let bFlat = b.flatMap { $0 }
    let aSize = NSC_Size(rows: Int32(aRows), columns: Int32(aCols))
    let bSize = NSC_Size(rows: Int32(bRows), columns: Int32(bCols))
    
    guard let aBuffer = createBuffer(from: aFlat, type: Float.self),
          let bBuffer = createBuffer(from: bFlat, type: Float.self),
          let resultBuffer = createBuffer(count: aRows * bCols, type: Float.self),
          let aSizeBuffer = createBuffer(from: [aSize], type: NSC_Size.self),
          let bSizeBuffer = createBuffer(from: [bSize], type: NSC_Size.self) else {
      fatalError("Failed to create buffers")
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(aSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(bSizeBuffer, offset: 0, index: 4)
    
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadgroups = MTLSize(
      width: (bCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (aRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: aRows * bCols)
    let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: aRows * bCols))
    
    var result = [[Float]]()
    for i in 0..<aRows {
      let startIndex = i * bCols
      let endIndex = startIndex + bCols
      result.append(Array(resultFlat[startIndex..<endIndex]))
    }
    
    return result
  }
  
  public func matmul(_ a: [[Float16]], _ b: [[Float16]]) -> [[Float16]] {
    let aShape = a.shape
    let bShape = b.shape
    
    let aRows = aShape[safe: 1] ?? 0
    let aCols = aShape[safe: 0] ?? 0
    let bRows = bShape[safe: 1] ?? 0
    let bCols = bShape[safe: 0] ?? 0
    
    guard aCols == bRows else {
      fatalError("Matrix dimensions don't match for multiplication")
    }
    
    let elementCount = aRows * bCols
    
    if !shouldUseMetal(for: elementCount) {
      // Fallback to CPU implementation
      return NumSwiftC.matmul(a, b: b, aSize: (aRows, aCols), bSize: (bRows, bCols))
    }
    
    // Metal implementation
    guard let pipeline = computePipeline(for: "nsc_matmul_kernel") else {
      fatalError("Failed to create matmul pipeline")
    }
    
    let aFlat = a.flatMap { $0 }
    let bFlat = b.flatMap { $0 }
    let aSize = NSC_Size(rows: Int32(aRows), columns: Int32(aCols))
    let bSize = NSC_Size(rows: Int32(bRows), columns: Int32(bCols))
    
    guard let aBuffer = createBuffer(from: aFlat, type: Float16.self),
          let bBuffer = createBuffer(from: bFlat, type: Float16.self),
          let resultBuffer = createBuffer(count: aRows * bCols, type: Float16.self),
          let aSizeBuffer = createBuffer(from: [aSize], type: NSC_Size.self),
          let bSizeBuffer = createBuffer(from: [bSize], type: NSC_Size.self) else {
      fatalError("Failed to create buffers")
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(aSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(bSizeBuffer, offset: 0, index: 4)
    
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadgroups = MTLSize(
      width: (bCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (aRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: aRows * bCols)
    let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: aRows * bCols))
    
    // Convert back to 2D array
    var result = [[Float16]]()
    for i in 0..<aRows {
      let startIndex = i * bCols
      let endIndex = startIndex + bCols
      result.append(Array(resultFlat[startIndex..<endIndex]))
    }
    
    return result
  }
  
  // MARK: - Transposed Convolution Operations
  
  public func transconv2d(_ signal: [[Float]], _ filter: [[Float]], stride: (rows: Int, cols: Int) = (1, 1), padding: NumSwift.ConvPadding = .valid) -> [[Float]] {
    let signalSize = signal.shape
    let fSize = filter.shape
    
    let inputRows = signalSize[safe: 1] ?? 0
    let inputCols = signalSize[safe: 0] ?? 0
    let filterRows = fSize[safe: 1] ?? 0
    let filterCols = fSize[safe: 0] ?? 0
    
    // Calculate output dimensions for transposed convolution
    let outputRows = (inputRows - 1) * stride.rows + filterRows
    let outputCols = (inputCols - 1) * stride.cols + filterCols
    
    let elementCount = outputRows * outputCols
    
    if !shouldUseMetal(for: elementCount) {
      return NumSwiftC.transConv2d(signal: signal,
                                   filter: filter,
                                   strides: stride,
                                   padding: padding,
                                   filterSize: (filterRows, filterCols),
                                   inputSize: (inputRows, inputCols))
    }
    
    guard let pipeline = computePipeline(for: "nsc_transconv2d_float_kernel") else {
      // Fallback to CPU implementation
      return NumSwiftC.transConv2d(signal: signal,
                                   filter: filter,
                                   strides: stride,
                                   padding: padding,
                                   filterSize: (filterRows, filterCols),
                                   inputSize: (inputRows, inputCols))
    }
    
    let signalFlat = signal.flatMap { $0 }
    let filterFlat = filter.flatMap { $0 }
    let inputSize = NSC_Size(rows: Int32(inputRows), columns: Int32(inputCols))
    let filterSize = NSC_Size(rows: Int32(filterRows), columns: Int32(filterCols))
    let strideSize = NSC_Size(rows: Int32(stride.rows), columns: Int32(stride.cols))
    let resultSize = NSC_Size(rows: Int32(outputRows), columns: Int32(outputCols))
    
    guard let signalBuffer = createBuffer(from: signalFlat, type: Float.self),
          let filterBuffer = createBuffer(from: filterFlat, type: Float.self),
          let resultBuffer = createBuffer(count: outputRows * outputCols, type: Float.self),
          let inputSizeBuffer = createBuffer(from: [inputSize], type: NSC_Size.self),
          let filterSizeBuffer = createBuffer(from: [filterSize], type: NSC_Size.self),
          let strideSizeBuffer = createBuffer(from: [strideSize], type: NSC_Size.self),
          let resultSizeBuffer = createBuffer(from: [resultSize], type: NSC_Size.self) else {
      // Fallback to CPU implementation
      return NumSwiftC.transConv2d(signal: signal,
                                   filter: filter,
                                   strides: stride,
                                   padding: padding,
                                   filterSize: (filterRows, filterCols),
                                   inputSize: (inputRows, inputCols))
    }
    
    // Initialize result buffer to zero
    var resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: outputRows * outputCols)
    for i in 0..<(outputRows * outputCols) {
      resultPointer[i] = 0.0
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(signalBuffer, offset: 0, index: 0)
    encoder.setBuffer(filterBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(inputSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(filterSizeBuffer, offset: 0, index: 4)
    encoder.setBuffer(strideSizeBuffer, offset: 0, index: 5)
    encoder.setBuffer(resultSizeBuffer, offset: 0, index: 6)
    
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadgroups = MTLSize(
      width: (inputCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (inputRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: outputRows * outputCols))
    
    // Apply padding if needed
    var finalResult: [[Float]] = []
    for i in 0..<outputRows {
      let startIndex = i * outputCols
      let endIndex = startIndex + outputCols
      finalResult.append(Array(resultFlat[startIndex..<endIndex]))
    }
    
    // Handle padding
    if padding == .same {
      let padLeft = Int(floor(Double(filterRows - stride.rows) / 2.0))
      let padRight = filterRows - stride.rows - padLeft
      let padTop = Int(floor(Double(filterCols - stride.cols) / 2.0))
      let padBottom = filterCols - stride.cols - padTop
      
      let startRow = padTop
      let endRow = outputRows - padBottom
      let startCol = padLeft
      let endCol = outputCols - padRight
      
      finalResult = Array(finalResult[startRow..<endRow].map { Array($0[startCol..<endCol]) })
    }
    
    return finalResult
  }
  
  public func transconv2d(_ signal: [[Float16]], _ filter: [[Float16]], stride: (rows: Int, cols: Int) = (1, 1), padding: NumSwift.ConvPadding = .valid) -> [[Float16]] {
    let signalSize = signal.shape
    let fSize = filter.shape
    
    let inputRows = signalSize[safe: 1] ?? 0
    let inputCols = signalSize[safe: 0] ?? 0
    let filterRows = fSize[safe: 1] ?? 0
    let filterCols = fSize[safe: 0] ?? 0
    
    // Calculate output dimensions for transposed convolution
    let outputRows = (inputRows - 1) * stride.rows + filterRows
    let outputCols = (inputCols - 1) * stride.cols + filterCols
    
    let elementCount = outputRows * outputCols
    
    if !shouldUseMetal(for: elementCount) {
      return NumSwiftC.transConv2d(signal: signal,
                                   filter: filter,
                                   strides: stride,
                                   padding: padding,
                                   filterSize: (filterRows, filterCols),
                                   inputSize: (inputRows, inputCols))
    }
    
    guard let pipeline = computePipeline(for: "nsc_transconv2d_kernel") else {
      // Fallback to CPU implementation
      return NumSwiftC.transConv2d(signal: signal,
                                   filter: filter,
                                   strides: stride,
                                   padding: padding,
                                   filterSize: (filterRows, filterCols),
                                   inputSize: (inputRows, inputCols))
    }
    
    let signalFlat = signal.flatMap { $0 }
    let filterFlat = filter.flatMap { $0 }
    let inputSize = NSC_Size(rows: Int32(inputRows), columns: Int32(inputCols))
    let filterSize = NSC_Size(rows: Int32(filterRows), columns: Int32(filterCols))
    let strideSize = NSC_Size(rows: Int32(stride.rows), columns: Int32(stride.cols))
    let resultSize = NSC_Size(rows: Int32(outputRows), columns: Int32(outputCols))
    
    guard let signalBuffer = createBuffer(from: signalFlat, type: Float16.self),
          let filterBuffer = createBuffer(from: filterFlat, type: Float16.self),
          let resultBuffer = createBuffer(count: outputRows * outputCols, type: Float16.self),
          let inputSizeBuffer = createBuffer(from: [inputSize], type: NSC_Size.self),
          let filterSizeBuffer = createBuffer(from: [filterSize], type: NSC_Size.self),
          let strideSizeBuffer = createBuffer(from: [strideSize], type: NSC_Size.self),
          let resultSizeBuffer = createBuffer(from: [resultSize], type: NSC_Size.self) else {
      // Fallback to CPU implementation
      return NumSwiftC.transConv2d(signal: signal,
                                   filter: filter,
                                   strides: stride,
                                   padding: padding,
                                   filterSize: (filterRows, filterCols),
                                   inputSize: (inputRows, inputCols))
    }
    
    // Initialize result buffer to zero
    var resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: outputRows * outputCols)
    for i in 0..<(outputRows * outputCols) {
      resultPointer[i] = 0.0
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(signalBuffer, offset: 0, index: 0)
    encoder.setBuffer(filterBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(inputSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(filterSizeBuffer, offset: 0, index: 4)
    encoder.setBuffer(strideSizeBuffer, offset: 0, index: 5)
    encoder.setBuffer(resultSizeBuffer, offset: 0, index: 6)
    
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadgroups = MTLSize(
      width: (inputCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (inputRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: outputRows * outputCols))
    
    // Apply padding if needed
    var finalResult: [[Float16]] = []
    for i in 0..<outputRows {
      let startIndex = i * outputCols
      let endIndex = startIndex + outputCols
      finalResult.append(Array(resultFlat[startIndex..<endIndex]))
    }
    
    // Handle padding
    if padding == .same {
      let padLeft = Int(floor(Double(filterRows - stride.rows) / 2.0))
      let padRight = filterRows - stride.rows - padLeft
      let padTop = Int(floor(Double(filterCols - stride.cols) / 2.0))
      let padBottom = filterCols - stride.cols - padTop
      
      let startRow = padTop
      let endRow = outputRows - padBottom
      let startCol = padLeft
      let endCol = outputCols - padRight
      
      finalResult = Array(finalResult[startRow..<endRow].map { Array($0[startCol..<endCol]) })
    }
    
    return finalResult
  }
  
  // MARK: - Convolution Helpers
  
  private func calculateConvolutionDimensions(
    inputRows: Int, inputCols: Int,
    filterRows: Int, filterCols: Int,
    stride: (rows: Int, cols: Int),
    padding: NumSwift.ConvPadding
  ) -> (outputRows: Int, outputCols: Int, padTop: Int, padLeft: Int) {
    switch padding {
    case .valid:
      let outputRows = (inputRows - filterRows) / stride.rows + 1
      let outputCols = (inputCols - filterCols) / stride.cols + 1
      return (outputRows, outputCols, 0, 0)
      
    case .same:
      let outputRows = (inputRows + stride.rows - 1) / stride.rows
      let outputCols = (inputCols + stride.cols - 1) / stride.cols
      
      let padAlongHeight = Swift.max(0, (outputRows - 1) * stride.rows + filterRows - inputRows)
      let padAlongWidth = Swift.max(0, (outputCols - 1) * stride.cols + filterCols - inputCols)
      
      let padTop = padAlongHeight / 2
      let padLeft = padAlongWidth / 2
      
      return (outputRows, outputCols, padTop, padLeft)
    }
  }
  
  // MARK: - Convolution Operations
  
  public func conv2d(_ signal: [[Float]], _ filter: [[Float]], stride: (rows: Int, cols: Int) = (1, 1), padding: NumSwift.ConvPadding = .valid) -> [[Float]] {
    let signalSize = signal.shape
    let fSize = filter.shape
    
    let inputRows = signalSize[safe: 1] ?? 0
    let inputCols = signalSize[safe: 0] ?? 0
    let filterRows = fSize[safe: 1] ?? 0
    let filterCols = fSize[safe: 0] ?? 0
    
    let (outputRows, outputCols, padTop, padLeft) = calculateConvolutionDimensions(
      inputRows: inputRows, inputCols: inputCols,
      filterRows: filterRows, filterCols: filterCols,
      stride: stride, padding: padding
    )
    
    let elementCount = outputRows * outputCols
    
    if !shouldUseMetal(for: elementCount) {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    guard let pipeline = computePipeline(for: "nsc_conv2d_float_kernel") else {
      fatalError("Failed to create conv2d pipeline")
    }
    
    let signalFlat = signal.flatMap { $0 }
    let filterFlat = filter.flatMap { $0 }
    let inputSize = NSC_Size(rows: Int32(inputRows), columns: Int32(inputCols))
    let filterSize = NSC_Size(rows: Int32(filterRows), columns: Int32(filterCols))
    let strideSize = NSC_Size(rows: Int32(stride.rows), columns: Int32(stride.cols))
    let resultSize = NSC_Size(rows: Int32(outputRows), columns: Int32(outputCols))
    let paddingValue = Int32(padding == .same ? 1 : 0)
    let padTopValue = Int32(padTop)
    let padLeftValue = Int32(padLeft)
    
    guard let signalBuffer = createBuffer(from: signalFlat, type: Float.self),
          let filterBuffer = createBuffer(from: filterFlat, type: Float.self),
          let resultBuffer = createBuffer(count: outputRows * outputCols, type: Float.self),
          let inputSizeBuffer = createBuffer(from: [inputSize], type: NSC_Size.self),
          let filterSizeBuffer = createBuffer(from: [filterSize], type: NSC_Size.self),
          let strideSizeBuffer = createBuffer(from: [strideSize], type: NSC_Size.self),
          let resultSizeBuffer = createBuffer(from: [resultSize], type: NSC_Size.self),
          let paddingBuffer = createBuffer(from: [paddingValue], type: Int32.self),
          let padTopBuffer = createBuffer(from: [padTopValue], type: Int32.self),
          let padLeftBuffer = createBuffer(from: [padLeftValue], type: Int32.self) else {
      fatalError("Failed to create buffers")
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(signalBuffer, offset: 0, index: 0)
    encoder.setBuffer(filterBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(inputSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(filterSizeBuffer, offset: 0, index: 4)
    encoder.setBuffer(strideSizeBuffer, offset: 0, index: 5)
    encoder.setBuffer(resultSizeBuffer, offset: 0, index: 6)
    encoder.setBuffer(paddingBuffer, offset: 0, index: 7)
    encoder.setBuffer(padTopBuffer, offset: 0, index: 8)
    encoder.setBuffer(padLeftBuffer, offset: 0, index: 9)
    
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadgroups = MTLSize(
      width: (outputCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (outputRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: outputRows * outputCols)
    let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: outputRows * outputCols))
    
    var result = [[Float]]()
    for i in 0..<outputRows {
      let startIndex = i * outputCols
      let endIndex = startIndex + outputCols
      result.append(Array(resultFlat[startIndex..<endIndex]))
    }
    
    return result
  }
  
  public func conv2d(_ signal: [[Float16]], _ filter: [[Float16]], stride: (rows: Int, cols: Int) = (1, 1), padding: NumSwift.ConvPadding = .valid) -> [[Float16]] {
    let signalSize = signal.shape
    let fSize = filter.shape
    
    let inputRows = signalSize[safe: 1] ?? 0
    let inputCols = signalSize[safe: 0] ?? 0
    let filterRows = fSize[safe: 1] ?? 0
    let filterCols = fSize[safe: 0] ?? 0
    
    let (outputRows, outputCols, padTop, padLeft) = calculateConvolutionDimensions(
      inputRows: inputRows, inputCols: inputCols,
      filterRows: filterRows, filterCols: filterCols,
      stride: stride, padding: padding
    )
    
    let elementCount = outputRows * outputCols
    
    if !shouldUseMetal(for: elementCount) {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    // Metal implementation
    guard let pipeline = computePipeline(for: "nsc_conv2d_kernel") else {
      fatalError("Failed to create conv2d pipeline")
    }
    
    let signalFlat = signal.flatMap { $0 }
    let filterFlat = filter.flatMap { $0 }
    let inputSize = NSC_Size(rows: Int32(inputRows), columns: Int32(inputCols))
    let filterSize = NSC_Size(rows: Int32(filterRows), columns: Int32(filterCols))
    let strideSize = NSC_Size(rows: Int32(stride.rows), columns: Int32(stride.cols))
    let resultSize = NSC_Size(rows: Int32(outputRows), columns: Int32(outputCols))
    let paddingValue = Int32(padding == .same ? 1 : 0)
    let padTopValue = Int32(padTop)
    let padLeftValue = Int32(padLeft)
    
    guard let signalBuffer = createBuffer(from: signalFlat, type: Float16.self),
          let filterBuffer = createBuffer(from: filterFlat, type: Float16.self),
          let resultBuffer = createBuffer(count: outputRows * outputCols, type: Float16.self),
          let inputSizeBuffer = createBuffer(from: [inputSize], type: NSC_Size.self),
          let filterSizeBuffer = createBuffer(from: [filterSize], type: NSC_Size.self),
          let strideSizeBuffer = createBuffer(from: [strideSize], type: NSC_Size.self),
          let resultSizeBuffer = createBuffer(from: [resultSize], type: NSC_Size.self),
          let paddingBuffer = createBuffer(from: [paddingValue], type: Int32.self),
          let padTopBuffer = createBuffer(from: [padTopValue], type: Int32.self),
          let padLeftBuffer = createBuffer(from: [padLeftValue], type: Int32.self) else {
      fatalError("Failed to create buffers")
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(signalBuffer, offset: 0, index: 0)
    encoder.setBuffer(filterBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(inputSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(filterSizeBuffer, offset: 0, index: 4)
    encoder.setBuffer(strideSizeBuffer, offset: 0, index: 5)
    encoder.setBuffer(resultSizeBuffer, offset: 0, index: 6)
    encoder.setBuffer(paddingBuffer, offset: 0, index: 7)
    encoder.setBuffer(padTopBuffer, offset: 0, index: 8)
    encoder.setBuffer(padLeftBuffer, offset: 0, index: 9)
    
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadgroups = MTLSize(
      width: (outputCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (outputRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: outputRows * outputCols)
    let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: outputRows * outputCols))
    
    // Convert back to 2D array
    var result = [[Float16]]()
    for i in 0..<outputRows {
      let startIndex = i * outputCols
      let endIndex = startIndex + outputCols
      result.append(Array(resultFlat[startIndex..<endIndex]))
    }
    
    return result
  }
  
  // MARK: - Activation Functions
  
  public func activation(_ data: [Float], type: ActivationType, limit: Float = 0.01) -> [Float] {
    let elementCount = data.count
    
    if !shouldUseMetal(for: elementCount) {
      // CPU fallback
      return data.map { value in
        switch type {
        case .relu:
          return Swift.max(0, value)
        case .leakyRelu:
          return value < 0 ? limit * value : value
        case .sigmoid:
          return 1.0 / (1.0 + exp(-value))
        case .swish:
          let sigmoid = 1.0 / (1.0 + exp(-value))
          return value * sigmoid
        case .tanh:
          let denom = 1.0 + exp(-2 * value)
          return (2.0 / denom) - 1.0
        case .none:
          return value
        case .selu:
          let alpha: Float = 1.6732632423543772848170429916717
          let scale: Float = 1.0507009873554804934193349852946
          return value <= 0 ? scale * alpha * (exp(value) - 1.0) : scale * value
        case .gelu:
          let sqrt2Pi: Float = 0.7978845608028654 // sqrt(2/pi)
          let a: Float = 0.044715
          let tanhInput = sqrt2Pi * (value + a * pow(value, 3))
          return 0.5 * value * (1.0 + tanh(tanhInput))
        case .softmax:
          return value
        }
      }
    }
    
    guard let pipeline = computePipeline(for: "nsc_activation_kernel"),
          let dataBuffer = createBuffer(from: data, type: Float.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float.self),
          let typeBuffer = createBuffer(from: [type.rawValue], type: UInt32.self),
          let limitBuffer = createBuffer(from: [limit], type: Float.self) else {
      // Fallback to CPU implementation
      return data.map { value in
        switch type {
        case .relu:
          return Swift.max(0, value)
        case .leakyRelu:
          return value < 0 ? limit * value : value
        case .sigmoid:
          return 1.0 / (1.0 + exp(-value))
        case .swish:
          let sigmoid = 1.0 / (1.0 + exp(-value))
          return value * sigmoid
        case .tanh:
          let denom = 1.0 + exp(-2 * value)
          return (2.0 / denom) - 1.0
        case .none:
          return value
        case .selu:
          let alpha: Float = 1.6732632423543772848170429916717
          let scale: Float = 1.0507009873554804934193349852946
          return value <= 0 ? scale * alpha * (exp(value) - 1.0) : scale * value
        case .gelu:
          let sqrt2Pi: Float = 0.7978845608028654 // sqrt(2/pi)
          let a: Float = 0.044715
          let tanhInput = sqrt2Pi * (value + a * pow(value, 3))
          return 0.5 * value * (1.0 + tanh(tanhInput))
        case .softmax:
          return value
        }
      }
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(typeBuffer, offset: 0, index: 2)
    encoder.setBuffer(limitBuffer, offset: 0, index: 3)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  public func derivative(_ data: [Float], type: ActivationType, limit: Float = 0.01) -> [Float] {
    let elementCount = data.count
    
    if !shouldUseMetal(for: elementCount) {
      // CPU fallback
      return data.map { value in
        switch type {
        case .relu:
          return value >= 0 ? 1 : 0
        case .leakyRelu:
          return value > 0 ? 1 : limit
        case .sigmoid:
          let sig = 1.0 / (1.0 + exp(-value))
          return sig * (1 - sig)
        case .swish:
          return (exp(-value) * (value + 1) + 1) / pow((1 + exp(-value)), 2)
        case .tanh:
          let denom = 1.0 + exp(-2 * value)
          let tanActivate = (2.0 / denom) - 1.0
          return 1 - (pow(tanActivate, 2))
        case .none:
          return 1
        case .selu:
          let alpha: Float = 1.6732632423543772848170429916717
          let scale: Float = 1.0507009873554804934193349852946
          return value <= 0 ? scale * alpha * exp(value) : scale
        case .gelu:
          let sqrt2Pi: Float = 0.7978845608028654 // sqrt(2/pi)
          let a: Float = 0.044715
          let xCubed = pow(value, 3)
          let tanhInput = sqrt2Pi * (value + a * xCubed)
          let tanhVal = tanh(tanhInput)
          let sechVal = 1.0 - tanhVal * tanhVal // sech^2(x) = 1 - tanh^2(x)
          return 0.5 * (1.0 + tanhVal) + 0.5 * value * sechVal * sqrt2Pi * (1.0 + 3.0 * a * value * value)
        case .softmax:
          return 1
        }
      }
    }
    
    guard let pipeline = computePipeline(for: "nsc_derivate_kernel"),
          let dataBuffer = createBuffer(from: data, type: Float.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float.self),
          let typeBuffer = createBuffer(from: [type.rawValue], type: UInt32.self),
          let limitBuffer = createBuffer(from: [limit], type: Float.self) else {
      // Fallback to CPU implementation
      return data.map { value in
        switch type {
        case .relu:
          return value >= 0 ? 1 : 0
        case .leakyRelu:
          return value > 0 ? 1 : limit
        case .sigmoid:
          let sig = 1.0 / (1.0 + exp(-value))
          return sig * (1 - sig)
        case .swish:
          return (exp(-value) * (value + 1) + 1) / pow((1 + exp(-value)), 2)
        case .tanh:
          let denom = 1.0 + exp(-2 * value)
          let tanActivate = (2.0 / denom) - 1.0
          return 1 - (pow(tanActivate, 2))
        case .none:
          return 1
        case .selu:
          let alpha: Float = 1.6732632423543772848170429916717
          let scale: Float = 1.0507009873554804934193349852946
          return value <= 0 ? scale * alpha * exp(value) : scale
        case .gelu:
          let sqrt2Pi: Float = 0.7978845608028654 // sqrt(2/pi)
          let a: Float = 0.044715
          let xCubed = pow(value, 3)
          let tanhInput = sqrt2Pi * (value + a * xCubed)
          let tanhVal = tanh(tanhInput)
          let sechVal = 1.0 - tanhVal * tanhVal // sech^2(x) = 1 - tanh^2(x)
          return 0.5 * (1.0 + tanhVal) + 0.5 * value * sechVal * sqrt2Pi * (1.0 + 3.0 * a * value * value)
        case .softmax:
          return 1
        }
      }
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(typeBuffer, offset: 0, index: 2)
    encoder.setBuffer(limitBuffer, offset: 0, index: 3)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  public func activation(_ data: [Float16], type: ActivationType, limit: Float16 = 0.01) -> [Float16] {
    let elementCount = data.count
    
    if !shouldUseMetal(for: elementCount) {
      // CPU fallback
      return data.map { value in
        switch type {
        case .relu:
          return Swift.max(0, value)
        case .leakyRelu:
          return value < 0 ? limit * value : value
        case .sigmoid:
          return 1.0 / (1.0 + Float16.exp(-value))
        case .swish:
          let sigmoid = 1.0 / (1.0 + Float16.exp(-value))
          return value * sigmoid
        case .tanh:
          let denom = 1.0 + Float16.exp(-2 * value)
          return (2.0 / denom) - 1.0
        case .none:
          return value
        case .selu:
          let alpha: Float16 = 1.6732632423543772848170429916717
          let scale: Float16 = 1.0507009873554804934193349852946
          return value <= 0 ? scale * alpha * (Float16.exp(value) - 1.0) : scale * value
        case .gelu:
          let sqrt2Pi: Float16 = 0.7978845608028654 // sqrt(2/pi)
          let a: Float16 = 0.044715
          let tanhInput = sqrt2Pi * (value + a * Float16.pow(value, 3))
          return 0.5 * value * (1.0 + Float16.tanh(tanhInput))
        case .softmax:
          return value
        }
      }
    }
    
    guard let pipeline = computePipeline(for: "nsc_activation_half_kernel"),
          let dataBuffer = createBuffer(from: data, type: Float16.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float16.self),
          let typeBuffer = createBuffer(from: [type.rawValue], type: UInt32.self),
          let limitBuffer = createBuffer(from: [limit], type: Float16.self) else {
      // Fallback to CPU implementation
      return data.map { value in
        switch type {
        case .relu:
          return Swift.max(0, value)
        case .leakyRelu:
          return value < 0 ? limit * value : value
        case .sigmoid:
          return 1.0 / (1.0 + Float16.exp(-value))
        case .swish:
          let sigmoid = 1.0 / (1.0 + Float16.exp(-value))
          return value * sigmoid
        case .tanh:
          let denom = 1.0 + Float16.exp(-2 * value)
          return (2.0 / denom) - 1.0
        case .none:
          return value
        case .selu:
          let alpha: Float16 = 1.6732632423543772848170429916717
          let scale: Float16 = 1.0507009873554804934193349852946
          return value <= 0 ? scale * alpha * (Float16.exp(value) - 1.0) : scale * value
        case .gelu:
          let sqrt2Pi: Float16 = 0.7978845608028654 // sqrt(2/pi)
          let a: Float16 = 0.044715
          let tanhInput = sqrt2Pi * (value + a * Float16.pow(value, 3))
          return 0.5 * value * (1.0 + Float16.tanh(tanhInput))
        case .softmax:
          return value
        }
      }
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(typeBuffer, offset: 0, index: 2)
    encoder.setBuffer(limitBuffer, offset: 0, index: 3)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  public func derivative(_ data: [Float16], type: ActivationType, limit: Float16 = 0.01) -> [Float16] {
    let elementCount = data.count
    
    if !shouldUseMetal(for: elementCount) {
      // CPU fallback
      return data.map { value in
        switch type {
        case .relu:
          return value >= 0 ? 1 : 0
        case .leakyRelu:
          return value > 0 ? 1 : limit
        case .sigmoid:
          let sig = 1.0 / (1.0 + Float16.exp(-value))
          return sig * (1 - sig)
        case .swish:
          return (Float16.exp(-value) * (value + 1) + 1) / Float16.pow((1 + Float16.exp(-value)), 2)
        case .tanh:
          let denom = 1.0 + Float16.exp(-2 * value)
          let tanActivate = (2.0 / denom) - 1.0
          return 1 - (Float16.pow(tanActivate, 2))
        case .none:
          return 1
        case .selu:
          let alpha: Float16 = 1.6732632423543772848170429916717
          let scale: Float16 = 1.0507009873554804934193349852946
          return value <= 0 ? scale * alpha * Float16.exp(value) : scale
        case .gelu:
          let sqrt2Pi: Float16 = 0.7978845608028654 // sqrt(2/pi)
          let a: Float16 = 0.044715
          let xCubed = Float16.pow(value, 3)
          let tanhInput = sqrt2Pi * (value + a * xCubed)
          let tanhVal = Float16.tanh(tanhInput)
          let sechVal = 1.0 - tanhVal * tanhVal // sech^2(x) = 1 - tanh^2(x)
          return 0.5 * (1.0 + tanhVal) + 0.5 * value * sechVal * sqrt2Pi * (1.0 + 3.0 * a * value * value)
        case .softmax:
          return 1
        }
      }
    }
    
    guard let pipeline = computePipeline(for: "nsc_derivate_half_kernel"),
          let dataBuffer = createBuffer(from: data, type: Float16.self),
          let resultBuffer = createBuffer(count: elementCount, type: Float16.self),
          let typeBuffer = createBuffer(from: [type.rawValue], type: UInt32.self),
          let limitBuffer = createBuffer(from: [limit], type: Float16.self) else {
      // Fallback to CPU implementation
      return data.map { value in
        switch type {
        case .relu:
          return value >= 0 ? 1 : 0
        case .leakyRelu:
          return value > 0 ? 1 : limit
        case .sigmoid:
          let sig = 1.0 / (1.0 + Float16.exp(-value))
          return sig * (1 - sig)
        case .swish:
          return (Float16.exp(-value) * (value + 1) + 1) / Float16.pow((1 + Float16.exp(-value)), 2)
        case .tanh:
          let denom = 1.0 + Float16.exp(-2 * value)
          let tanActivate = (2.0 / denom) - 1.0
          return 1 - (Float16.pow(tanActivate, 2))
        case .none:
          return 1
        case .selu:
          let alpha: Float16 = 1.6732632423543772848170429916717
          let scale: Float16 = 1.0507009873554804934193349852946
          return value <= 0 ? scale * alpha * Float16.exp(value) : scale
        case .gelu:
          let sqrt2Pi: Float16 = 0.7978845608028654 // sqrt(2/pi)
          let a: Float16 = 0.044715
          let xCubed = Float16.pow(value, 3)
          let tanhInput = sqrt2Pi * (value + a * xCubed)
          let tanhVal = Float16.tanh(tanhInput)
          let sechVal = 1.0 - tanhVal * tanhVal // sech^2(x) = 1 - tanh^2(x)
          return 0.5 * (1.0 + tanhVal) + 0.5 * value * sechVal * sqrt2Pi * (1.0 + 3.0 * a * value * value)
        case .softmax:
          return 1
        }
      }
    }
    
    guard let commandBuffer = config.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(typeBuffer, offset: 0, index: 2)
    encoder.setBuffer(limitBuffer, offset: 0, index: 3)
    
    let threadsPerThreadgroup = MTLSize(width: Swift.min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
    let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
}
