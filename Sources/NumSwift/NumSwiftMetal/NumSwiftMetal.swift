import Foundation
import Metal
import MetalKit
import NumSwiftC
import Numerics

// MARK: - Metal Backend Configuration

public class ThreadLocal<T> {
  private var key: pthread_key_t
  
  public init() {
    key = pthread_key_t()
    pthread_key_create(&key, nil)
  }
  
  var value: T? {
    get {
      return pthread_getspecific(key)?.assumingMemoryBound(to: T.self).pointee
    }
    set {
      if let newValue = newValue {
        let pointer = UnsafeMutablePointer<T>.allocate(capacity: 1)
        pointer.initialize(to: newValue)
        pthread_setspecific(key, pointer)
      } else {
        pthread_setspecific(key, nil)
      }
    }
  }
}

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
  public let library: MTLLibrary
  
  public init?(device: MTLDevice? = nil) {
    guard let device = device ?? MTLCreateSystemDefaultDevice() else {
      return nil
    }
    
    self.device = device
    
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
  private let threadLocal = ThreadLocal<MTLCommandQueue>()
  
  private func getCommandQueue() -> MTLCommandQueue {
    if let queue = threadLocal.value {
      return queue
    }
    
    let newQueue = config.device.makeCommandQueue()!
    threadLocal.value = newQueue
    return newQueue
  }
  
  // Cache for frequently used compute pipelines
  private var pipelineCache: [String: MTLComputePipelineState] = [:]
  
  public var isAsyncMode = false
  
  // GPU utilization monitoring (thread-safe)
  private var metalOperationCount: Int = 0
  private var cpuFallbackCount: Int = 0
  
  private let metalLock = NSLock()
  
  private var commandBuffersToExectute: [MTLCommandBuffer] = []
  
  public init?(configuration: MetalConfiguration? = nil) {
    if let config = configuration {
      self.config = config
    } else {
      guard let config = MetalConfiguration() else { return nil }
      self.config = config
    }
  }
  
  deinit {
    cleanup()
  }
  
  // MARK: - Lifecycle Management
  
  public func cleanup() {
    bufferPool.clear()
    pipelineCache.removeAll()
    commandBuffersToExectute.removeAll()
  }
  
  public func getBufferPoolStats() -> [String: Int] {
    return bufferPool.stats()
  }
  
  public func clearBufferPool() {
    bufferPool.clear()
  }
  
  public func setBackend(_ backend: ComputeBackend) {
    self.backend = backend
  }
  
  public func getBackend() -> ComputeBackend {
    return backend
  }
  
  // Optimized threadgroup sizes based on problem size
  private func getOptimalThreadgroupSize(width: Int, height: Int = 1) -> MTLSize {
    let device = config.device
    
    // Apple Silicon optimized sizes
    if device.name.contains("Apple") {
      if width * height > 100000 {
        return MTLSize(width: 32, height: 32, depth: 1) // Large problems
      } else if width * height > 10000 {
        return MTLSize(width: 16, height: 16, depth: 1) // Medium problems
      } else {
        return MTLSize(width: 8, height: 8, depth: 1)   // Small problems
      }
    }
    
    // Intel/AMD GPU optimized sizes
    if width * height > 50000 {
      return MTLSize(width: 16, height: 16, depth: 1)
    } else {
      return MTLSize(width: 8, height: 8, depth: 1)
    }
  }
  
  private func getOptimal1DThreadgroupSize(elementCount: Int) -> Int {
    let device = config.device
    
    // Apple Silicon can handle larger threadgroups
    if device.name.contains("Apple") {
      if elementCount > 100000 {
        return 1024
      } else if elementCount > 10000 {
        return 512
      } else {
        return 256
      }
    }
    
    // Conservative sizes for other GPUs
    return elementCount > 10000 ? 256 : 128
  }
  
  private func getOptimalTileSize(aRows: Int, aCols: Int, bCols: Int) -> Int {
    let totalElements = aRows * aCols * bCols
    
    // Larger tiles for bigger problems, but limited by shared memory
    if totalElements > 1000000 {
      return 32
    } else if totalElements > 100000 {
      return 16
    } else {
      return 8
    }
  }

  // MARK: - Pipeline Management
  
  // pipeline cache is a good thing. It speeds things up and reduces memory
  private func computePipeline(for functionName: String) -> MTLComputePipelineState? {
    defer { metalLock.unlock() }
    metalLock.lock()
    
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
      // Lowered threshold for neural network workloads
      return elementCount > 100  // Changed from 1000 to 100
    }
  }
  
  private func shouldUseParallelReduction(for elementCount: Int) -> Bool {
    // Use parallel reduction for arrays larger than 4096 elements
    // Below this threshold, the overhead of parallel reduction isn't worth it
    return elementCount > 4096
  }
  
  private func shouldUseTiledMatmul(aRows: Int, aCols: Int, bCols: Int) -> Bool {
    // Use tiled matrix multiplication for larger matrices
    // The tile size is 16x16, so it's beneficial when at least one dimension is > 32
    let totalOps = aRows * aCols * bCols
    return totalOps > 32768 && (aRows >= 32 || aCols >= 32 || bCols >= 32)
  }
  
  private func shouldUseOptimizedConv(inputHeight: Int, inputWidth: Int, filterSize: Int) -> Bool {
    // Use optimized convolution for larger inputs or small filter sizes
    let inputSize = inputHeight * inputWidth
    return inputSize > 1024 || filterSize <= 5
  }
  
  // MARK: - Buffer Pool Management
  
  private class BufferPool {
    private var pools: [Int: [MTLBuffer]] = [:]
    private let lock = NSLock()
    private let device: MTLDevice
    private let maxPoolSize: Int
    private let bucketSizes: [Int]
    
    init(device: MTLDevice, maxPoolSize: Int = 50) {
      self.device = device
      self.maxPoolSize = maxPoolSize
      // Define bucket sizes for common buffer sizes (in bytes)
      self.bucketSizes = [
        16,      // Small parameters (1-4 Float32s)
        64,      // Medium parameters (4-16 Float32s) 
        256,     // Small arrays (64 Float32s)
        1024,    // Medium arrays (256 Float32s)
        4096,    // Large arrays (1K Float32s)
        16384,   // Very large arrays (4K Float32s)
        65536,   // Huge arrays (16K Float32s)
        262144,  // Matrix operations (64K Float32s)
        1048576, // Large matrix operations (256K Float32s)
        4194304  // Very large matrix operations (1M Float32s)
      ]
    }
    
    private func bucketSize(for requestedSize: Int) -> Int {
      return bucketSizes.first { $0 >= requestedSize } ?? requestedSize
    }
    
    func getBuffer(size: Int) -> MTLBuffer? {
      let bucketSize = self.bucketSize(for: size)
      
      lock.lock()
      defer { lock.unlock() }
      
      if let pool = pools[bucketSize], let buffer = pool.last {
        pools[bucketSize]?.removeLast()
        return buffer
      }
      
      // Create new buffer if pool is empty
      return device.makeBuffer(length: bucketSize, options: .storageModeShared)
    }
    
    func getBuffer<T>(from data: [T]) -> MTLBuffer? {
      let size = data.count * MemoryLayout<T>.stride
      
      if let buffer = getBuffer(size: size) {
        // Copy data to buffer
        let contents = buffer.contents()
        data.withUnsafeBufferPointer { bufferPtr in
          contents.copyMemory(from: bufferPtr.baseAddress!, byteCount: size)
        }
        return buffer
      }
      
      // Fallback to direct creation if pool fails
      return device.makeBuffer(bytes: data, length: size, options: .storageModeShared)
    }
    
    func returnBuffer(_ buffer: MTLBuffer) {
      let bufferSize = buffer.length
      let bucketSize = self.bucketSize(for: bufferSize)
      
      lock.lock()
      defer { lock.unlock() }
      
      if pools[bucketSize] == nil {
        pools[bucketSize] = []
      }
      
      if let pool = pools[bucketSize], pool.count < maxPoolSize {
        pools[bucketSize]?.append(buffer)
      }
      // If pool is full, buffer will be deallocated automatically
    }
    
    func clear() {
      lock.lock()
      defer { lock.unlock() }
      pools.removeAll()
    }
    
    func stats() -> [String: Int] {
      lock.lock()
      defer { lock.unlock() }
      return Dictionary(uniqueKeysWithValues: pools.map { (String($0.key), $0.value.count) })
    }
  }
  
  private lazy var bufferPool = BufferPool(device: config.device)
  
  // MARK: - Optimized Buffer Management
  
  private func createBuffer<T>(from data: [T], type: T.Type) -> MTLBuffer? {
    return bufferPool.getBuffer(from: data)
  }
  
  private func createBuffer<T>(count: Int, type: T.Type) -> MTLBuffer? {
    let size = count * MemoryLayout<T>.stride
    return bufferPool.getBuffer(size: size)
  }
  
  private func returnBuffer(_ buffer: MTLBuffer?) {
    guard let buffer = buffer else { return }
    bufferPool.returnBuffer(buffer)
  }

  private func executeCommandAsync(
    _ commandBuffer: MTLCommandBuffer,
    completion: @escaping () -> Void = {}
  ) {
//    defer { metalLock.unlock() }
//    metalLock.lock()
    
    commandBuffer.addCompletedHandler { _ in
      completion()
    }
    
   // metalLock.withLock {
    //commandBuffersToExectute.append(commandBuffer)
    //}
    
    commandBuffer.commit()
  }
  
  public func executePendingCommands() {
//    for commandBuffer in commandBuffersToExectute {
//      commandBuffer.commit()
//    }
//    
////    for commandBuffer in commandBuffersToExectute {
////      commandBuffer.waitUntilCompleted()
////    }
//    
//    commandBuffersToExectute.removeAll()
  }
  
  // MARK: - Parallel Reduction Helper
  
  private func executeParallelReduction<T>(
    pipeline: MTLComputePipelineState,
    inputBuffer: MTLBuffer,
    elementCount: Int,
    type: T.Type,
    finalReduction: @escaping ([T]) -> T
  ) -> T? {
    let threadsPerThreadgroup = getOptimal1DThreadgroupSize(elementCount: elementCount)
    let threadgroups = (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup
    let sharedMemorySize = threadsPerThreadgroup * MemoryLayout<T>.stride
    
    guard let resultBuffer = createBuffer(count: threadgroups, type: type),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return nil
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(sizeBuffer, offset: 0, index: 2)
    encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)
    
    let threadgroupSize = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
    let threadgroupCount = MTLSize(width: threadgroups, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    // Get partial results from each threadgroup
    let resultPointer = resultBuffer.contents().bindMemory(to: type, capacity: threadgroups)
    let partialResults = Array(UnsafeBufferPointer(start: resultPointer, count: threadgroups))
    
    // Return buffers to pool
    returnBuffer(resultBuffer)
    returnBuffer(sizeBuffer)
    
    // Final reduction on CPU (small array now)
    return finalReduction(partialResults)
  }
  
  // MARK: - Tiled Matrix Multiplication Helper
  
  private func executeTiledMatmul<T>(
    pipeline: MTLComputePipelineState,
    aBuffer: MTLBuffer,
    bBuffer: MTLBuffer,
    aRows: Int,
    aCols: Int,
    bCols: Int,
    type: T.Type
  ) -> (MTLBuffer?, MTLCommandBuffer?) {
    let tileSize = getOptimalTileSize(aRows: aRows, aCols: aCols, bCols: bCols)
    let sharedMemorySize = tileSize * tileSize * MemoryLayout<T>.stride
    
    let aSize = NSC_Size(rows: Int32(aRows), columns: Int32(aCols))
    let bSize = NSC_Size(rows: Int32(aCols), columns: Int32(bCols))
    
    guard let resultBuffer = createBuffer(count: aRows * bCols, type: type),
          let aSizeBuffer = createBuffer(from: [aSize], type: NSC_Size.self),
          let bSizeBuffer = createBuffer(from: [bSize], type: NSC_Size.self) else {
      return (nil, nil)
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return (nil, nil)
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(aSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(bSizeBuffer, offset: 0, index: 4)
    encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0) // shared_a
    encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 1) // shared_b
    
    let threadsPerThreadgroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
    let threadgroups = MTLSize(
      width: (bCols + tileSize - 1) / tileSize,
      height: (aRows + tileSize - 1) / tileSize,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    return (resultBuffer, commandBuffer)
  }
  
  // MARK: - Batched Operations (Neural Network Optimized)
  
  public func batchedMatmul(_ a: [[[Float]]], _ b: [[[Float]]]) -> [[[Float]]] {
    guard a.count == b.count else {
      fatalError("Batch sizes must match")
    }
    
    let batchSize = a.count
    guard batchSize > 0 else { return [] }
    
    let aRows = a[0].count
    let aCols = a[0][0].count
    let bRows = b[0].count
    let bCols = b[0][0].count
    
    guard aCols == bRows else {
      fatalError("Matrix dimensions don't match for multiplication")
    }
    
    let elementCount = batchSize * aRows * bCols
    
    if !shouldUseMetal(for: elementCount) {
      // CPU fallback - process each batch individually
      var results = [[[Float]]]()
      for i in 0..<batchSize {
        results.append(matmul(a[i], b[i]))
      }
      return results
    }
    
    guard let pipeline = computePipeline(for: "nsc_batched_matmul_float_kernel") else {
      // Fallback to individual matrix multiplications
      var results = [[[Float]]]()
      for i in 0..<batchSize {
        results.append(matmul(a[i], b[i]))
      }
      return results
    }
    
    // Flatten all batches into single arrays
    let aFlat = a.flatMap { $0.flatMap { $0 } }
    let bFlat = b.flatMap { $0.flatMap { $0 } }
    
    let aSize = NSC_Size(rows: Int32(aRows), columns: Int32(aCols))
    let bSize = NSC_Size(rows: Int32(bRows), columns: Int32(bCols))
    
    guard let aBuffer = createBuffer(from: aFlat, type: Float.self),
          let bBuffer = createBuffer(from: bFlat, type: Float.self),
          let resultBuffer = createBuffer(count: batchSize * aRows * bCols, type: Float.self),
          let aSizeBuffer = createBuffer(from: [aSize], type: NSC_Size.self),
          let bSizeBuffer = createBuffer(from: [bSize], type: NSC_Size.self),
          let batchSizeBuffer = createBuffer(from: [UInt32(batchSize)], type: UInt32.self) else {
      // Fallback to individual matrix multiplications
      var results = [[[Float]]]()
      for i in 0..<batchSize {
        results.append(matmul(a[i], b[i]))
      }
      return results
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(aSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(bSizeBuffer, offset: 0, index: 4)
    encoder.setBuffer(batchSizeBuffer, offset: 0, index: 5)
    
    let threadsPerThreadgroup = getOptimalThreadgroupSize(width: 256, height: 1)
    let threadgroups = MTLSize(
      width: (bCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (aRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: batchSize
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * aRows * bCols)
    let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: batchSize * aRows * bCols))
    
    // Convert back to 3D array
    var results = [[[Float]]]()
    for batch in 0..<batchSize {
      var batchResult = [[Float]]()
      for row in 0..<aRows {
        let startIndex = batch * aRows * bCols + row * bCols
        let endIndex = startIndex + bCols
        batchResult.append(Array(resultFlat[startIndex..<endIndex]))
      }
      results.append(batchResult)
    }

    return results
  }
  
  public func batchedMatmul(_ a: [[[Float16]]], _ b: [[[Float16]]]) -> [[[Float16]]] {
    guard a.count == b.count else {
      fatalError("Batch sizes must match")
    }
    
    let batchSize = a.count
    guard batchSize > 0 else { return [] }
    
    let aRows = a[0].count
    let aCols = a[0][0].count
    let bRows = b[0].count
    let bCols = b[0][0].count
    
    guard aCols == bRows else {
      fatalError("Matrix dimensions don't match for multiplication")
    }
    
    let elementCount = batchSize * aRows * bCols
    
    if !shouldUseMetal(for: elementCount) {
      // CPU fallback - process each batch individually
      var results = [[[Float16]]]()
      for i in 0..<batchSize {
        results.append(matmul(a[i], b[i]))
      }
      return results
    }
    
    guard let pipeline = computePipeline(for: "nsc_batched_matmul_kernel") else {
      // Fallback to individual matrix multiplications
      var results = [[[Float16]]]()
      for i in 0..<batchSize {
        results.append(matmul(a[i], b[i]))
      }
      return results
    }
    
    // Flatten all batches into single arrays
    let aFlat = a.flatMap { $0.flatMap { $0 } }
    let bFlat = b.flatMap { $0.flatMap { $0 } }
    
    let aSize = NSC_Size(rows: Int32(aRows), columns: Int32(aCols))
    let bSize = NSC_Size(rows: Int32(bRows), columns: Int32(bCols))
    
    guard let aBuffer = createBuffer(from: aFlat, type: Float16.self),
          let bBuffer = createBuffer(from: bFlat, type: Float16.self),
          let resultBuffer = createBuffer(count: batchSize * aRows * bCols, type: Float16.self),
          let aSizeBuffer = createBuffer(from: [aSize], type: NSC_Size.self),
          let bSizeBuffer = createBuffer(from: [bSize], type: NSC_Size.self),
          let batchSizeBuffer = createBuffer(from: [UInt32(batchSize)], type: UInt32.self) else {
      // Fallback to individual matrix multiplications
      var results = [[[Float16]]]()
      for i in 0..<batchSize {
        results.append(matmul(a[i], b[i]))
      }
      return results
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(aSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(bSizeBuffer, offset: 0, index: 4)
    encoder.setBuffer(batchSizeBuffer, offset: 0, index: 5)
    
    let threadsPerThreadgroup = getOptimalThreadgroupSize(width: 256, height: 1)
    let threadgroups = MTLSize(
      width: (bCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (aRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: batchSize
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: batchSize * aRows * bCols)
    let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: batchSize * aRows * bCols))
    
    // Convert back to 3D array
    var results = [[[Float16]]]()
    for batch in 0..<batchSize {
      var batchResult = [[Float16]]()
      for row in 0..<aRows {
        let startIndex = batch * aRows * bCols + row * bCols
        let endIndex = startIndex + bCols
        batchResult.append(Array(resultFlat[startIndex..<endIndex]))
      }
      results.append(batchResult)
    }
    
    return results
  }
  
  // MARK: - Basic Array Operations
  
  public func sum(_ array: [Float]) -> Float {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.reduce(0, +)
    }
    
    guard let inputBuffer = createBuffer(from: array, type: Float.self) else {
      return array.reduce(0, +)
    }
    
    // Use parallel reduction for large arrays
    if shouldUseParallelReduction(for: elementCount) {
      guard let pipeline = computePipeline(for: "nsc_parallel_sum_float_kernel") else {
        return array.reduce(0, +)
      }
      
      let result = executeParallelReduction(
        pipeline: pipeline,
        inputBuffer: inputBuffer,
        elementCount: elementCount,
        type: Float.self,
        finalReduction: { $0.reduce(0, +) }
      ) ?? array.reduce(0, +)
      
      return result
    }
    
    // Use single-threaded kernel for smaller arrays
    guard let pipeline = computePipeline(for: "nsc_sum_float_kernel"),
          let resultBuffer = createBuffer(count: 1, type: Float.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.reduce(0, +)
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1)
    let result = resultPointer[0]
    
    // Return buffers to pool
    returnBuffer(inputBuffer)
    returnBuffer(resultBuffer)
    returnBuffer(sizeBuffer)
    
    return result
  }
  
  public func sum(_ array: [Float16]) -> Float16 {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_sum(ptr.baseAddress!)
      }
    }
    
    guard let inputBuffer = createBuffer(from: array, type: Float16.self) else {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_sum(ptr.baseAddress!)
      }
    }
    
    // Use parallel reduction for large arrays
    if shouldUseParallelReduction(for: elementCount) {
      guard let pipeline = computePipeline(for: "nsc_parallel_sum_kernel") else {
        return array.withUnsafeBufferPointer { ptr in
          return nsc_sum(ptr.baseAddress!)
        }
      }
      
      let result = executeParallelReduction(
        pipeline: pipeline,
        inputBuffer: inputBuffer,
        elementCount: elementCount,
        type: Float16.self,
        finalReduction: { partialResults in
          return partialResults.withUnsafeBufferPointer { ptr in
            var result: Float16 = 0
            for i in 0..<partialResults.count {
              result += ptr[i]
            }
            return result
          }
        }
      ) ?? array.withUnsafeBufferPointer { ptr in
        return nsc_sum(ptr.baseAddress!)
      }
      
      return result
    }
    
    // Use single-threaded kernel for smaller arrays
    guard let pipeline = computePipeline(for: "nsc_sum_kernel"),
          let resultBuffer = createBuffer(count: 1, type: Float16.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_sum(ptr.baseAddress!)
      }
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: 1)
    let result = resultPointer[0]
    
    return result
  }
  
  public func max(_ array: [Float]) -> Float {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.max() ?? 0
    }
    
    guard let inputBuffer = createBuffer(from: array, type: Float.self) else {
      return array.max() ?? 0
    }
    
    // Use parallel reduction for large arrays
    if shouldUseParallelReduction(for: elementCount) {
      guard let pipeline = computePipeline(for: "nsc_parallel_max_float_kernel") else {
        return array.max() ?? 0
      }
      
      return executeParallelReduction(
        pipeline: pipeline,
        inputBuffer: inputBuffer,
        elementCount: elementCount,
        type: Float.self,
        finalReduction: { $0.max() ?? -Float.infinity }
      ) ?? array.max() ?? 0
    }
    
    // Use single-threaded kernel for smaller arrays
    guard let pipeline = computePipeline(for: "nsc_max_float_kernel"),
          let resultBuffer = createBuffer(count: 1, type: Float.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.max() ?? 0
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1)
    return resultPointer[0]
  }
  
  public func min(_ array: [Float]) -> Float {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.min() ?? 0
    }
    
    guard let inputBuffer = createBuffer(from: array, type: Float.self) else {
      return array.min() ?? 0
    }
    
    // Use parallel reduction for large arrays
    if shouldUseParallelReduction(for: elementCount) {
      guard let pipeline = computePipeline(for: "nsc_parallel_min_float_kernel") else {
        return array.min() ?? 0
      }
      
      return executeParallelReduction(
        pipeline: pipeline,
        inputBuffer: inputBuffer,
        elementCount: elementCount,
        type: Float.self,
        finalReduction: { $0.min() ?? Float.infinity }
      ) ?? array.min() ?? 0
    }
    
    // Use single-threaded kernel for smaller arrays
    guard let pipeline = computePipeline(for: "nsc_min_float_kernel"),
          let resultBuffer = createBuffer(count: 1, type: Float.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.min() ?? 0
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let inputBuffer = createBuffer(from: array, type: Float16.self) else {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_max(ptr.baseAddress!)
      }
    }
    
    // Use parallel reduction for large arrays
    if shouldUseParallelReduction(for: elementCount) {
      guard let pipeline = computePipeline(for: "nsc_parallel_max_kernel") else {
        return array.withUnsafeBufferPointer { ptr in
          return nsc_max(ptr.baseAddress!)
        }
      }
      
      return executeParallelReduction(
        pipeline: pipeline,
        inputBuffer: inputBuffer,
        elementCount: elementCount,
        type: Float16.self,
        finalReduction: { partialResults in
          var maxVal: Float16 = -Float16.infinity
          for val in partialResults {
            if val > maxVal {
              maxVal = val
            }
          }
          return maxVal
        }
      ) ?? array.withUnsafeBufferPointer { ptr in
        return nsc_max(ptr.baseAddress!)
      }
    }
    
    // Use single-threaded kernel for smaller arrays
    guard let pipeline = computePipeline(for: "nsc_max_kernel"),
          let resultBuffer = createBuffer(count: 1, type: Float16.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_max(ptr.baseAddress!)
      }
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: 1)
    let result = resultPointer[0]
    
    // Return buffers to pool
    returnBuffer(inputBuffer)
    returnBuffer(resultBuffer)
    returnBuffer(sizeBuffer)
    
    return result
  }
  
  public func min(_ array: [Float16]) -> Float16 {
    let elementCount = array.count
    
    if !shouldUseMetal(for: elementCount) {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_min(ptr.baseAddress!)
      }
    }
    
    guard let inputBuffer = createBuffer(from: array, type: Float16.self) else {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_min(ptr.baseAddress!)
      }
    }
    
    // Use parallel reduction for large arrays
    if shouldUseParallelReduction(for: elementCount) {
      guard let pipeline = computePipeline(for: "nsc_parallel_min_kernel") else {
        return array.withUnsafeBufferPointer { ptr in
          return nsc_min(ptr.baseAddress!)
        }
      }
      
      return executeParallelReduction(
        pipeline: pipeline,
        inputBuffer: inputBuffer,
        elementCount: elementCount,
        type: Float16.self,
        finalReduction: { partialResults in
          var minVal: Float16 = Float16.infinity
          for val in partialResults {
            if val < minVal {
              minVal = val
            }
          }
          return minVal
        }
      ) ?? array.withUnsafeBufferPointer { ptr in
        return nsc_min(ptr.baseAddress!)
      }
    }
    
    // Use single-threaded kernel for smaller arrays
    guard let pipeline = computePipeline(for: "nsc_min_kernel"),
          let resultBuffer = createBuffer(count: 1, type: Float16.self),
          let sizeBuffer = createBuffer(from: [UInt32(elementCount)], type: UInt32.self) else {
      return array.withUnsafeBufferPointer { ptr in
        return nsc_min(ptr.baseAddress!)
      }
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
  
  // MARK: - Matrix Operations
  
  public func matmul(_ a: [[Float]], _ b: [[Float]], completion: (([[Float]]) -> ())? = nil) -> [[Float]] {
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
    
    let aFlat = a.flatMap { $0 }
    let bFlat = b.flatMap { $0 }
    
    guard let aBuffer = createBuffer(from: aFlat, type: Float.self),
          let bBuffer = createBuffer(from: bFlat, type: Float.self) else {
      fatalError("Failed to create buffers")
    }
    
    // Use tiled matrix multiplication for large matrices
    let resultBuffer: MTLBuffer?
    var runningCommandBuffer: MTLCommandBuffer?
    
    func processResult(resultBuffer: MTLBuffer) -> [[Float]] {
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
    
    if shouldUseTiledMatmul(aRows: aRows, aCols: aCols, bCols: bCols) {
      guard let pipeline = computePipeline(for: "nsc_tiled_matmul_float_kernel") else {
        fatalError("Failed to create tiled matmul pipeline")
      }
      
      (resultBuffer, runningCommandBuffer) = executeTiledMatmul(
        pipeline: pipeline,
        aBuffer: aBuffer,
        bBuffer: bBuffer,
        aRows: aRows,
        aCols: aCols,
        bCols: bCols,
        type: Float.self
      )
      
    } else {
      // Use simple matrix multiplication for smaller matrices
      guard let pipeline = computePipeline(for: "nsc_matmul_float_kernel") else {
        fatalError("Failed to create matmul pipeline")
      }
      
      let aSize = NSC_Size(rows: Int32(aRows), columns: Int32(aCols))
      let bSize = NSC_Size(rows: Int32(bRows), columns: Int32(bCols))
      
      guard let buffer = createBuffer(count: aRows * bCols, type: Float.self),
            let aSizeBuffer = createBuffer(from: [aSize], type: NSC_Size.self),
            let bSizeBuffer = createBuffer(from: [bSize], type: NSC_Size.self) else {
        fatalError("Failed to create buffers")
      }
      
      guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder() else {
        return []
      }
      
      runningCommandBuffer = commandBuffer
      
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(aBuffer, offset: 0, index: 0)
      encoder.setBuffer(bBuffer, offset: 0, index: 1)
      encoder.setBuffer(buffer, offset: 0, index: 2)
      encoder.setBuffer(aSizeBuffer, offset: 0, index: 3)
      encoder.setBuffer(bSizeBuffer, offset: 0, index: 4)
      
      let threadsPerThreadgroup = getOptimalThreadgroupSize(width: 256, height: 1)
      let threadgroups = MTLSize(
        width: (bCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
        height: (aRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
        depth: 1
      )
      
      encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
      encoder.endEncoding()
      
      resultBuffer = buffer
    }
    
    guard let finalBuffer = resultBuffer,
          let commandBuffer = runningCommandBuffer else {
      fatalError("Matrix multiplication failed")
    }
    
    // Execute with GPU pipeline optimization
    if isAsyncMode == false {
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    } else {
      
      executeCommandAsync(commandBuffer) {
        completion?(processResult(resultBuffer: finalBuffer))
      }
      
      return []
    }
  
    let processedResult = processResult(resultBuffer: finalBuffer)
    
    completion?(processedResult)

    return processedResult
  }
  
  public func matmul(_ a: [[Float16]], _ b: [[Float16]], completion: (([[Float16]]) -> ())? = nil) -> [[Float16]] {
    let aRows = a.count
    let aCols = a[0].count
    let bRows = b.count
    let bCols = b[0].count
    
    guard aCols == bRows else {
      fatalError("Matrix dimensions don't match for multiplication")
    }
    
    let elementCount = aRows * bCols
    
    if !shouldUseMetal(for: elementCount) {
      var result = [[Float16]]()
      for i in 0..<aRows {
        var row = [Float16]()
        for j in 0..<bCols {
          var sum: Float16 = 0
          for k in 0..<aCols {
            sum += a[i][k] * b[k][j]
          }
          row.append(sum)
        }
        result.append(row)
      }
      return result
    }
    
    let aFlat = a.flatMap { $0 }
    let bFlat = b.flatMap { $0 }
    
    guard let aBuffer = createBuffer(from: aFlat, type: Float16.self),
          let bBuffer = createBuffer(from: bFlat, type: Float16.self) else {
      fatalError("Failed to create buffers")
    }
    
    // Use tiled matrix multiplication for large matrices
    let resultBuffer: MTLBuffer?
    var runningCommandBuffer: MTLCommandBuffer?
    
    func processResult(resultBuffer: MTLBuffer) -> [[Float16]] {
      let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: aRows * bCols)
      let resultFlat = Array(UnsafeBufferPointer(start: resultPointer, count: aRows * bCols))
      
      var result = [[Float16]]()
      for i in 0..<aRows {
        let startIndex = i * bCols
        let endIndex = startIndex + bCols
        result.append(Array(resultFlat[startIndex..<endIndex]))
      }
      
      return result
    }
    
    if shouldUseTiledMatmul(aRows: aRows, aCols: aCols, bCols: bCols) {
      guard let pipeline = computePipeline(for: "nsc_tiled_matmul_float_kernel") else {
        fatalError("Failed to create tiled matmul pipeline")
      }
      
      (resultBuffer, runningCommandBuffer) = executeTiledMatmul(
        pipeline: pipeline,
        aBuffer: aBuffer,
        bBuffer: bBuffer,
        aRows: aRows,
        aCols: aCols,
        bCols: bCols,
        type: Float.self
      )
      
    } else {
      // Use simple matrix multiplication for smaller matrices
      guard let pipeline = computePipeline(for: "nsc_matmul_float_kernel") else {
        fatalError("Failed to create matmul pipeline")
      }
      
      let aSize = NSC_Size(rows: Int32(aRows), columns: Int32(aCols))
      let bSize = NSC_Size(rows: Int32(bRows), columns: Int32(bCols))
      
      guard let buffer = createBuffer(count: aRows * bCols, type: Float.self),
            let aSizeBuffer = createBuffer(from: [aSize], type: NSC_Size.self),
            let bSizeBuffer = createBuffer(from: [bSize], type: NSC_Size.self) else {
        fatalError("Failed to create buffers")
      }
      
      guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder() else {
        return []
      }
      
      runningCommandBuffer = commandBuffer
      
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(aBuffer, offset: 0, index: 0)
      encoder.setBuffer(bBuffer, offset: 0, index: 1)
      encoder.setBuffer(buffer, offset: 0, index: 2)
      encoder.setBuffer(aSizeBuffer, offset: 0, index: 3)
      encoder.setBuffer(bSizeBuffer, offset: 0, index: 4)
      
      let threadsPerThreadgroup = getOptimalThreadgroupSize(width: 256, height: 1)
      let threadgroups = MTLSize(
        width: (bCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
        height: (aRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
        depth: 1
      )
      
      encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
      encoder.endEncoding()
      
      resultBuffer = buffer
    }
    
    guard let finalBuffer = resultBuffer,
          let commandBuffer = runningCommandBuffer else {
      fatalError("Matrix multiplication failed")
    }
    
    // Execute with GPU pipeline optimization


    if isAsyncMode == false {
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    } else {
      executeCommandAsync(commandBuffer) {
        completion?(processResult(resultBuffer: finalBuffer))
      }
      
      return []
    }
    
    let processedResult = processResult(resultBuffer: finalBuffer)
    
    completion?(processedResult)

    return processedResult
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    let threadsPerThreadgroup = getOptimalThreadgroupSize(width: 256, height: 1)
    let threadgroups = MTLSize(
      width: (inputCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (inputRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    let threadsPerThreadgroup = getOptimalThreadgroupSize(width: 256, height: 1)
    let threadgroups = MTLSize(
      width: (inputCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (inputRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
  
  public func conv2d(_ signal: [[Float]], _ filter: [[Float]],
                     stride: (rows: Int, cols: Int) = (1, 1),
                     padding: NumSwift.ConvPadding = .valid,
                     completion: (([[Float]]) -> ())? = nil) -> [[Float]] {
    
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
    
    func processResult(resultBuffer: MTLBuffer) -> [[Float]] {
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
    
    if !shouldUseMetal(for: elementCount) {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    let signalFlat = signal.flatMap { $0 }
    let filterFlat = filter.flatMap { $0 }
    
    guard let signalBuffer = createBuffer(from: signalFlat, type: Float.self),
          let filterBuffer = createBuffer(from: filterFlat, type: Float.self) else {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    // Use optimized convolution kernels based on problem characteristics
    let pipeline: MTLComputePipelineState?
    let useSharedMemory = shouldUseOptimizedConv(inputHeight: inputRows, inputWidth: inputCols, filterSize: Swift.max(filterRows, filterCols))
    
    if useSharedMemory {
      pipeline = computePipeline(for: "nsc_im2col_conv2d_float_kernel")
    } else {
      pipeline = computePipeline(for: "nsc_conv2d_float_kernel")
    }
    
    guard let convPipeline = pipeline else {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    let inputSize = NSC_Size(rows: Int32(inputRows), columns: Int32(inputCols))
    let filterSize = NSC_Size(rows: Int32(filterRows), columns: Int32(filterCols))
    let strideSize = NSC_Size(rows: Int32(stride.rows), columns: Int32(stride.cols))
    let resultSize = NSC_Size(rows: Int32(outputRows), columns: Int32(outputCols))
    let paddingValue = Int32(padding == .same ? 1 : 0)
    let padTopValue = Int32(padTop)
    let padLeftValue = Int32(padLeft)
    
    guard let resultBuffer = createBuffer(count: outputRows * outputCols, type: Float.self),
          let inputSizeBuffer = createBuffer(from: [inputSize], type: NSC_Size.self),
          let filterSizeBuffer = createBuffer(from: [filterSize], type: NSC_Size.self),
          let strideSizeBuffer = createBuffer(from: [strideSize], type: NSC_Size.self),
          let resultSizeBuffer = createBuffer(from: [resultSize], type: NSC_Size.self),
          let padTopBuffer = createBuffer(from: [padTopValue], type: Int32.self),
          let padLeftBuffer = createBuffer(from: [padLeftValue], type: Int32.self) else {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(convPipeline)
    encoder.setBuffer(signalBuffer, offset: 0, index: 0)
    encoder.setBuffer(filterBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(inputSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(filterSizeBuffer, offset: 0, index: 4)
    encoder.setBuffer(strideSizeBuffer, offset: 0, index: 5)
    encoder.setBuffer(resultSizeBuffer, offset: 0, index: 6)
    encoder.setBuffer(padTopBuffer, offset: 0, index: 7)
    encoder.setBuffer(padLeftBuffer, offset: 0, index: 8)
    
    if useSharedMemory {
      // Set shared memory for im2col kernel
      let sharedMemorySize = 16 * 16 * MemoryLayout<Float>.stride
      encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)
    } else {
      // Add padding buffer for legacy kernel
      let paddingBuffer = createBuffer(from: [paddingValue], type: Int32.self)
      encoder.setBuffer(paddingBuffer, offset: 0, index: 7)
      encoder.setBuffer(padTopBuffer, offset: 0, index: 8)
      encoder.setBuffer(padLeftBuffer, offset: 0, index: 9)
    }
    
    let threadsPerThreadgroup = getOptimalThreadgroupSize(width: 256, height: 1)
    let threadgroups = MTLSize(
      width: (outputCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (outputRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    // Execute with GPU pipeline optimization
    if isAsyncMode == false {
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    } else {
      executeCommandAsync(commandBuffer) {
        completion?(processResult(resultBuffer: resultBuffer))
      }
      
      return []
    }

    let processedResult = processResult(resultBuffer: resultBuffer)
    
    completion?(processedResult)

    return processedResult
  }
  
  // TODO: migrate to async support
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
    
    let signalFlat = signal.flatMap { $0 }
    let filterFlat = filter.flatMap { $0 }
    
    guard let signalBuffer = createBuffer(from: signalFlat, type: Float16.self),
          let filterBuffer = createBuffer(from: filterFlat, type: Float16.self) else {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    // Use optimized convolution kernels based on problem characteristics
    let pipeline: MTLComputePipelineState?
    let useSharedMemory = shouldUseOptimizedConv(inputHeight: inputRows, inputWidth: inputCols, filterSize: Swift.max(filterRows, filterCols))
    
    if useSharedMemory {
      pipeline = computePipeline(for: "nsc_im2col_conv2d_kernel")
    } else {
      pipeline = computePipeline(for: "nsc_conv2d_kernel")
    }
    
    guard let convPipeline = pipeline else {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    let inputSize = NSC_Size(rows: Int32(inputRows), columns: Int32(inputCols))
    let filterSize = NSC_Size(rows: Int32(filterRows), columns: Int32(filterCols))
    let strideSize = NSC_Size(rows: Int32(stride.rows), columns: Int32(stride.cols))
    let resultSize = NSC_Size(rows: Int32(outputRows), columns: Int32(outputCols))
    let paddingValue = Int32(padding == .same ? 1 : 0)
    let padTopValue = Int32(padTop)
    let padLeftValue = Int32(padLeft)
    
    guard let resultBuffer = createBuffer(count: outputRows * outputCols, type: Float16.self),
          let inputSizeBuffer = createBuffer(from: [inputSize], type: NSC_Size.self),
          let filterSizeBuffer = createBuffer(from: [filterSize], type: NSC_Size.self),
          let strideSizeBuffer = createBuffer(from: [strideSize], type: NSC_Size.self),
          let resultSizeBuffer = createBuffer(from: [resultSize], type: NSC_Size.self),
          let padTopBuffer = createBuffer(from: [padTopValue], type: Int32.self),
          let padLeftBuffer = createBuffer(from: [padLeftValue], type: Int32.self) else {
      return NumSwiftC.conv2d(signal: signal,
                              filter: filter,
                              strides: stride,
                              padding: padding,
                              filterSize: (filterRows, filterCols),
                              inputSize: (inputRows, inputCols))
    }
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return []
    }
    
    encoder.setComputePipelineState(convPipeline)
    encoder.setBuffer(signalBuffer, offset: 0, index: 0)
    encoder.setBuffer(filterBuffer, offset: 0, index: 1)
    encoder.setBuffer(resultBuffer, offset: 0, index: 2)
    encoder.setBuffer(inputSizeBuffer, offset: 0, index: 3)
    encoder.setBuffer(filterSizeBuffer, offset: 0, index: 4)
    encoder.setBuffer(strideSizeBuffer, offset: 0, index: 5)
    encoder.setBuffer(resultSizeBuffer, offset: 0, index: 6)
    encoder.setBuffer(padTopBuffer, offset: 0, index: 7)
    encoder.setBuffer(padLeftBuffer, offset: 0, index: 8)
    
    if useSharedMemory {
      // Set shared memory for im2col kernel
      let sharedMemorySize = 16 * 16 * MemoryLayout<Float16>.stride
      encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)
    } else {
      // Add padding buffer for legacy kernel
      let paddingBuffer = createBuffer(from: [paddingValue], type: Int32.self)
      encoder.setBuffer(paddingBuffer, offset: 0, index: 7)
      encoder.setBuffer(padTopBuffer, offset: 0, index: 8)
      encoder.setBuffer(padLeftBuffer, offset: 0, index: 9)
    }
    
    let threadsPerThreadgroup = getOptimalThreadgroupSize(width: 256, height: 1)
    let threadgroups = MTLSize(
      width: (outputCols + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
      height: (outputRows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
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
    
    guard let commandBuffer = getCommandQueue().makeCommandBuffer(),
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
    
    // Execute with GPU pipeline optimization
    executeCommandAsync(commandBuffer)
    
    // In async mode, we still need to wait for THIS operation but allow others to pipeline
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.waitUntilCompleted()
    let duration = CFAbsoluteTimeGetCurrent() - startTime
    
    if duration > 1.0 {  // Log operations taking >1 second
      print("⚠️  Slow GPU operation: \(String(format: "%.2f", duration))s")
    }
    
    let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
  }
}