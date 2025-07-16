import Foundation
import Metal
import MetalKit
import NumSwiftC

// MARK: - Metal Backend Configuration

public enum ComputeBackend {
    case cpu
    case metal
    case auto // Automatically chooses based on problem size
}

public struct MetalConfiguration {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    
    public init?(device: MTLDevice? = nil) {
        self.device = device ?? MTLCreateSystemDefaultDevice() ?? { return nil }()
        guard let commandQueue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = commandQueue
        
        guard let library = self.device.makeDefaultLibrary() else { return nil }
        self.library = library
    }
}

// MARK: - NumSwift Metal Backend

public class NumSwiftMetal {
    private let config: MetalConfiguration
    private var backend: ComputeBackend = .auto
    
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
        if let cached = pipelineCache[functionName] {
            return cached
        }
        
        guard let function = config.library.makeFunction(name: functionName) else {
            print("Failed to create function: \(functionName)")
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
        
        let threadsPerThreadgroup = MTLSize(width: min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
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
        
        let threadsPerThreadgroup = MTLSize(width: min(pipeline.threadExecutionWidth, elementCount), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (elementCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let resultPointer = resultBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
        return Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
    }
    
    // MARK: - Matrix Operations
    
    public func matmul(_ a: [[Float16]], _ b: [[Float16]]) -> [[Float16]] {
        let aRows = a.count
        let aCols = a[0].count
        let bRows = b.count
        let bCols = b[0].count
        
        guard aCols == bRows else {
            fatalError("Matrix dimensions don't match for multiplication")
        }
        
        let elementCount = aRows * bCols
        
        if !shouldUseMetal(for: elementCount) {
            // Fallback to CPU implementation
            let aSize = NSC_Size(rows: Int32(aRows), columns: Int32(aCols))
            let bSize = NSC_Size(rows: Int32(bRows), columns: Int32(bCols))
            
            // Convert to flat arrays for C function
            let aFlat = a.flatMap { $0 }
            let bFlat = b.flatMap { $0 }
            var resultFlat = [Float16](repeating: 0, count: aRows * bCols)
            
            // Create array of pointers for C function
            var aPointers = a.map { $0.withUnsafeBufferPointer { $0.baseAddress! } }
            var bPointers = b.map { $0.withUnsafeBufferPointer { $0.baseAddress! } }
            var resultPointers = (0..<aRows).map { _ in 
                UnsafeMutablePointer<Float16>.allocate(capacity: bCols)
            }
            
            defer {
                resultPointers.forEach { $0.deallocate() }
            }
            
            aPointers.withUnsafeMutableBufferPointer { aPtr in
                bPointers.withUnsafeMutableBufferPointer { bPtr in
                    resultPointers.withUnsafeMutableBufferPointer { resultPtr in
                        nsc_matmul_16(aSize, bSize, aPtr.baseAddress!, bPtr.baseAddress!, resultPtr.baseAddress!)
                    }
                }
            }
            
            // Convert back to 2D array
            var result = [[Float16]]()
            for i in 0..<aRows {
                let row = Array(UnsafeBufferPointer(start: resultPointers[i], count: bCols))
                result.append(row)
            }
            
            return result
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
    
    // MARK: - Convolution Operations
    
    public func conv2d(_ signal: [[Float16]], _ filter: [[Float16]], stride: (rows: Int, cols: Int) = (1, 1)) -> [[Float16]] {
        let inputRows = signal.count
        let inputCols = signal[0].count
        let filterRows = filter.count
        let filterCols = filter[0].count
        
        let outputRows = (inputRows - filterRows) / stride.rows + 1
        let outputCols = (inputCols - filterCols) / stride.cols + 1
        let elementCount = outputRows * outputCols
        
        if !shouldUseMetal(for: elementCount) {
            // Fallback to CPU implementation
            let signalSize = NSC_Size(rows: Int32(inputRows), columns: Int32(inputCols))
            let filterSize = NSC_Size(rows: Int32(filterRows), columns: Int32(filterCols))
            let strideSize = NSC_Size(rows: Int32(stride.rows), columns: Int32(stride.cols))
            
            // Convert to format expected by C function
            var signalPointers = signal.map { $0.withUnsafeBufferPointer { $0.baseAddress! } }
            var filterPointers = filter.map { $0.withUnsafeBufferPointer { $0.baseAddress! } }
            var resultPointers = (0..<outputRows).map { _ in 
                UnsafeMutablePointer<Float16>.allocate(capacity: outputCols)
            }
            
            defer {
                resultPointers.forEach { $0.deallocate() }
            }
            
            signalPointers.withUnsafeMutableBufferPointer { signalPtr in
                filterPointers.withUnsafeMutableBufferPointer { filterPtr in
                    resultPointers.withUnsafeMutableBufferPointer { resultPtr in
                        nsc_conv2d_f16(signalPtr.baseAddress!, filterPtr.baseAddress!, resultPtr.baseAddress!, strideSize, valid, filterSize, signalSize)
                    }
                }
            }
            
            // Convert back to 2D array
            var result = [[Float16]]()
            for i in 0..<outputRows {
                let row = Array(UnsafeBufferPointer(start: resultPointers[i], count: outputCols))
                result.append(row)
            }
            
            return result
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
        
        guard let signalBuffer = createBuffer(from: signalFlat, type: Float16.self),
              let filterBuffer = createBuffer(from: filterFlat, type: Float16.self),
              let resultBuffer = createBuffer(count: outputRows * outputCols, type: Float16.self),
              let inputSizeBuffer = createBuffer(from: [inputSize], type: NSC_Size.self),
              let filterSizeBuffer = createBuffer(from: [filterSize], type: NSC_Size.self),
              let strideSizeBuffer = createBuffer(from: [strideSize], type: NSC_Size.self),
              let resultSizeBuffer = createBuffer(from: [resultSize], type: NSC_Size.self) else {
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
}

// MARK: - Extension for NSC_Size Compatibility

extension NSC_Size {
    init(rows: Int32, columns: Int32) {
        self.rows = rows
        self.columns = columns
    }
}

// MARK: - Public Interface

public func createNumSwiftMetal() -> NumSwiftMetal? {
    return NumSwiftMetal()
}

// MARK: - Global Instance for Easy Access

public let numSwiftMetal = NumSwiftMetal()