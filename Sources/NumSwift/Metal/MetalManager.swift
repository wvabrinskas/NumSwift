import Foundation
import Metal

public typealias DataType = [[CFloat]]
public typealias ResultType = CFloat

public class MetalManager {
  public static let shared = MetalManager()
  
  public enum MetalFunction: String {
    case activation
    case conv2d
  }
  
  private var currentRunningPipelines: [MTLComputePipelineState] = []
  private var device: MTLDevice? = MTLCreateSystemDefaultDevice()

  private func getFunction(_ function: MetalFunction) -> MTLFunction? {
    return try? device?.makeDefaultLibrary(bundle: Bundle.module).makeFunction(name: function.rawValue)
  }
  
  private func pipelineIfExists(type: MetalFunction) -> MTLComputePipelineState? {
    return self.currentRunningPipelines.filter({ $0.label == type.rawValue }).first
  }
  
  private func addPipeline(for type: MetalFunction) -> MTLComputePipelineState? {
    guard let device = self.device,
          let function = getFunction(type) else {
      return nil
    }
    
    do {
      let descriptor = MTLComputePipelineDescriptor()
      descriptor.label = type.rawValue
      descriptor.computeFunction = function

      let pipeline = try device.makeComputePipelineState(descriptor: descriptor,
                                                         options: [],
                                                         reflection: nil)
      
      self.currentRunningPipelines.append(pipeline)
      return pipeline
      
    } catch {
      print(error)
      return nil
    }
  }
  
  public func conv2D(signal: [[Float]],
                     filter: [[Float]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int)) -> [Float] {
    
    return []
  }
  
  public func activate(_ num: Float,
                       _ activationType: Int) -> Float {
    var data = num
    
    guard let device = self.device else {
      return 0
    }
    
    let pipeline: MTLComputePipelineState? = self.pipelineIfExists(type: .activation) ?? self.addPipeline(for: .activation)

    guard let dataBuffer = device.makeBuffer(bytes: &data,
                                             length: MemoryLayout<Float>.stride,
                                             options: []),
          
          let resultsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride,
                                                options: []) else {
      return 0
    }
    
    
    // Our results in convenient form to compute the actual result later:
    let pointer = resultsBuffer.contents().bindMemory(to: Float.self, capacity: 1)
    let results = UnsafeBufferPointer<Float>(start: pointer, count: 1)

    let queue = self.device?.makeCommandQueue()
    let cmds = queue?.makeCommandBuffer()
    let newEncoder = cmds?.makeComputeCommandEncoder()

    guard let encoder = newEncoder, let pipelineStrong = pipeline else {
      return 0
    }
    
    var activationType = CUnsignedInt(activationType)

    encoder.setComputePipelineState(pipelineStrong)
    
    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultsBuffer, offset: 0, index: 1)
    encoder.setBytes(&activationType, length: MemoryLayout<CUnsignedInt>.size, index: 2)
    
    let threadgroupsPerGrid = MTLSize(width: (1 + pipelineStrong.threadExecutionWidth - 1) / pipelineStrong.threadExecutionWidth,
                                      height: 10,
                                      depth: 10)
    
    let threadsPerThreadgroup = MTLSize(width: pipelineStrong.threadExecutionWidth,
                                        height: 10,
                                        depth: 10)

    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    //execution step
    cmds?.commit()
    cmds?.waitUntilCompleted()
   
    var sum: Float = 0

    for elem in results {
      sum += elem
    }
        
    return sum
  }
}
