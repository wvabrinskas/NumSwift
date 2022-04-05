import Foundation
import Metal
import MetalPerformanceShaders

public typealias DataType = [[CFloat]]
public typealias ResultType = CFloat

public class MetalManager: NSObject {
  public static let shared = MetalManager()
  
  private var incomingWeights: [Float] = []
  private var incomingBiases: [Float] = []
  
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
  
  private func buildImage(size: (rows: Int, columns: Int, depth: Int), data: [Float]? = nil) -> MPSImage {
    guard let device = device else {
      fatalError("no metal device")
    }

    let outputImageDescriptor = MPSImageDescriptor(channelFormat: .float32,
                                                   width: size.columns,
                                                   height: size.rows,
                                                   featureChannels: size.depth)
    
    let outputImage = MPSImage(device: device, imageDescriptor: outputImageDescriptor)
    
    if let data = data {
      let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                             size: MTLSize(width: size.columns,
                                           height: size.rows,
                                           depth: size.depth))
      
      data.withUnsafeBufferPointer { input32 in
        
        let bytesPerRow: Int = MemoryLayout<Float>.size * outputImage.width
        
        let ff16 = UnsafeMutableBufferPointer<Float>.allocate(capacity: data.count)
        
        ff16.initialize(repeating: 0)
              
        let rr16 = UnsafeRawPointer(ff16.baseAddress!)
        
        outputImage.texture.replace(
          region: region,
          mipmapLevel: 0,
          withBytes: rr16,
          bytesPerRow: bytesPerRow)
        
        ff16.deallocate()
      }
    }
    
    return outputImage
  }
  
  private func extractData(image: MPSImage) -> [Float] {
    
    let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                           size: MTLSize(width: image.width,
                                         height: image.height,
                                         depth: image.featureChannels))
    
    let bytesPerRow: Int = MemoryLayout<Float>.size * image.width
    
    let ff16 = UnsafeMutableBufferPointer<Float>.allocate(capacity: image.width * image.height)
    
    image.texture.getBytes(
      UnsafeMutableRawPointer(ff16.baseAddress!),
      bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0
    )
    
    return Array(ff16)
  }
  
  public func conv2D(signal: [[Float]],
                     filter: [[Float]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int, depth: Int),
                     outputSize: (rows: Int, columns: Int, depth: Int)) -> [Float] {
    
    guard let device = self.device else {
      return []
    }
    
    let descriptor = MPSCNNConvolutionDescriptor(kernelWidth: filterSize.columns,
                                                 kernelHeight: filterSize.rows,
                                                 inputFeatureChannels: inputSize.depth,
                                                 outputFeatureChannels: outputSize.depth)
    
    descriptor.strideInPixelsX = strides.1
    descriptor.strideInPixelsY = strides.0
    
    let conv = MPSCNNConvolution(device: de, weights: <#T##MPSCNNConvolutionDataSource#>)
    conv.offset = MPSOffset(x: inputSize.columns / 2, y: inputSize.rows / 2, z: 0)
    conv.edgeMode = .zero
    
    let queue = self.device?.makeCommandQueue()
    let cmds = queue?.makeCommandBuffer()
    
    guard let buffer = cmds else {
      return []
    }
    
    let image = buildImage(size: inputSize, data: signal.flatten())
    let outputImage = buildImage(size: outputSize)
    
    conv.encode(commandBuffer: buffer,
                sourceImage: image,
                destinationImage: outputImage)
    
    buffer.commit()
    buffer.waitUntilCompleted()
    
    return extractData(image: outputImage)
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

extension MetalManager: MPSCNNConvolutionDataSource {
    public func dataType() ->   MPSDataType { .float32 }
  public func descriptor() -> MPSCNNConvolutionDescriptor { convolutionDescriptor }
  public func weights() ->    UnsafeMutableRawPointer { incomingWeights }
  public func biasTerms() ->  UnsafeMutablePointer<Float>? { incomingBiases }
  public func load() -> Bool { true }
  public func purge() { }
  public func label() -> String? { nil }
  public func copy(with zone: NSZone? = nil) -> Any { false }
}
