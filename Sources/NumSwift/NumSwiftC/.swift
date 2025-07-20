//
//  NumSwiftCPerformanceManager.swift
//  NumSwift
//
//  Global performance management and optimization selection
//

import Foundation

public enum NumSwiftCPerformanceMode {
    case automatic    // Choose based on problem size
    case optimized    // Always use optimized implementations
    case original     // Use original implementations
}

/// Global performance manager for NumSwiftC operations
public class NumSwiftCPerformanceManager {
    
    /// Global performance mode setting
    public static var globalPerformanceMode: NumSwiftCPerformanceMode = .automatic
    
    /// Enable/disable performance monitoring
    public static var enablePerformanceMonitoring = false
    
    /// Performance thresholds for different operations
    public struct OptimizationThresholds {
        public static var matmulFloat32 = 64 * 64   // 4K elements
        public static var matmulFloat16 = 32 * 32   // 1K elements
        public static var convFloat32 = 32 * 32     // 1K elements  
        public static var convFloat16 = 16 * 16     // 256 elements
        public static var transposeFloat32 = 1024   // 1K elements
        public static var transposeFloat16 = 512    // 512 elements
        public static var elementWise = 256         // 256 elements
    }
    
    /// Configure performance mode globally
    public static func setGlobalPerformanceMode(_ mode: NumSwiftCPerformanceMode) {
        globalPerformanceMode = mode
        
        // Update individual optimizers
        NumSwiftCOptimized.performanceMode = mode
        NumSwiftCFloat16Optimized.performanceMode = mode
        
        if enablePerformanceMonitoring {
            print("🔧 NumSwiftC Global Performance Mode: \(mode)")
        }
    }
    
    /// Run comprehensive benchmarks across all data types
    public static func runComprehensiveBenchmark() {
        print("🚀 NumSwiftC Comprehensive Performance Benchmark")
        print("================================================")
        
        // Float32 benchmarks
        print("\n📊 Float32 Performance:")
        NumSwiftCOptimized.benchmarkSwift()
        
        #if arch(arm64)
        // Float16 benchmarks
        print("\n📊 Float16 Performance:")
        NumSwiftCFloat16Optimized.benchmark()
        #endif
        
        // Memory usage comparison
        print("\n💾 Memory Efficiency:")
        let matrixSize = 256
        let float32Memory = matrixSize * matrixSize * MemoryLayout<Float>.size * 3
        let float16Memory = matrixSize * matrixSize * MemoryLayout<Float16>.size * 3
        
        print("   Float32 memory (256x256x3): \(float32Memory / 1024)KB")
        print("   Float16 memory (256x256x3): \(float16Memory / 1024)KB")
        print("   Memory savings: \((float32Memory - float16Memory) / 1024)KB (\(Int(Double(float32Memory - float16Memory) / Double(float32Memory) * 100))%)")
        
        print("\n✅ Benchmark complete. Use NumSwiftCPerformanceManager.setGlobalPerformanceMode() to configure optimization.")
    }
    
    /// Auto-tune thresholds based on device performance
    public static func autoTuneThresholds() {
        print("🔧 Auto-tuning performance thresholds...")
        
        let testSizes = [64, 128, 256, 512]
        var optimalThresholds: [String: Int] = [:]
        
        for size in testSizes {
            let elementCount = size * size
            let a = (0..<size).map { _ in (0..<size).map { _ in Float.random(in: 0...1) } }
            let b = (0..<size).map { _ in (0..<size).map { _ in Float.random(in: 0...1) } }
            
            // Test original
            let startOriginal = CFAbsoluteTimeGetCurrent()
            NumSwiftCOptimized.performanceMode = .original
            _ = NumSwiftCOptimized.matmul(a, b: b, aSize: (size, size), bSize: (size, size))
            let originalTime = CFAbsoluteTimeGetCurrent() - startOriginal
            
            // Test optimized
            let startOptimized = CFAbsoluteTimeGetCurrent()
            NumSwiftCOptimized.performanceMode = .optimized
            _ = NumSwiftCOptimized.matmul(a, b: b, aSize: (size, size), bSize: (size, size))
            let optimizedTime = CFAbsoluteTimeGetCurrent() - startOptimized
            
            let speedup = originalTime / optimizedTime
            
            if speedup > 1.5 && optimalThresholds["matmul"] == nil {
                optimalThresholds["matmul"] = elementCount
                print("   Matmul optimization threshold: \(elementCount) elements (size \(size)x\(size))")
            }
        }
        
        // Apply auto-tuned thresholds
        if let matmulThreshold = optimalThresholds["matmul"] {
            OptimizationThresholds.matmulFloat32 = matmulThreshold
            OptimizationThresholds.matmulFloat16 = matmulThreshold / 2  // Float16 is more efficient
        }
        
        // Reset to automatic
        NumSwiftCOptimized.performanceMode = .automatic
        
        print("✅ Auto-tuning complete!")
    }
}

/// Convenience extensions for easy performance configuration
public extension NumSwiftCPerformanceManager {
    
    /// Quick performance configurations
    enum PerformanceProfile {
        case maxSpeed      // Always use optimized, lowest thresholds
        case balanced      // Automatic with default thresholds
        case maxCompatibility  // Always use original implementations
        case memoryOptimized   // Prefer Float16, optimized implementations
    }
    
    static func setProfile(_ profile: PerformanceProfile) {
        switch profile {
        case .maxSpeed:
            setGlobalPerformanceMode(.optimized)
            OptimizationThresholds.matmulFloat32 = 16 * 16
            OptimizationThresholds.matmulFloat16 = 8 * 8
            OptimizationThresholds.convFloat32 = 16 * 16
            OptimizationThresholds.convFloat16 = 8 * 8
            OptimizationThresholds.elementWise = 64
            
        case .balanced:
            setGlobalPerformanceMode(.automatic)
            // Use default thresholds
            
        case .maxCompatibility:
            setGlobalPerformanceMode(.original)
            
        case .memoryOptimized:
            setGlobalPerformanceMode(.optimized)
            OptimizationThresholds.matmulFloat16 = 16 * 16
            OptimizationThresholds.convFloat16 = 8 * 8
            print("💡 Consider using Float16 operations for 50% memory savings")
        }
        
        if enablePerformanceMonitoring {
            print("🎯 Applied performance profile: \(profile)")
        }
    }
}