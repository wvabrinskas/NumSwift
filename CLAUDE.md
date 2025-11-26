# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NumSwift is a Swift package that adds complex arithmetic to Swift with support for array arithmetic, utilizing Apple's Accelerate framework when possible. The library provides efficient matrix operations, convolutions, and mathematical computations across Float, Double, and Float16 data types.

## Development Commands

### Build and Test
- **Build**: `swift build -v`
- **Run tests**: `swift test -v`
- **Single test**: Use `swift test --filter <TestName>` for specific test cases

### Package Management
This is a Swift Package Manager (SPM) project. The main configuration is in `Package.swift`.

## Architecture

### Core Structure
- **NumSwift**: Main Swift module providing high-level APIs and operator overloads
- **NumSwiftC**: C library for performance-critical operations (convolutions, matrix multiplication)
- **NumSwiftMetal**: GPU-accelerated operations using Metal framework

### Key Components

#### 1. Multi-Target Architecture
- `Sources/NumSwift/`: Swift implementation with protocol extensions and operator overloads
- `Sources/NumSwiftC/`: C implementations for performance-critical functions
- `Sources/NumSwift/NumSwiftC/`: Swift wrappers for C functions
- `Tests/NumSwiftTests/`: Test suite including benchmarks

#### 2. Performance Layers
- **Apple Accelerate**: Used for optimized mathematical operations when available
- **C Backend**: Custom C implementations in `numswiftc.c` for convolutions, matrix operations
- **Metal GPU**: GPU acceleration for matrix multiplication using Metal Performance Shaders
- **Float16 Support**: ARM64-specific half-precision floating point optimizations

#### 3. Core Functionality
- Array arithmetic operators (`+`, `-`, `*`, `/`) for element-wise and broadcast operations
- Matrix operations: multiplication, transpose, dot products
- Convolution operations: 1D/2D convolutions, transposed convolutions
- Utility functions: shape analysis, padding, scaling, normalization

### Data Type Support
- `Float` (32-bit)
- `Double` (64-bit) 
- `Float16` (16-bit, ARM64 only)

### Platform Support
- iOS 14+, macOS 11+, tvOS 14+, watchOS 7+
- ARM64 architecture required for Float16 operations

## Key Files

### Core Implementation
- `Sources/NumSwift/NumSwift.swift`: Main class with convolution and utility functions
- `Sources/NumSwift/BaseNumeric.swift`: Protocol extensions for array operations
- `Sources/NumSwift/Extensions.swift`: Collection extensions and helper functions
- `Sources/NumSwiftC/numswiftc.c`: C implementations for performance operations

### Metal GPU Implementation
- `Sources/NumSwift/NumSwiftMetal/NumSwiftMetal.swift`: Metal backend wrapper with automatic CPU/GPU selection
- `Sources/NumSwift/Resources/NumSwiftMetal.metal`: Compute shaders for all GPU operations
- `Sources/NumSwift/NumSwiftC/NumSwiftC.swift`: Swift wrappers for C functions with Metal integration

### Type-Specific Files
- `Sources/NumSwift/Double.swift`: Double-specific extensions
- `Sources/NumSwift/Float32.swift`: Float32-specific extensions  
- `Sources/NumSwift/Float16.swift`: Float16-specific extensions (ARM64 only)

### Testing
- `Tests/NumSwiftTests/NumSwiftTests.swift`: Main test suite
- `Tests/NumSwiftTests/Benchmarks.swift`: Performance benchmarks comparing CPU and GPU implementations
- `Tests/NumSwiftTests/NumSwiftTestsFloat16.swift`: Float16-specific tests
- `Tests/NumSwiftTests/NumSwiftMetalTests.swift`: Metal GPU backend tests and validation

## Development Notes

### C Integration
The project uses a hybrid Swift/C approach where performance-critical operations are implemented in C (`NumSwiftC` target) and wrapped with Swift APIs. When modifying C code, ensure header declarations in `include/numswiftc.h` match implementations.

### Metal GPU Operations
NumSwift now includes a comprehensive Metal GPU backend through the `NumSwiftMetal` class and compute shaders. The implementation provides:

- **Automatic Backend Selection**: Intelligently chooses between CPU and GPU based on problem size
- **Seamless API**: Same interface regardless of backend (CPU/Metal)
- **Fallback Support**: Automatically falls back to CPU if Metal initialization fails
- **Comprehensive Operations**: Full suite of array, matrix, and convolution operations
- **Performance Optimized**: Uses appropriate parallelization strategies for each operation type

Key Metal features:
- Custom compute shaders in `NumSwiftMetal.metal` for all core operations
- Swift wrapper class `NumSwiftMetal.swift` for seamless CPU/GPU switching
- Support for both Float32 and Float16 (half-precision) operations
- Optimized thread dispatch strategies and memory management
- Async operation support for better GPU utilization

### Float16 Considerations
Float16 operations are only available on ARM64 architectures. Code using Float16 should be wrapped in `#if arch(arm64)` preprocessor conditions.

### Performance Strategy
The library prioritizes performance through a multi-tier approach:
1. **Apple Accelerate framework** for standard mathematical operations
2. **Custom C implementations** for specialized operations not covered by Accelerate
3. **Metal GPU acceleration** for large-scale computations that benefit from parallelization
4. **Automatic backend selection** that chooses the optimal implementation based on problem size
5. **Concurrent processing utilities** in array extensions for CPU-bound operations

The Metal backend includes sophisticated optimizations:
- Parallel reduction algorithms for sum, max, min operations
- Tiled matrix multiplication with shared memory optimization
- Optimized convolution kernels with im2col transformations
- Smart thread group sizing based on hardware capabilities
- Memory pooling and buffer reuse for reduced allocation overhead

### Testing Strategy
- Unit tests cover all major mathematical operations
- Benchmarks compare performance across different implementations
- Tests include edge cases for padding, stride operations, and boundary conditions