//
//  GPUProtocols.swift
//  VectorCore
//
//  GPU acceleration protocol definitions for v0.2.0
//

import Foundation

// MARK: - Core Types

/// Marker protocol for types that can be efficiently processed on GPU
public protocol GPUCompatible {
    /// Size in bytes for GPU buffer allocation
    static var gpuSize: Int { get }
    
    /// Alignment requirement for GPU memory
    static var gpuAlignment: Int { get }
}

/// Protocol for numeric types compatible with GPU operations
public protocol GPUNumeric: GPUCompatible, Numeric {
    /// Metal Shading Language type name
    static var metalTypeName: String { get }
}

// MARK: - Device Management

/// Represents a GPU compute device
public protocol GPUDevice {
    /// Unique identifier for the device
    var id: String { get }
    
    /// Human-readable name
    var name: String { get }
    
    /// Maximum memory available in bytes
    var maxMemoryAllocation: Int { get }
    
    /// Maximum thread group size
    var maxThreadsPerThreadgroup: Int { get }
    
    /// Whether the device supports unified memory (e.g., Apple Silicon)
    var hasUnifiedMemory: Bool { get }
    
    /// Check if a specific feature is supported
    func supports(feature: GPUFeature) -> Bool
}

/// GPU features that may vary by device
public enum GPUFeature {
    case float16
    case float64
    case int64
    case atomicOperations
    case sharedMemory
    case tensorCores
}

// MARK: - Memory Management

/// Type-safe GPU memory buffer
public protocol GPUBuffer {
    associatedtype Element: GPUCompatible
    
    /// Number of elements in the buffer
    var count: Int { get }
    
    /// Size in bytes
    var byteSize: Int { get }
    
    /// Associated GPU device
    var device: GPUDevice { get }
    
    /// Whether the buffer is currently accessible from CPU
    var isCPUAccessible: Bool { get }
    
    /// Synchronize GPU memory with CPU memory
    func synchronize() async throws
    
    /// Copy data from CPU to GPU
    func upload(from source: UnsafePointer<Element>, count: Int) async throws
    
    /// Copy data from GPU to CPU
    func download(to destination: UnsafeMutablePointer<Element>, count: Int) async throws
}

/// Manages GPU memory allocation and lifecycle
public protocol GPUMemoryManager {
    /// Current memory usage in bytes
    var currentUsage: Int { get }
    
    /// Maximum available memory
    var maxAvailable: Int { get }
    
    /// Allocate a new buffer
    func allocateBuffer<T: GPUCompatible>(
        for type: T.Type,
        count: Int,
        options: GPUBufferOptions
    ) throws -> any GPUBuffer<T>
    
    /// Release unused memory
    func purgeCache() async
}

/// Options for GPU buffer creation
public struct GPUBufferOptions: OptionSet {
    public let rawValue: Int
    
    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
    
    /// Buffer can be read by CPU
    public static let cpuReadable = GPUBufferOptions(rawValue: 1 << 0)
    
    /// Buffer can be written by CPU
    public static let cpuWritable = GPUBufferOptions(rawValue: 1 << 1)
    
    /// Buffer should use unified memory if available
    public static let preferUnifiedMemory = GPUBufferOptions(rawValue: 1 << 2)
    
    /// Buffer will be used frequently
    public static let persistent = GPUBufferOptions(rawValue: 1 << 3)
    
    /// Default options for general use
    public static let `default`: GPUBufferOptions = [.cpuReadable, .cpuWritable]
}

// MARK: - Operations

/// Represents a GPU computation
public protocol GPUOperation {
    /// Input buffer requirements
    var inputRequirements: [GPUBufferRequirement] { get }
    
    /// Output buffer requirements
    var outputRequirements: [GPUBufferRequirement] { get }
    
    /// Validate that buffers meet requirements
    func validate(inputs: [any GPUBuffer], outputs: [any GPUBuffer]) throws
    
    /// Estimated computation cost (arbitrary units)
    var computationalCost: Int { get }
    
    /// Whether this operation can be fused with others
    var isFusable: Bool { get }
}

/// Describes buffer requirements for an operation
public struct GPUBufferRequirement {
    public let elementType: any GPUCompatible.Type
    public let minimumCount: Int
    public let alignment: Int
    public let access: GPUBufferAccess
    
    public init(
        elementType: any GPUCompatible.Type,
        minimumCount: Int,
        alignment: Int = 1,
        access: GPUBufferAccess = .read
    ) {
        self.elementType = elementType
        self.minimumCount = minimumCount
        self.alignment = alignment
        self.access = access
    }
}

/// Buffer access patterns
public enum GPUBufferAccess {
    case read
    case write
    case readWrite
}

// MARK: - Execution

/// Manages GPU operation execution
public protocol GPUContext {
    /// Associated device
    var device: GPUDevice { get }
    
    /// Memory manager
    var memoryManager: GPUMemoryManager { get }
    
    /// Execute a single operation
    func execute<Op: GPUOperation>(
        _ operation: Op,
        inputs: [any GPUBuffer],
        outputs: [any GPUBuffer]
    ) async throws
    
    /// Execute a batch of operations
    func executeBatch(_ operations: [GPUExecutionNode]) async throws
    
    /// Wait for all pending operations to complete
    func synchronize() async throws
    
    /// Create a new execution batch
    func createBatch() -> GPUBatch
}

/// Node in GPU execution graph
public struct GPUExecutionNode {
    public let id: UUID
    public let operation: any GPUOperation
    public let inputs: [any GPUBuffer]
    public let outputs: [any GPUBuffer]
    public let dependencies: [UUID]
    
    public init(
        operation: any GPUOperation,
        inputs: [any GPUBuffer],
        outputs: [any GPUBuffer],
        dependencies: [UUID] = []
    ) {
        self.id = UUID()
        self.operation = operation
        self.inputs = inputs
        self.outputs = outputs
        self.dependencies = dependencies
    }
}

/// Batch of GPU operations for efficient execution
public protocol GPUBatch {
    /// Add an operation to the batch
    @discardableResult
    func add<Op: GPUOperation>(
        _ operation: Op,
        inputs: [any GPUBuffer],
        outputs: [any GPUBuffer],
        dependencies: [UUID]
    ) -> UUID
    
    /// Execute all operations in the batch
    func execute() async throws
    
    /// Cancel pending operations
    func cancel()
}

// MARK: - Acceleratable Operations

/// Protocol for types that support GPU acceleration
public protocol GPUAcceleratable {
    associatedtype GPURepresentation
    
    /// Convert to GPU-compatible representation
    func toGPU(context: GPUContext) async throws -> GPURepresentation
    
    /// Create from GPU representation
    static func fromGPU(_ gpu: GPURepresentation, context: GPUContext) async throws -> Self
}

/// Vector operations that can be GPU-accelerated
public protocol GPUVectorOperations {
    associatedtype Scalar: GPUNumeric
    
    /// Element-wise addition
    static func add(
        _ lhs: any GPUBuffer<Scalar>,
        _ rhs: any GPUBuffer<Scalar>,
        to result: any GPUBuffer<Scalar>,
        context: GPUContext
    ) async throws
    
    /// Element-wise multiplication
    static func multiply(
        _ lhs: any GPUBuffer<Scalar>,
        _ rhs: any GPUBuffer<Scalar>,
        to result: any GPUBuffer<Scalar>,
        context: GPUContext
    ) async throws
    
    /// Scalar multiplication
    static func scale(
        _ vector: any GPUBuffer<Scalar>,
        by scalar: Scalar,
        to result: any GPUBuffer<Scalar>,
        context: GPUContext
    ) async throws
    
    /// Dot product
    static func dot(
        _ lhs: any GPUBuffer<Scalar>,
        _ rhs: any GPUBuffer<Scalar>,
        context: GPUContext
    ) async throws -> Scalar
}

/// Matrix operations that can be GPU-accelerated
public protocol GPUMatrixOperations {
    associatedtype Scalar: GPUNumeric
    
    /// Matrix multiplication
    static func multiply(
        _ lhs: any GPUBuffer<Scalar>,
        lhsRows: Int, lhsCols: Int,
        _ rhs: any GPUBuffer<Scalar>,
        rhsRows: Int, rhsCols: Int,
        to result: any GPUBuffer<Scalar>,
        context: GPUContext
    ) async throws
    
    /// Matrix transpose
    static func transpose(
        _ matrix: any GPUBuffer<Scalar>,
        rows: Int, cols: Int,
        to result: any GPUBuffer<Scalar>,
        context: GPUContext
    ) async throws
}

// MARK: - Error Handling

/// Errors that can occur during GPU operations
public enum GPUError: Error {
    /// No compatible GPU device available
    case deviceNotAvailable
    
    /// Insufficient GPU memory for operation
    case insufficientMemory(required: Int, available: Int)
    
    /// Kernel compilation or loading failed
    case kernelCompilationFailed(String)
    
    /// Operation not supported on current device
    case operationNotSupported(String)
    
    /// Timeout waiting for GPU operation
    case synchronizationTimeout
    
    /// Invalid buffer configuration
    case invalidBuffer(String)
    
    /// Data type mismatch
    case typeMismatch(expected: String, actual: String)
}

// MARK: - Performance Monitoring

/// Protocol for GPU performance monitoring
public protocol GPUPerformanceMonitor {
    /// Start timing an operation
    func beginTiming(label: String) -> GPUTimingToken
    
    /// End timing and record result
    func endTiming(token: GPUTimingToken)
    
    /// Get performance statistics
    func statistics(for label: String) -> GPUPerformanceStats?
    
    /// Reset all statistics
    func reset()
}

/// Token for timing GPU operations
public struct GPUTimingToken {
    public let id: UUID
    public let label: String
    public let startTime: DispatchTime
}

/// Performance statistics for GPU operations
public struct GPUPerformanceStats {
    public let label: String
    public let count: Int
    public let totalTime: TimeInterval
    public let averageTime: TimeInterval
    public let minTime: TimeInterval
    public let maxTime: TimeInterval
    public let standardDeviation: TimeInterval
}

// MARK: - Default Implementations

extension GPUCompatible where Self: FixedWidthInteger {
    public static var gpuSize: Int { MemoryLayout<Self>.size }
    public static var gpuAlignment: Int { MemoryLayout<Self>.alignment }
}

extension GPUCompatible where Self: FloatingPoint {
    public static var gpuSize: Int { MemoryLayout<Self>.size }
    public static var gpuAlignment: Int { MemoryLayout<Self>.alignment }
}

extension Float32: GPUNumeric {
    public static var metalTypeName: String { "float" }
}

extension Float64: GPUNumeric {
    public static var metalTypeName: String { "double" }
}

extension Int32: GPUNumeric {
    public static var metalTypeName: String { "int" }
}

extension UInt32: GPUNumeric {
    public static var metalTypeName: String { "uint" }
}