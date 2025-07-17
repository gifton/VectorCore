// VectorCore: Thread-Safe Configuration
//
// Provides thread-safe configuration management for VectorCore
//

import Foundation

/// Thread-safe configuration container using actor isolation
@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
public actor ThreadSafeConfiguration<T: Sendable> {
    private var value: T
    
    public init(_ initialValue: T) {
        self.value = initialValue
    }
    
    /// Get the current configuration value
    public func get() -> T {
        value
    }
    
    /// Update the configuration value
    public func update(_ newValue: T) {
        value = newValue
    }
    
    /// Update a specific property of the configuration
    public func update<V>(_ keyPath: WritableKeyPath<T, V>, to newValue: V) {
        value[keyPath: keyPath] = newValue
    }
}

/// Pre-iOS 16 thread-safe configuration using NSLock
public final class LegacyThreadSafeConfiguration<T>: @unchecked Sendable {
    private var value: T
    private let lock = NSLock()
    
    public init(_ initialValue: T) {
        self.value = initialValue
    }
    
    /// Get the current configuration value
    public func get() -> T {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
    
    /// Update the configuration value
    public func update(_ newValue: T) {
        lock.lock()
        defer { lock.unlock() }
        value = newValue
    }
    
    /// Update a specific property of the configuration
    public func update<V>(_ keyPath: WritableKeyPath<T, V>, to newValue: V) {
        lock.lock()
        defer { lock.unlock() }
        value[keyPath: keyPath] = newValue
    }
}

/// Protocol for types that can provide thread-safe configuration
public protocol ConfigurationProvider {
    associatedtype ConfigType
    
    /// Get the current configuration synchronously (for legacy support)
    var currentConfiguration: ConfigType { get }
    
    /// Get the configuration asynchronously (preferred)
    @available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
    func configuration() async -> ConfigType
}