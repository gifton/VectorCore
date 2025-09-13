// VectorCore: Thread-Safe Configuration
//
// Provides thread-safe configuration management for VectorCore
//

import Foundation

/// Thread-safe configuration container using actor isolation
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
