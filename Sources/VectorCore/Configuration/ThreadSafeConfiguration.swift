// VectorCore: ThreadSafeConfiguration
//
// An actor that holds one configuration value for VectorCore
//

import Foundation

/// Configuration container whose value is actor-isolated: every read and write is an actor method
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

    /// Mutates and atomically publishes the configuration binding after the body succeeds.
    /// The synchronous body runs within this actor without a suspension point.
    /// If it throws, the original binding remains and the same error propagates.
    ///
    /// Assignment to a candidate does not deeply copy mutable reference members.
    /// Rollback of aliased state requires caller-supplied value or snapshot semantics;
    /// external side effects performed by the body cannot be undone here.
    public func modify(_ body: @Sendable (inout T) throws -> Void) rethrows {
        var candidate = value
        try body(&candidate)
        value = candidate
    }

    /// Update a specific property of the configuration
    public func update<V>(_ keyPath: WritableKeyPath<T, V>, to newValue: V) {
        value[keyPath: keyPath] = newValue
    }
}
