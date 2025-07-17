// VectorCore: Safe Operations Example
//
// Demonstrates the safe parameter variants that return optionals or Results
// instead of crashing with precondition failures
//

import Foundation
import VectorCore

struct SafeOperationsExample {
    static func run() {
        print("VectorCore Safe Operations Examples")
        print("===================================\n")
        
        demonstrateSafeSubscript()
        demonstrateSafeInitialization()
        demonstrateSafeBasisVector()
        demonstrateSafeDivision()
        demonstrateSafePattern()
    }
    
    // MARK: - Safe Subscript Access
    
    static func demonstrateSafeSubscript() {
        print("1. Safe Subscript Access")
        print("------------------------")
        
        let vector = Vector<Dim128>.random(in: -1...1)
        
        // Traditional subscript (crashes on out of bounds)
        let value1 = vector[10]  // OK
        print("  Traditional subscript [10]: \(value1)")
        
        // Safe subscript access
        if let value2 = vector.at(10) {
            print("  Safe access at(10): \(value2)")
        }
        
        if let value3 = vector.at(200) {
            print("  Safe access at(200): \(value3)")
        } else {
            print("  Safe access at(200): nil (out of bounds)")
        }
        
        // Safe mutation
        var mutableVector = vector
        let success1 = mutableVector.setAt(10, to: 99.0)
        let success2 = mutableVector.setAt(200, to: 99.0)
        
        print("  Set at index 10: \(success1 ? "success" : "failed")")
        print("  Set at index 200: \(success2 ? "success" : "failed")")
        print()
    }
    
    // MARK: - Safe Initialization
    
    static func demonstrateSafeInitialization() {
        print("2. Safe Initialization")
        print("----------------------")
        
        let values128 = Array(repeating: 1.0, count: 128)
        let values256 = Array(repeating: 2.0, count: 256)
        
        // Traditional initialization (crashes on wrong dimension)
        let vector1 = Vector<Dim128>(values128)  // OK
        print("  Traditional init with correct dimension: OK")
        
        // Safe initialization
        if let vector2 = Vector<Dim128>(safe: values128) {
            print("  Safe init with correct dimension: OK")
        }
        
        if let vector3 = Vector<Dim128>(safe: values256) {
            print("  Safe init with wrong dimension: \(vector3)")
        } else {
            print("  Safe init with wrong dimension: nil (dimension mismatch)")
        }
        
        // Dynamic vector safe initialization
        if let dynamic1 = DynamicVector(safe: 128, values: values128) {
            print("  DynamicVector safe init: OK")
        }
        
        if let dynamic2 = DynamicVector(safe: 128, values: values256) {
            print("  DynamicVector safe init with mismatch: \(dynamic2)")
        } else {
            print("  DynamicVector safe init with mismatch: nil")
        }
        
        print()
    }
    
    // MARK: - Safe Basis Vector Creation
    
    static func demonstrateSafeBasisVector() {
        print("3. Safe Basis Vector Creation")
        print("-----------------------------")
        
        // Traditional basis vector (crashes on out of bounds)
        let basis1 = Vector<Dim64>.basis(at: 10)  // OK
        print("  Traditional basis(at: 10): OK")
        
        // Safe basis vector creation
        if let basis2 = Vector<Dim64>.basis(safe: 10) {
            print("  Safe basis(safe: 10): OK")
        }
        
        if let basis3 = Vector<Dim64>.basis(safe: 100) {
            print("  Safe basis(safe: 100): \(basis3)")
        } else {
            print("  Safe basis(safe: 100): nil (index out of bounds)")
        }
        
        // Using convenience initializer
        if let basis4 = Vector<Dim64>.basis(safe: 32) {
            print("  Safe basis(safe axis: 32): OK")
        }
        
        print()
    }
    
    // MARK: - Safe Division
    
    static func demonstrateSafeDivision() {
        print("4. Safe Division Operations")
        print("---------------------------")
        
        let numerator = Vector<Dim32>.random(in: 1...10)
        let denominator1 = Vector<Dim32>.random(in: 1...10)
        var denominator2 = Vector<Dim32>.random(in: 1...10)
        denominator2[10] = 0  // Insert a zero
        denominator2[20] = 0  // Insert another zero
        
        // Traditional division (no zero check)
        let result1 = numerator ./ denominator1  // OK
        print("  Traditional division: OK")
        
        // Safe division (throws on zero)
        do {
            let result2 = try Vector.safeDivide(numerator, by: denominator1)
            print("  Safe division (no zeros): OK")
        } catch {
            print("  Safe division error: \(error)")
        }
        
        do {
            let result3 = try Vector.safeDivide(numerator, by: denominator2)
            print("  Safe division (with zeros): \(result3)")
        } catch let error as VectorError {
            print("  Safe division (with zeros): \(error.kind.rawValue) - \(error.description)")
        } catch {
            print("  Safe division error: \(error)")
        }
        
        // Safe division with default value
        let result4 = Vector.safeDivide(numerator, by: denominator2, default: 0.0)
        print("  Safe division with default: OK (zeros replaced with 0.0)")
        
        print()
    }
    
    // MARK: - Safe Pattern Creation
    
    static func demonstrateSafePattern() {
        print("5. Safe Pattern Creation")
        print("------------------------")
        
        let pattern1 = [1.0, 2.0, 3.0]
        let pattern2: [Float] = []
        
        // Traditional pattern (crashes on empty)
        let vector1 = Vector<Dim128>.repeatingPattern(pattern1)  // OK
        print("  Traditional pattern with values: OK")
        
        // Safe pattern creation
        if let vector2 = Vector<Dim128>.repeatingPattern(safe: pattern1) {
            print("  Safe pattern with values: OK")
        }
        
        if let vector3 = Vector<Dim128>.repeatingPattern(safe: pattern2) {
            print("  Safe pattern with empty array: \(vector3)")
        } else {
            print("  Safe pattern with empty array: nil (empty pattern)")
        }
        
        print()
    }
}

// To run this example:
// SafeOperationsExample.run()