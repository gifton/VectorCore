# Contributing to VectorCore

Thank you for your interest in contributing to VectorCore! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/VectorCore.git`
3. Add the upstream repository: `git remote add upstream https://github.com/ORIGINAL_OWNER/VectorCore.git`
4. Create a new branch: `git checkout -b feature/your-feature-name`

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in the [Issues](https://github.com/OWNER/VectorCore/issues)
- If not, create a new issue with:
  - A clear, descriptive title
  - Steps to reproduce the issue
  - Expected behavior
  - Actual behavior
  - System information (OS, Swift version, etc.)
  - Code samples if applicable

### Suggesting Features

- Check if the feature has already been suggested
- Create a new issue with:
  - A clear description of the feature
  - Use cases and benefits
  - Potential implementation approach
  - Any relevant examples

### Submitting Code

1. Ensure your code follows the project's coding standards
2. Add tests for any new functionality
3. Update documentation as needed
4. Ensure all tests pass
5. Submit a pull request

## Development Setup

### Requirements

- Swift 5.9 or later
- Xcode 15.0 or later (for macOS development)
- macOS 13.0+ (for Accelerate framework)

### Building the Project

```bash
swift build
```

### Running Tests

```bash
swift test
```

Or use the included test script for a summary:

```bash
./test_summary.sh
```

### Running Benchmarks

```bash
swift run -c release VectorCoreBenchmarks
```

## Coding Standards

### Swift Style Guide

- Follow the [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/)
- Use 4 spaces for indentation (no tabs)
- Keep lines under 120 characters when possible
- Use meaningful variable and function names
- Document all public APIs

### Code Organization

- Group related functionality together
- Use `MARK:` comments to organize code sections
- Keep files focused on a single responsibility
- Use extensions to organize protocol conformances

### Performance Considerations

- Use `@inlinable` and `@inline(__always)` judiciously for performance-critical paths
- Prefer value types with COW semantics
- Document complexity with Big O notation
- Include benchmarks for performance-critical code

### Example Code Style

```swift
/// Computes the dot product of two vectors.
/// 
/// - Parameters:
///   - other: The vector to compute dot product with
/// - Returns: The dot product result
/// - Complexity: O(n) where n is the vector dimension
@inlinable
public func dot(_ other: Vector<D>) -> Float {
    var result: Float = 0
    withUnsafeBufferPointer { selfBuffer in
        other.withUnsafeBufferPointer { otherBuffer in
            vDSP_dotpr(selfBuffer.baseAddress!, 1,
                      otherBuffer.baseAddress!, 1,
                      &result, vDSP_Length(D.value))
        }
    }
    return result
}
```

## Testing

### Test Requirements

- All new features must include tests
- Maintain or improve code coverage
- Test edge cases and error conditions
- Include performance tests for critical paths

### Test Organization

- Place unit tests in `Tests/VectorCoreTests/`
- Name test files as `<Feature>Tests.swift`
- Use descriptive test method names: `test<Scenario>_<ExpectedResult>()`

### Example Test

```swift
func testDotProduct_orthogonalVectors_returnsZero() {
    let v1 = Vector<Dim3>([1, 0, 0])
    let v2 = Vector<Dim3>([0, 1, 0])
    
    let result = v1.dot(v2)
    
    XCTAssertEqual(result, 0, accuracy: 1e-6)
}
```

## Documentation

### API Documentation

- Document all public types, methods, and properties
- Use Swift's documentation comments (`///`)
- Include parameter descriptions, return values, and complexity
- Add code examples for complex APIs

### README Updates

- Update the README when adding significant features
- Keep examples current and working
- Maintain the feature list

### Additional Documentation

- Update relevant files in `/Documentation/`
- Add performance characteristics for new operations
- Update migration guides if introducing breaking changes

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Run the linter/formatter if available
   - Update documentation
   - Rebase on the latest main branch

2. **PR Title and Description**
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes were made and why
   - Include any breaking changes
   - Add screenshots for UI changes

3. **Review Process**
   - Address reviewer feedback promptly
   - Keep the PR focused on a single concern
   - Be open to suggestions and alternative approaches

4. **After Approval**
   - Squash commits if requested
   - Ensure the PR is up to date with main
   - Delete your branch after merging

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Check existing documentation
- Review previous PRs for examples

Thank you for contributing to VectorCore!