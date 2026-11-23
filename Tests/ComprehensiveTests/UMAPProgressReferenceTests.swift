import Foundation
import Testing
import VectorCore

@Suite
struct UMAPProgressReferenceTests {
    // Reference v0.4.0-alpha.1, 44c9ffaf145a59688f420023b28cdcf87c158070;
    // captured before progress edits using Xcode 27.2 (27B52), Swift 6.4.2,
    // arm64 macOS 27.0.1 (26A434), default LAPACK provider. Debug/Release
    // controls are compared before these words are accepted. Cross-build words
    // are opt-in; handlerPreservesCoordinates always checks within-build words.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["VECTORCORE_UMAP_CHECK_BASELINE_BITS"] == "1"),
          arguments: UMAPProgressFixtures.Route.allCases, [2, 5])
    func frozenReference(route: UMAPProgressFixtures.Route, dimensions: Int) throws {
        let prefix: String
        switch route {
        case .builtPCA, .suppliedPCA: prefix = "pca"
        case .graphProvided: prefix = "provided"
        default: prefix = "random"
        }
        let expected = try #require(Self.words["\(prefix)-\(dimensions)"])
        let result = try UMAPProgressFixtures.run(route, dimensions: dimensions)
        #expect(result.coordinates.map(\.bitPattern) == expected)
    }

    private static let words: [String: [UInt32]] = [
        "pca-2": [
            0xc0d7cd54, 0x3f42805f, 0x40a64659, 0xbfcf8be8, 0x4084c46e,
            0xc01df3f4, 0x3f42f31e, 0x40a81d00, 0xbfd7d218, 0xbfa624d7,
            0xbf8ef853, 0xbf23d650, 0xc0b51244, 0x3d91afcc, 0x40d48b4e,
            0x4032cc43,
        ],
        "pca-5": [
            0xc106e43c, 0x3fe6e6ef, 0x400b8205, 0xbf2c05ce, 0xbf0e93b0,
            0x40bc809a, 0xbf72c747, 0x4038edb6, 0xbf1a6a8b, 0x3ec80ef6,
            0x40873e7e, 0xc07bdc34, 0x3f91bc20, 0xbfbd4163, 0xbfaf9a6e,
            0x3f2358fa, 0x40b34b36, 0xc034eba9, 0xbe0c2991, 0xbf786268,
            0xbfe6ca0b, 0xc086139f, 0xc07581e5, 0x3fd8e266, 0x3db7c666,
            0xbfd5b0ab, 0xbfc6e57a, 0xc060610d, 0xbf873093, 0x3fd6e45c,
            0xc0c078f2, 0xbdc8edb4, 0x406ce196, 0x3fadeea6, 0x3f16d81a,
            0x40e57fd9, 0x4054cefd, 0x3e857ccc, 0x3f60cc78, 0x3e2c98af,
        ],
        "provided-2": [
            0x407ac63f, 0x406951df, 0xbf5a96e3, 0xbf903431, 0x3efde8fd,
            0x3e648b94, 0xbe5d42b4, 0xbefa43f6, 0x404d486b, 0x403bd40c,
            0x3f94e991, 0x3f640187, 0x40923cbf, 0x4089828e, 0x3fa1071d,
            0x3f7c3cc9,
        ],
        "provided-5": [
            0x40646430, 0x4052b7dd, 0x40414eec, 0x402f9b49, 0x401e307a,
            0x3fd635cf, 0x3fb34a38, 0x3f906216, 0x3f5aec19, 0x3f151bae,
            0x3fe5dcaa, 0x3fc30436, 0x3fa01940, 0x3f7a85f3, 0x3f34af0f,
            0x3f1cca75, 0x3ead7bc5, 0x3d87bc1c, 0xbe547bc4, 0xbef5cafb,
            0x3f0cc239, 0x3e8dd8e8, 0x3b8de1b8, 0xbe8974fe, 0xbf0a8b3c,
            0x401d8fe2, 0x400c05ab, 0x3ff52956, 0x3fd20fb9, 0x3faf2e29,
            0x40308dbc, 0x401f0956, 0x400d9817, 0x3ff82349, 0x3fd53fff,
            0x3f9cbee7, 0x3f73a0cc, 0x3f2dd1d1, 0x3ecfe54b, 0x3e088db5,
        ],
        "random-2": [
            0x4094b733, 0xc0c52b3a, 0xc004aea0, 0xc0e4b0d2, 0x40b43ad3,
            0x40e6f162, 0x4023f329, 0x403d468d, 0xc05b51ac, 0x3fbc54fb,
            0x409225b4, 0x40ada9ce, 0x40b3a0df, 0x40b453bc, 0x404d12a4,
            0x40573957,
        ],
        "random-5": [
            0x408f8cf5, 0xc0e4b2fa, 0xc023b8b5, 0xc0f69ffa, 0x40a46ce4,
            0x410a35d1, 0x3e428030, 0x3fd5c29e, 0xc08a4bd2, 0x3f84a130,
            0x40f8c06d, 0x40fa32bb, 0x40bbe0b5, 0x40d2c3a2, 0x40a7eace,
            0x40855c45, 0x3f782a8a, 0x3fa85d5a, 0xbf9644e6, 0x3ffdcc9c,
            0x3fa2d749, 0xc037abbd, 0x3f46e5d1, 0xc106c201, 0x3face937,
            0xc0001ad3, 0xc0d1e993, 0xbfaa7041, 0x405cd0c3, 0xc0dfd81e,
            0xbfab1404, 0x407ca3f4, 0xbf94643f, 0x408ac440, 0xc0d0fe2e,
            0xbf71988e, 0x40bd4b90, 0x40596aff, 0xbed8aacf, 0x4100f9e9,
        ],
    ]
}
