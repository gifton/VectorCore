import XCTest
@testable import VectorCore

final class CKernelsSmokeTests: XCTestCase {
    func testDot512Stub() throws {
        #if VC_USE_C_KERNELS
        var a = [Float](repeating: 1, count: 512)
        var b = [Float](repeating: 2, count: 512)
        let r = a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                CKernels.dot512(ap.baseAddress!, bp.baseAddress!)
            }
        }
        XCTAssertEqual(r, 1024, accuracy: 1e-5)
        #else
        throw XCTSkip("C kernels are not compiled into this build (VC_USE_C_KERNELS is not defined)")
        #endif
    }
}

