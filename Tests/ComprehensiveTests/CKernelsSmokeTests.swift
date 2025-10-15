import XCTest
@testable import VectorCore

final class CKernelsSmokeTests: XCTestCase {
    func testDot512Stub() {
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
        // If C kernels are disabled, nothing to test here.
        XCTAssertTrue(true)
        #endif
    }
}

