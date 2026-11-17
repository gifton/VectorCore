import CryptoKit
import Darwin
import UIKit

// The host reports its own executable digest and runtime. IOSQualificationTests
// rejects mismatched identities, missing bodies, and incomplete result frames.
@main
@MainActor
final class QualificationApp: UIResponder, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        true
    }

    func application(
        _ application: UIApplication, configurationForConnecting session: UISceneSession,
        options: UIScene.ConnectionOptions
    ) -> UISceneConfiguration {
        let configuration = UISceneConfiguration(name: "Qualification", sessionRole: session.role)
        configuration.delegateClass = QualificationScene.self
        return configuration
    }
}

@MainActor
final class QualificationScene: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options: UIScene.ConnectionOptions) {
        guard let scene = scene as? UIWindowScene else { return }
        let controller = UIViewController()
        controller.view.backgroundColor = .systemBackground
        let label = UILabel(frame: CGRect(x: 24, y: 100, width: 340, height: 100))
        label.numberOfLines = 0
        label.text = "VectorCore compiler qualification"
        controller.view.addSubview(label)
        let window = UIWindow(windowScene: scene)
        window.rootViewController = controller
        window.makeKeyAndVisible()
        self.window = window
        DispatchQueue.main.async {
            do {
                try self.collect()
                label.text = "Qualification records saved.\nReview required."
            } catch {
                label.text = "Qualification incomplete.\n\(error)"
                fputs("VC_IOS_HOST_ERROR \(error)\n", stderr)
            }
        }
    }

    private func argument(_ flag: String) throws -> String {
        let arguments = ProcessInfo.processInfo.arguments
        guard arguments.filter({ $0 == flag }).count == 1,
              let index = arguments.firstIndex(of: flag), index + 1 < arguments.count
        else { throw HostError.arguments }
        return arguments[index + 1]
    }

    private func record(_ prefix: String, _ fields: [String: Any]) throws {
        let bytes = try JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys])
        guard let text = String(data: bytes, encoding: .utf8) else { throw HostError.encoding }
        print(prefix + " " + text)
        fflush(stdout)
    }

    private func collect() throws {
        let invocation = try argument("--invocation")
        let caseID = try argument("--case")
        guard UUID(uuidString: invocation)?.uuidString.lowercased() == invocation,
              !caseID.isEmpty,
              let executable = Bundle.main.executableURL
        else { throw HostError.arguments }
        let documents = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let result = documents.appendingPathComponent(invocation + ".records")
        // O_EXCL makes a reused invocation fail even if an old app container remains.
        let descriptor = open(result.path, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw HostError.output }
        defer { close(descriptor) }
        guard dup2(descriptor, STDOUT_FILENO) >= 0 else { throw HostError.output }
        let hash = SHA256.hash(data: try Data(contentsOf: executable))
            .map { String(format: "%02x", $0) }.joined()
        #if targetEnvironment(simulator)
        let platform = "ios-simulator"
        #else
        let platform = "ios-device"
        #endif
        #if !arch(arm64)
        #error("This qualification host requires arm64")
        #endif
        try record("VC_IOS_BEGIN", [
            "invocationId": invocation, "caseId": caseID, "binarySHA256": hash,
            "runtime": ["version": UIDevice.current.systemVersion,
                        "description": ProcessInfo.processInfo.operatingSystemVersionString],
            "platform": platform, "architecture": "arm64"
        ])
        let numericalExit: Int32 = HostedProbe.run()
        try record("VC_IOS_END", [
            "invocationId": invocation, "caseId": caseID, "numericalExit": numericalExit
        ])
        fsync(descriptor)
    }

    private enum HostError: Error { case arguments, encoding, output }
}
