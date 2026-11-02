import Foundation
import Darwin
import Testing

// T10: pins the contract of ChildProbeLauncher.swift (see its header). The end-to-end tests launch
// this binary's `launcherSelfTestChild` probe. That probe is the only place in Tests/ that creates a
// process without the launcher: it spawns /bin/sleep as a member of the child's process group, which
// the group cleanup must remove (test_child_probe_launch_policy.py pins exactly that one spawn here).

private let selfTestFilter = "ChildProbeLauncherTests/launcherSelfTestChild"
private let selfTestScenarioKey = "T10_LAUNCHER_SELF_TEST"

private func selfTestMarker(_ line: String) {
    FileHandle.standardOutput.write(Data((line + "\n").utf8))
}

@Suite struct ChildProbeLauncherTests {
    // MARK: selection (clause 1)

    private static let helper = "/toolchain/usr/libexec/swift/pm/swiftpm-testing-helper"

    @Test func selectionIsReplacedNotEdited() throws {
        let parent = ["--test-bundle-path", "/b/T", "--skip-build", "--filter", "A/one", "--skip", "S", "--filter=B/two",
                      "--skip=C", "--specifier", "Old.X/y", "-s", "Z", "--xunit-output", "r.xml",
                      "--event-stream-output-path", "/p", "--event-stream-version", "0",
                      "--experimental-event-stream-output", "/q", "--experimental-event-stream-version", "1",
                      "--xunit-output=r2.xml", "/b/T", "--testing-library", "swift-testing", "--filter", "C/three"]
        let child = try childProbeArguments(for: .swiftTesting(filter: "Probe/child"), executable: Self.helper,
                                            parentArguments: parent)
        #expect(child == ["--test-bundle-path", "/b/T", "--skip-build", "/b/T", "--testing-library", "swift-testing",
                          "--filter", "Probe/child"])
        #expect(child.filter { $0 == "--filter" }.count == 1)
    }

    @Test func onlySelectionAndSinksAreRemoved() throws {
        // Look-alikes stay: `--skip-build` is not `--skip`, `--filtering` is not `--filter`, and a
        // value that merely contains `=` is not a flag.
        let parent = ["--test-bundle-path", "/b/T", "--skip-build", "--filtering", "x", "--parallel", "--repetitions", "2",
                      "--scratch-path", "/s/a=b", "--jobs", "6", "/b/T", "--testing-library", "swift-testing"]
        let child = try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: Self.helper, parentArguments: parent)
        #expect(child == parent + ["--filter", "P/c"])
        // A dropped flag consumes its value even when that value looks like another flag.
        let tricky = ["--test-bundle-path", "/b/T", "--filter", "--skip", "--skip", "--filter", "/b/T"]
        #expect(try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: Self.helper, parentArguments: tricky)
                == ["--test-bundle-path", "/b/T", "/b/T", "--filter", "P/c"])
        // The value of a -X pass-through flag is kept: `-Xlinker -s` keeps its `-s` (SwiftPM's alias,
        // which swift-testing does not parse) ...
        let passThrough = ["--test-bundle-path", "/b/T", "-Xlinker", "-s", "-Xswiftc", "-Xfrontend", "-Xlinker", "-rpath", "/b/T"]
        #expect(try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: Self.helper, parentArguments: passThrough)
                == passThrough + ["--filter", "P/c"])
        // ... unless swift-testing would parse it (it ignores -X pairing): then the launch is refused.
        let parsedBySwiftTesting = ["--filter", "--skip", "--filter=A/b", "--xunit-output", "--configuration-path",
                                    "--experimental-configuration-path=/c"]
        for value in parsedBySwiftTesting {
            let parent = ["--test-bundle-path", "/b/T", "-Xswiftc", value, "x", "/b/T"]
            do {
                _ = try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: Self.helper, parentArguments: parent)
                Issue.record("-X value \(value) was kept")
            } catch ChildProbeLaunchError.refusedRunnerArgument(let message) {
                #expect(message.contains(value), "\(message)")
            }
        }
    }

    /// Runner shapes from T10 fix1 evidence (m8-*): the same probe package under Command Line Tools
    /// Swift 6.2 (Testing 1085) and Xcode 27 Swift 6.4 (Testing 2084), both run with two filters, a
    /// skip and an xunit path. Paths are shortened; token order is as recorded.
    @Test func runnerShapesAreRecognizedOrRefused() throws {
        let clt62 = ["--test-bundle-path", "/s/debug/ArgvProbePackageTests.xctest/Contents/MacOS/ArgvProbePackageTests",
                     "--jobs", "6", "--scratch-path", "/s", "-Xswiftc", "-F", "-Xswiftc", "/CLT/Frameworks",
                     "-Xlinker", "-rpath", "-Xlinker", "/CLT/Frameworks", "--filter", "ProbeTests/recordArgv",
                     "--filter=ProbeTests/otherTest", "--skip", "NoSuchTest", "--xunit-output", "/s/x.xml",
                     "/s/debug/ArgvProbePackageTests.xctest/Contents/MacOS/ArgvProbePackageTests",
                     "--testing-library", "swift-testing"]
        let xcode64 = ["--test-bundle-path", "/s/Debug/ProbeTests.xctest/Contents/MacOS/ProbeTests", "--jobs", "6",
                       "--scratch-path", "/s", "--filter", "ProbeTests/recordArgv", "--filter=ProbeTests/otherTest",
                       "--skip", "NoSuchTest", "--xunit-output", "/s/x-swift-testing.xml",
                       "/s/Debug/ProbeTests.xctest/Contents/MacOS/ProbeTests", "--testing-library", "swift-testing"]
        for parent in [clt62, xcode64] {
            #expect(childProbeRunnerShape(executable: Self.helper, arguments: parent) == .swiftPMTestingHelper)
            let child = try childProbeArguments(for: .swiftTesting(filter: "ProbeTests/recordArgv"), executable: Self.helper,
                                                parentArguments: parent)
            #expect(child.filter { $0 == "--filter" || $0.hasPrefix("--filter=") } == ["--filter"])
            #expect(child.last == "ProbeTests/recordArgv")
            #expect(!child.contains { $0 == "--skip" || $0.hasPrefix("--xunit-output") })
            #expect(Array(child.suffix(4)) == ["--testing-library", "swift-testing", "--filter", "ProbeTests/recordArgv"])
        }
        #expect(try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: Self.helper, parentArguments: clt62)
                .contains("/CLT/Frameworks"))
        // Direct test executable (accepted, not observed on a local toolchain).
        let direct = ["--testing-library", "swift-testing", "--filter", "A/b", "--parallel"]
        #expect(childProbeRunnerShape(executable: "/s/PkgPackageTests.xctest", arguments: direct) == .directTestExecutable)
        #expect(try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: "/s/PkgPackageTests.xctest",
                                        parentArguments: direct) == ["--testing-library", "swift-testing", "--parallel", "--filter", "P/c"])
        // Unrecognized: swift-testing hosted by xctest, a helper without its bundle, or no swift-testing marker.
        let unknown: [(String, [String])] = [("/usr/bin/xctest", ["-XCTest", "A/b", "/b.xctest", "--testing-library", "swift-testing"]),
                                              (Self.helper, ["--testing-library", "swift-testing"]),
                                              (Self.helper, ["--test-bundle-path"]),
                                              ("/s/PkgPackageTests.xctest", ["--filter", "A/b"])]
        for (executable, arguments) in unknown {
            #expect(childProbeRunnerShape(executable: executable, arguments: arguments) == nil, "\(executable) \(arguments)")
            do {
                _ = try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: executable, parentArguments: arguments)
                Issue.record("unrecognized runner accepted: \(executable) \(arguments)")
            } catch ChildProbeLaunchError.unrecognizedRunner(let message) {
                #expect(message.hasPrefix("CHILD_PROBE_UNRECOGNIZED_RUNNER") && message.contains("executable=\(executable)")
                        && message.contains("arguments=\(arguments)"), "\(message)")
            }
        }
    }

    @Test func configurationFileIsRefused() {
        for flag in ["--configuration-path", "--experimental-configuration-path"] {
            #expect(throws: ChildProbeLaunchError.self) {
                _ = try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: Self.helper,
                                            parentArguments: ["--test-bundle-path", "/b/T", flag, "/c.json"])
            }
            #expect(throws: ChildProbeLaunchError.self) {
                _ = try childProbeArguments(for: .swiftTesting(filter: "P/c"), executable: Self.helper,
                                            parentArguments: ["--test-bundle-path", "/b/T", flag + "=/c.json"])
            }
        }
    }

    @Test func selectorNamesExactlyOneTest() throws {
        for good in ["A/b", "Suite_1/test_2", "M.C/testProbe", "ComprehensiveTests.ExecutionBoundaryTests/testChildProbe"] {
            try validateSingleTestSelection(good)
        }
        for bad in ["", "A", "/b", "A/", "A/b/c", ".*", "A/.*", "A/b|C/d", "A/b,C/d", "A/b()", "A/b c", "A.(B)/c",
                    "M..C/t", "1A/b", "A/b$", "Ä/b"] {
            #expect(throws: ChildProbeLaunchError.self, "\(bad)") { try validateSingleTestSelection(bad) }
            #expect(throws: ChildProbeLaunchError.self, "\(bad)") {
                _ = try childProbeArguments(for: .xctest(selector: bad, bundlePath: "/b.xctest"), executable: "/usr/bin/xctest",
                                            parentArguments: [])
            }
        }
        #expect(throws: ChildProbeLaunchError.self) {
            _ = try childProbeArguments(for: .swiftTesting(filter: ".*"), executable: Self.helper,
                                        parentArguments: ["--test-bundle-path", "/b/T"])
        }
    }

    @Test func xctestSelectionIsFresh() throws {
        let child = try childProbeArguments(for: .xctest(selector: "M.C/testProbe", bundlePath: "/b.xctest"),
                                            executable: "/usr/bin/xctest",
                                            parentArguments: ["-XCTest", "M.C/testA,M.C/testB", "/b.xctest", "--filter", "x"])
        #expect(child == ["-XCTest", "M.C/testProbe", "/b.xctest"])
    }

    // MARK: results (clause 5)

    @Test func unreapedChildReadsFailWithoutTrapping() {
        let unreaped = ChildProbeRun(executable: "/x", arguments: [], processIdentifier: 4242, processGroup: 4242,
                                     timedOut: true, cleaned: false, termination: nil)
        #expect(unreaped.terminationStatus == nil && unreaped.terminationReason == nil && unreaped.isRunning)
        #expect((unreaped.terminationStatus == 0) == false)
        #expect((unreaped.terminationReason == .uncaughtSignal) == false)
        #expect(unreaped.statusDescription == "unreaped" && unreaped.reasonDescription == "unreaped")
        #expect(throws: ChildProbeLaunchError.self) { _ = try unreaped.reaped() }
        let exited = ChildProbeRun(executable: "/x", arguments: [], processIdentifier: 4242, processGroup: 4242,
                                   timedOut: false, cleaned: true, termination: ChildProbeTermination(status: 0, reason: .exit))
        #expect(exited.terminationStatus == 0 && exited.terminationReason == .exit && !exited.isRunning)
        #expect(exited.statusDescription == "0" && exited.reasonDescription == "\(Process.TerminationReason.exit.rawValue)")
        #expect((try? exited.reaped()) == ChildProbeTermination(status: 0, reason: .exit))
    }

    // MARK: marker (clause 2)

    @Test func nestedLaunchIsRefused() throws {
        #expect(throws: ChildProbeLaunchError.self) {
            try refuseNestedChildProbeLaunch(.swiftTesting(filter: "P/c"), environment: [childProbeMarkerKey: "Outer/probe"])
        }
        try refuseNestedChildProbeLaunch(.swiftTesting(filter: "P/c"), environment: ["PATH": "/usr/bin"])
    }

    @Test func childProbeCannotLaunchAProbe() throws {
        let (run, log) = try launchSelfTest("nested", deadline: 15, cleanup: .kill(seconds: 2))
        try #require(!run.timedOut, "\(log)")
        #expect(run.terminationStatus == 0, "\(log)")
        #expect(log.contains("T10_SELF_TEST_MARKER \(selfTestFilter)"), "\(log)")
        #expect(log.contains("CHILD_PROBE_NESTED_LAUNCH_REFUSED"), "\(log)")
        #expect(log.contains("T10_NESTED_REFUSED"), "\(log)")
        #expect(!log.contains("T10_NESTED_NOT_REFUSED"), "\(log)")
        #expect(log.contains("T10_NESTED_SPAWNED_NOTHING"), "refusal must precede the spawn: \(log)")
    }

    // MARK: process group (clause 3)

    @Test func exitedChildLeavesNoGroupMember() throws {
        let (run, log) = try launchSelfTest("leave", deadline: 15, cleanup: .kill(seconds: 2))
        try #require(!run.timedOut, "\(log)")
        #expect(run.termination == ChildProbeTermination(status: 0, reason: .exit), "\(log)")
        #expect(run.processGroup == run.processIdentifier)
        #expect(log.contains("T10_CHILD pid=\(run.processIdentifier) pgid=\(run.processIdentifier)"), "\(log)")
        try expectGrandchildRemoved(log, group: run.processIdentifier)
    }

    @Test func deadlineKillsTheWholeGroup() throws {
        let (run, log) = try launchSelfTest("hang", deadline: 3, cleanup: .kill(seconds: 2))
        #expect(run.timedOut && run.cleaned, "\(log)")
        #expect(run.termination == ChildProbeTermination(status: SIGKILL, reason: .uncaughtSignal), "\(log)")
        try expectGrandchildRemoved(log, group: run.processIdentifier)
    }

    @Test func deadlineTerminatesTheWholeGroup() throws {
        let (run, log) = try launchSelfTest("hang", deadline: 3, cleanup: .terminateThenKill(grace: 1, seconds: 1))
        #expect(run.timedOut && run.cleaned, "\(log)")
        #expect(run.termination == ChildProbeTermination(status: SIGTERM, reason: .uncaughtSignal), "\(log)")
        try expectGrandchildRemoved(log, group: run.processIdentifier)
    }

    /// The probe body; a no-op unless launched by the tests above.
    @Test func launcherSelfTestChild() throws {
        guard let scenario = ProcessInfo.processInfo.environment[selfTestScenarioKey] else { return }
        selfTestMarker("T10_CHILD pid=\(getpid()) pgid=\(getpgrp())")
        selfTestMarker("T10_SELF_TEST_MARKER \(ProcessInfo.processInfo.environment[childProbeMarkerKey] ?? "absent")")
        switch scenario {
        case "nested":
            do {
                _ = try launchChildProbe(.swiftTesting(filter: "Never/launched"), environment: [:],
                                         output: FileHandle.standardOutput, deadline: 1, cleanup: .kill(seconds: 1))
                selfTestMarker("T10_NESTED_NOT_REFUSED")
            } catch ChildProbeLaunchError.nestedLaunch {
                selfTestMarker("T10_NESTED_REFUSED")
            }
            // The refusal must come BEFORE the spawn: this probe process must have no child at all.
            // waitid(P_ALL, WNOHANG | WNOWAIT) reaps nothing; it fails with ECHILD only when there is none.
            // ASSUMES the test runner spawns no helper subprocess inside this probe child (true for the
            // runners observed here); a runner that did would make this report a false "spawned a child".
            var info = siginfo_t()
            let waited = waitid(P_ALL, 0, &info, WEXITED | WNOHANG | WNOWAIT)
            selfTestMarker(waited == -1 && errno == ECHILD ? "T10_NESTED_SPAWNED_NOTHING" : "T10_NESTED_SPAWNED_A_CHILD")
        case "leave", "hang":
            // A grandchild in this child's process group: posix_spawn without POSIX_SPAWN_SETPGROUP
            // inherits the group. /bin/sleep 30 outlives every deadline used above.
            var pid: pid_t = 0
            let argv: [UnsafeMutablePointer<CChar>?] = [strdup("/bin/sleep"), strdup("30"), nil]
            let noEnvironment: [UnsafeMutablePointer<CChar>?] = [nil]
            defer { argv.forEach { free($0) } }
            let spawned = posix_spawn(&pid, "/bin/sleep", nil, nil, argv, noEnvironment)
            #expect(spawned == 0)
            selfTestMarker("T10_GRANDCHILD pid=\(pid) pgid=\(getpgid(pid))")
            if scenario == "hang" { while true { sleep(60) } }
        default:
            Issue.record("unknown self-test scenario \(scenario)")
        }
    }

    // MARK: helpers

    private func launchSelfTest(_ scenario: String, deadline: Double,
                                cleanup: ChildProbeCleanup) throws -> (ChildProbeRun, String) {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("T10-launcher-\(scenario)-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let output = try FileHandle(forWritingTo: url)
        defer { try? output.close() }
        var environment = ProcessInfo.processInfo.environment
        environment[selfTestScenarioKey] = scenario
        environment["SWIFT_BACKTRACE"] = "enable=no"
        let run = try launchChildProbe(.swiftTesting(filter: selfTestFilter), environment: environment, output: output,
                                       deadline: deadline, cleanup: cleanup)
        try output.close()
        let log = try String(contentsOf: url, encoding: .utf8)
        print("T10 launcher self-test \(scenario): pid=\(run.processIdentifier) timedOut=\(run.timedOut) capture=\(url.path)")
        try? FileManager.default.removeItem(at: url)
        return (run, log)
    }

    /// The grandchild named in `log` was in `group` and is gone within 3 s of the launcher returning.
    private func expectGrandchildRemoved(_ log: String, group: Int32) throws {
        let line = try #require(log.split(separator: "\n").first { $0.hasPrefix("T10_GRANDCHILD ") }, "\(log)")
        let fields = Dictionary(uniqueKeysWithValues: line.split(separator: " ").dropFirst().compactMap { field -> (String, Int32)? in
            let parts = field.split(separator: "=")
            guard parts.count == 2, let value = Int32(parts[1]) else { return nil }
            return (String(parts[0]), value)
        })
        let grandchild = try #require(fields["pid"], "\(log)")
        #expect(grandchild > 1 && fields["pgid"] == group, "grandchild must share the child's group: \(line)")
        // After SIGKILL the orphaned grandchild is reaped by launchd; until then kill(pid, 0) still succeeds.
        let limit = ContinuousClock.now + .seconds(3)
        var gone = false
        while ContinuousClock.now < limit {
            if kill(grandchild, 0) == -1 && errno == ESRCH { gone = true; break }
            usleep(20_000)
        }
        #expect(gone, "grandchild \(grandchild) of group \(group) survived the launcher")
        // A failing run must not leak the 30 s sleep: remove it only while it is still in that group.
        if !gone && getpgid(grandchild) == group { kill(grandchild, SIGKILL) }
    }
}
