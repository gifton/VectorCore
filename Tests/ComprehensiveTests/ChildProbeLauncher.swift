// T10: the only way a ComprehensiveTests test re-runs this test binary as a child probe.
//
// Scripts/ci/tests/test_child_probe_launch_policy.py rejects process creation and test-runner selector
// literals in Tests/ outside this file, except the exceptions it pins by exact count (one of them is
// BenchmarkContractTests' RunnerContractTests leaf launch with a fixed one-selector XCTest argument list).
//
// Before T10, 22 call sites copied `CommandLine.arguments`, replaced only the FIRST `--filter` value
// and launched the copy with Foundation `Process`. With several `--filter` arguments the child also
// ran the other selected tests; each of them that launches a probe launched again, without bound.
// A deadline killed only the direct child (by PID; in T10's before-run every `Process` child also led
// its own process group), so the rest of the chain was orphaned and kept spawning.
//
// Contract (each clause names the test that pins it; ChildProbeLauncherTests is in ComprehensiveTests):
// 1. Selection. The child keeps NO selector of the parent and gets exactly one selector, which must
//    pass a SHAPE check (`validateSingleTestSelection`, pinned by `selectorNamesExactlyOneTest`):
//    `Owner/name` where Owner is one or more ASCII identifiers joined by `.` and name is one ASCII
//    identifier. That rejects commas (XCTest lists) and every regex metacharacter except `.`, which the
//    `Module.Suite` form needs and which `--filter` treats as a wildcard. It does NOT prove the selector
//    names exactly one test: swift-testing matches `--filter` unanchored, and `Suite/NestedSuite`
//    passes the check yet selects a whole nested suite. Over-match is bounded by the marker (clause 2):
//    an over-selected test that launches a probe fails loudly instead of recursing. Anchoring the
//    filter is a routed follow-up (unverified on Swift 6.0).
//    - `.swiftTesting(filter:)`: the parent's runner arguments minus every value-taking selection flag
//      (`--filter`, `--skip`, SwiftPM's `--specifier` / `-s`, separate-value and `=` forms) and every
//      report sink (`--xunit-output`, the event-stream path and version flags), then one
//      `--filter <probe>`. A token that is the value of a `-X…` pass-through flag (`-Xlinker -s`) is
//      kept as is, unless it is a flag swift-testing itself parses as selection or report sink
//      (`--filter`, `--skip`, the sinks, the configuration path; either form): swift-testing ignores
//      `-X` pairing, so such a value is refused (`refusedRunnerArgument`). SwiftPM's `--specifier` /
//      `-s` are not parsed by swift-testing, so `-Xlinker -s` stays legal.
//      `--configuration-path` (a JSON file that can carry selection) is refused.
//      Pinned by `selectionIsReplacedNotEdited`, `onlySelectionAndSinksAreRemoved`,
//      `configurationFileIsRefused`.
//    - Runner shapes (`childProbeRunnerShape`, pinned by `runnerShapesAreRecognizedOrRefused`):
//      * SwiftPM helper: `swiftpm-testing-helper --test-bundle-path <bin> <raw swift test argv> <bin>
//        --testing-library swift-testing`. Observed with Xcode 27 / Swift 6.4 (Testing 2084, this
//        suite) and with the Command Line Tools 6.2.0 toolchain (swiftlang-6.2.0.19.9, Testing 1085,
//        a swift-testing-only probe package; T10 fix1 evidence m8-*), which is only a PROXY for CI's
//        Xcode 26.3 (Swift 6.2.x) toolchain: that toolchain was not run. Swift 6.0 (Xcode 16.2, CI) is
//        NOT verified locally. The CI matrix is the real gate. Detection accepts any argv with
//        `--test-bundle-path <value>`; it does not check argv[0] or `--testing-library`.
//      * Direct test executable: argv[0] is the test binary itself and the arguments contain
//        `--testing-library swift-testing`. Accepted, but NOT observed on any toolchain here.
//      * Anything else (for example swift-testing hosted by `xctest`): the launch fails with
//        `CHILD_PROBE_UNRECOGNIZED_RUNNER` on stderr and in the thrown error, both naming the observed
//        argv, so a CI failure is diagnosable; it never silently skips.
//    - `.xctest(selector:bundlePath:)`: a fresh `-XCTest <selector> <bundle>`; nothing is inherited.
//      Pinned by `xctestSelectionIsFresh`.
// 2. Marker. The child's environment carries `VECTORCORE_CHILD_PROBE=<selector>`. A process that
//    has the marker refuses to launch a probe, before anything is spawned: a diagnostic line on
//    stderr, then `ChildProbeLaunchError.nestedLaunch`. Pinned by `nestedLaunchIsRefused` (unit),
//    `childProbeCannotLaunchAProbe` (end to end, through a real child) and the policy test's
//    refusal-before-spawn order check.
// 3. Process group. The child is spawned with `POSIX_SPAWN_SETPGROUP` (pgroup 0), so it leads a new
//    group whose ID is its PID (pinned by `exitedChildLeavesNoGroupMember`, which reads the child's
//    own getpgrp()). The deadline signals the GROUP. After the leader exits, a reaper thread sends
//    SIGKILL to the group BEFORE it reaps the leader (waitid with WNOWAIT keeps the zombie, so the
//    group ID cannot be recycled first); members left behind therefore die on every exit path.
//    Group signals and every reap attempt happen under the reaper lock, and a group is never signalled
//    after THIS reaper has reaped the leader. If waitid itself fails (not expected for our own child),
//    the reaper polls waitpid(WNOHANG) under the lock and sends no sweep (there is no zombie
//    guarantee); that fallback's locking is derivable from the code but no test exercises it.
//    UNVERIFIED CONTRACT: no foreign reaper exists in the process (no `waitpid(-1, …)` elsewhere,
//    SIGCHLD not ignored, no SA_NOCLDWAIT). If one reaps the leader first, the fallback and the
//    deadline paths can still `kill(-pid, …)` a process group whose ID may by then be recycled.
//    `sendToGroup` sends nothing at all for a pid <= 1 or equal to getpgrp(), so the caller's own
//    group, kill(0, …) and kill(-1, …) are unreachable. Pinned by `exitedChildLeavesNoGroupMember`,
//    `deadlineKillsTheWholeGroup`, `deadlineTerminatesTheWholeGroup`.
// 4. Bounds. The caller supplies the deadline and cleanup budget (each call site keeps its previous
//    values). The caller never waits without a deadline; only the reaper thread blocks (in waitid,
//    then waitpid), and it ends when the leader exits.
// 5. Results never crash the caller. `terminationStatus` / `terminationReason` are nil for a child
//    that was not reaped (deadline expired and cleanup failed), so an assertion on them FAILS instead
//    of trapping; `reaped()` throws `ChildProbeLaunchError.unreaped` for callers that need values.
//    Pinned by `unreapedChildReadsFailWithoutTrapping`.
// 6. Child process state: stdin is /dev/null, stdout and stderr go to `output`, every other
//    descriptor is closed (POSIX_SPAWN_CLOEXEC_DEFAULT), the signal mask is empty and every
//    signal disposition is reset to its default (so SIGTERM is not inherited as ignored).
//    Unverified beyond the SIGTERM path of `deadlineTerminatesTheWholeGroup`.

import Foundation
import Darwin

/// Environment key that marks a child probe. Its value is the selector the child was launched for.
let childProbeMarkerKey = "VECTORCORE_CHILD_PROBE"

/// Value-taking flags removed from a swift-testing child, in both `flag value` and `flag=value` forms.
let childProbeDroppedFlags: Set<String> = [
    "--filter", "--skip", "--specifier", "-s",
    "--xunit-output", "--event-stream-output-path", "--experimental-event-stream-output",
    "--event-stream-version", "--experimental-event-stream-version"
]

/// Flags whose presence makes the child's selection unknowable; a launch with them is refused.
let childProbeRefusedFlags: Set<String> = ["--configuration-path", "--experimental-configuration-path"]

/// The dropped and refused flags that swift-testing itself parses (not SwiftPM's `--specifier` / `-s`).
/// As the value of a `-X…` pass-through flag one of these is refused, never kept.
let childProbeTestingParsedFlags = childProbeDroppedFlags.subtracting(["--specifier", "-s"]).union(childProbeRefusedFlags)

enum ChildProbeSelector: Sendable, CustomStringConvertible {
    /// A swift-testing probe, re-run through the parent's swift-testing runner.
    case swiftTesting(filter: String)
    /// An XCTest probe, re-run through `xctest`.
    case xctest(selector: String, bundlePath: String)

    var description: String {
        switch self {
        case .swiftTesting(let filter): return filter
        case .xctest(let selector, _): return selector
        }
    }
}

enum ChildProbeCleanup: Sendable {
    /// At the deadline: SIGKILL the group, then wait up to `seconds` for the child to be reaped.
    case kill(seconds: Double)
    /// At the deadline: SIGTERM the group and wait `grace`; if the child is still unreaped,
    /// SIGKILL the group and wait up to `seconds`.
    case terminateThenKill(grace: Double, seconds: Double)
}

/// Runner argv shapes the launcher can re-select a single test from (contract clause 1).
enum ChildProbeRunnerShape: Sendable, Equatable {
    case swiftPMTestingHelper
    case directTestExecutable
}

enum ChildProbeLaunchError: Error, CustomStringConvertible {
    /// A child probe tried to launch a probe.
    case nestedLaunch(String)
    /// The parent's runner argv has a shape the launcher cannot re-select from.
    case unrecognizedRunner(String)
    /// The parent's runner argv carries a selection source that cannot be removed safely.
    case refusedRunnerArgument(String)
    /// The selector does not name exactly one test.
    case invalidSelection(String)
    /// posix_spawn or its attribute setup failed.
    case spawnFailed(String)
    /// The child exited but its wait status could not be read.
    case waitStatusUnavailable(String)
    /// `reaped()` was called for a child that was not reaped within the cleanup budget.
    case unreaped(String)

    var description: String {
        switch self {
        case .nestedLaunch(let message), .unrecognizedRunner(let message), .refusedRunnerArgument(let message),
             .invalidSelection(let message), .spawnFailed(let message), .waitStatusUnavailable(let message),
             .unreaped(let message):
            return message
        }
    }
}

struct ChildProbeTermination: Sendable, Equatable {
    /// Exit code for `.exit`, signal number for `.uncaughtSignal` (as Foundation `Process` reports).
    let status: Int32
    let reason: Process.TerminationReason
}

struct ChildProbeRun: Sendable {
    let executable: String
    /// The child's arguments without argv[0].
    let arguments: [String]
    let processIdentifier: Int32
    /// Equal to `processIdentifier`: the child leads its own process group.
    let processGroup: Int32
    let timedOut: Bool
    /// False only when the deadline expired and the child was not reaped within the cleanup budget.
    let cleaned: Bool
    /// Nil only when `cleaned` is false.
    let termination: ChildProbeTermination?

    /// `.success` when the child was reaped within the budget, as the call sites reported it before.
    var cleanup: DispatchTimeoutResult { cleaned ? .success : .timedOut }
    var isRunning: Bool { termination == nil }
    /// Nil for an unreaped child, so `== 0` / `== .uncaughtSignal` assertions fail rather than trap.
    var terminationStatus: Int32? { termination?.status }
    var terminationReason: Process.TerminationReason? { termination?.reason }
    /// For diagnostics: the value as before, or "unreaped".
    var statusDescription: String { termination.map { String($0.status) } ?? "unreaped" }
    var reasonDescription: String { termination.map { String($0.reason.rawValue) } ?? "unreaped" }

    /// The termination, or a thrown `ChildProbeLaunchError.unreaped` (a test failure, not a trap).
    func reaped() throws -> ChildProbeTermination {
        guard let termination else {
            throw ChildProbeLaunchError.unreaped(
                "child probe pid \(processIdentifier) was not reaped within its cleanup budget (timedOut=\(timedOut))")
        }
        return termination
    }
}

/// Throws when `environment` is a child probe's environment; writes the reason to stderr first so it
/// lands in the child's capture.
func refuseNestedChildProbeLaunch(_ selector: ChildProbeSelector, environment: [String: String]) throws {
    guard let outer = environment[childProbeMarkerKey] else { return }
    let message = "CHILD_PROBE_NESTED_LAUNCH_REFUSED: child probe '\(outer)' tried to launch probe "
        + "'\(selector)'; a child probe must not launch probes"
    FileHandle.standardError.write(Data((message + "\n").utf8))
    throw ChildProbeLaunchError.nestedLaunch(message)
}

/// SHAPE check only (contract clause 1): throws unless `selection` is `Owner/name`, Owner one or more
/// ASCII identifiers joined by `.`, name one ASCII identifier. Rejects comma lists (XCTest `-XCTest`
/// accepts `A/b,C/d`) and every regex metacharacter except `.` (a `--filter` wildcard, needed for
/// `Module.Suite`). It does not prove that the selector matches exactly one test.
func validateSingleTestSelection(_ selection: String) throws {
    func isIdentifier(_ part: Substring) -> Bool {
        guard let first = part.unicodeScalars.first, first.isASCII,
              first == "_" || CharacterSet.letters.contains(first) else { return false }
        return part.unicodeScalars.allSatisfy { $0.isASCII && ($0 == "_" || CharacterSet.alphanumerics.contains($0)) }
    }
    let halves = selection.split(separator: "/", omittingEmptySubsequences: false)
    let owner = halves.first?.split(separator: ".", omittingEmptySubsequences: false) ?? []
    guard halves.count == 2, !owner.isEmpty, owner.allSatisfy(isIdentifier), isIdentifier(halves[1]) else {
        throw ChildProbeLaunchError.invalidSelection(
            "child probe selector '\(selection)' must name one test as Suite/test or Module.Suite/test")
    }
}

/// The runner shape of the parent (`executable` is argv[0], `arguments` the rest), or nil.
func childProbeRunnerShape(executable: String, arguments: [String]) -> ChildProbeRunnerShape? {
    if let index = arguments.firstIndex(of: "--test-bundle-path"), index + 1 < arguments.count {
        return .swiftPMTestingHelper
    }
    let name = executable.split(separator: "/").last.map(String.init) ?? executable
    let hosts: Set<String> = ["xctest", "swiftpm-testing-helper", "swiftpm-xctest-helper"]
    let selectsSwiftTesting = zip(arguments, arguments.dropFirst()).contains { $0 == "--testing-library" && $1 == "swift-testing" }
    return !hosts.contains(name) && selectsSwiftTesting ? .directTestExecutable : nil
}

/// The child's arguments (without argv[0]) for `selector`, given the parent's argv[0] and arguments.
func childProbeArguments(for selector: ChildProbeSelector, executable: String,
                         parentArguments: [String]) throws -> [String] {
    try validateSingleTestSelection(selector.description)
    switch selector {
    case .xctest(let selection, let bundlePath):
        return ["-XCTest", selection, bundlePath]
    case .swiftTesting(let filter):
        guard childProbeRunnerShape(executable: executable, arguments: parentArguments) != nil else {
            let message = "CHILD_PROBE_UNRECOGNIZED_RUNNER: child probe '\(filter)' cannot derive a one-test argv "
                + "from executable=\(executable) arguments=\(parentArguments)"
            FileHandle.standardError.write(Data((message + "\n").utf8))
            throw ChildProbeLaunchError.unrecognizedRunner(message)
        }
        var kept: [String] = []
        var index = 0
        var passThroughValue = false
        while index < parentArguments.count {
            let argument = parentArguments[index]
            if passThroughValue || (argument.hasPrefix("-X") && !argument.contains("=")) {
                // A -Xlinker/-Xswiftc/... flag, or its value (`-Xlinker -s` keeps the `-s`). swift-testing
                // ignores -X pairing, so a value it would parse as selection or sink is refused.
                let valueName = String(argument.prefix { $0 != "=" })
                if passThroughValue && childProbeTestingParsedFlags.contains(valueName) {
                    throw ChildProbeLaunchError.refusedRunnerArgument(
                        "child probe '\(filter)': \(argument) follows a -X pass-through flag but swift-testing "
                        + "would parse it: \(parentArguments)")
                }
                kept.append(argument)
                passThroughValue.toggle()
                index += 1
                continue
            }
            let name = argument.hasPrefix("-") ? String(argument.prefix { $0 != "=" }) : argument
            if childProbeRefusedFlags.contains(name) {
                throw ChildProbeLaunchError.refusedRunnerArgument(
                    "child probe '\(filter)': \(name) can carry test selection and is not forwarded: \(parentArguments)")
            }
            if childProbeDroppedFlags.contains(argument) {
                index += 2  // the flag and its separate value
                continue
            }
            if name != argument && childProbeDroppedFlags.contains(name) {
                index += 1  // flag=value
                continue
            }
            kept.append(argument)
            index += 1
        }
        return kept + ["--filter", filter]
    }
}

/// Launches the probe selected by `selector` in a new process group, waits at most `deadline`
/// seconds, applies `cleanup` to the whole group on expiry, and always removes group members the
/// child leaves behind. See the file header for the contract.
func launchChildProbe(_ selector: ChildProbeSelector, environment: [String: String], output: FileHandle,
                      deadline: Double, cleanup: ChildProbeCleanup) throws -> ChildProbeRun {
    try refuseNestedChildProbeLaunch(selector, environment: ProcessInfo.processInfo.environment)
    let executable = CommandLine.arguments[0]
    let arguments = try childProbeArguments(for: selector, executable: executable,
                                            parentArguments: Array(CommandLine.arguments.dropFirst()))
    var childEnvironment = environment
    childEnvironment[childProbeMarkerKey] = selector.description
    let pid = try spawnChildProbeGroupLeader(executable: executable, arguments: arguments,
                                             environment: childEnvironment, output: output.fileDescriptor)
    let reaper = ChildProbeReaper(pid: pid)
    let thread = Thread { reaper.run() }
    thread.name = "child-probe-reaper-\(pid)"
    thread.start()

    var timedOut = false
    var cleaned = true
    if reaper.reaped.wait(timeout: .now() + deadline) == .timedOut {
        timedOut = true
        switch cleanup {
        case .kill(let seconds):
            reaper.signalGroup(SIGKILL)
            cleaned = reaper.reaped.wait(timeout: .now() + seconds) == .success
        case .terminateThenKill(let grace, let seconds):
            reaper.signalGroup(SIGTERM)
            if reaper.reaped.wait(timeout: .now() + grace) == .timedOut {
                reaper.signalGroup(SIGKILL)
                cleaned = reaper.reaped.wait(timeout: .now() + seconds) == .success
            }
        }
    }
    let termination = cleaned ? reaper.termination : nil
    if cleaned && termination == nil {
        throw ChildProbeLaunchError.waitStatusUnavailable(
            "child probe '\(selector)': pid \(pid) exited but its wait status was unavailable (reaped elsewhere)")
    }
    return ChildProbeRun(executable: executable, arguments: arguments, processIdentifier: pid, processGroup: pid,
                         timedOut: timedOut, cleaned: cleaned, termination: termination)
}

/// Owns the child's PID and process group from spawn to reap.
private final class ChildProbeReaper: @unchecked Sendable {
    let pid: pid_t
    let reaped = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var isReaped = false
    private var result: ChildProbeTermination?

    init(pid: pid_t) { self.pid = pid }

    var termination: ChildProbeTermination? {
        lock.lock(); defer { lock.unlock() }
        return result
    }

    /// Signals the child's group while the leader is unreaped (alive or a zombie), so the group ID
    /// still belongs to this child. Never signals the caller's own group.
    func signalGroup(_ signal: Int32) {
        lock.lock(); defer { lock.unlock() }
        guard !isReaped else { return }
        sendToGroup(signal)
    }

    /// Only ever this child's own group: nothing is sent for pid <= 1 (kill(0, …) and kill(-1, …) mean
    /// "my group" and "every process") or for the caller's own group.
    private func sendToGroup(_ signal: Int32) {
        guard pid > 1, pid != getpgrp() else { return }
        _ = kill(-pid, signal)
    }

    func run() {
        var info = siginfo_t()
        var observed: Int32
        repeat { observed = waitid(P_PID, id_t(pid), &info, WEXITED | WNOWAIT) } while observed == -1 && errno == EINTR
        var status: Int32 = 0
        var waited: pid_t
        if observed == 0 {
            lock.lock()
            // The leader is a zombie: the group ID is still reserved, so this reaches only members
            // the child left in its group. The reap below returns at once.
            sendToGroup(SIGKILL)
            repeat { waited = waitpid(pid, &status, 0) } while waited == -1 && errno == EINTR
        } else {
            // waitid failed (not expected for our own child). Without WNOWAIT there is no zombie
            // guarantee, so no sweep is sent; each non-blocking reap attempt and the `isReaped` change
            // happen under the lock, so a deadline signal never lands between them.
            while true {
                lock.lock()
                waited = waitpid(pid, &status, WNOHANG)
                if waited == pid || (waited == -1 && errno != EINTR) { break }  // keeps the lock
                lock.unlock()
                usleep(10_000)
            }
        }
        isReaped = true
        if waited == pid {
            let low = status & 0x7f
            result = low == 0 ? ChildProbeTermination(status: (status >> 8) & 0xff, reason: .exit)
                              : ChildProbeTermination(status: low, reason: .uncaughtSignal)
        }
        lock.unlock()
        reaped.signal()
    }
}

private func spawnChildProbeGroupLeader(executable: String, arguments: [String], environment: [String: String],
                                        output: Int32) throws -> pid_t {
    func check(_ code: Int32, _ step: String) throws {
        guard code == 0 else {
            throw ChildProbeLaunchError.spawnFailed("child probe spawn: \(step) failed: \(String(cString: strerror(code)))")
        }
    }
    var attributes: posix_spawnattr_t?
    try check(posix_spawnattr_init(&attributes), "posix_spawnattr_init")
    defer { posix_spawnattr_destroy(&attributes) }
    var actions: posix_spawn_file_actions_t?
    try check(posix_spawn_file_actions_init(&actions), "posix_spawn_file_actions_init")
    defer { posix_spawn_file_actions_destroy(&actions) }

    var noSignals = sigset_t()
    sigemptyset(&noSignals)
    var everySignal = sigset_t()
    sigfillset(&everySignal)
    sigdelset(&everySignal, SIGKILL)
    sigdelset(&everySignal, SIGSTOP)
    let flags = POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF
    try check(posix_spawnattr_setflags(&attributes, Int16(flags)), "setflags")
    try check(posix_spawnattr_setpgroup(&attributes, 0), "setpgroup")
    try check(posix_spawnattr_setsigmask(&attributes, &noSignals), "setsigmask")
    try check(posix_spawnattr_setsigdefault(&attributes, &everySignal), "setsigdefault")
    try check(posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0), "stdin")
    try check(posix_spawn_file_actions_adddup2(&actions, output, 1), "stdout")
    try check(posix_spawn_file_actions_adddup2(&actions, output, 2), "stderr")

    let argv: [UnsafeMutablePointer<CChar>?] = ([executable] + arguments).map { strdup($0) } + [nil]
    let envp: [UnsafeMutablePointer<CChar>?] = environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
    defer {
        argv.forEach { free($0) }
        envp.forEach { free($0) }
    }
    var pid: pid_t = 0
    try check(posix_spawn(&pid, executable, &actions, &attributes, argv, envp), "posix_spawn \(executable)")
    return pid
}
