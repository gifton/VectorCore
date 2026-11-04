#if os(macOS)
import Foundation
import Darwin

/// Private metadata process ownership. `testMetadataLifetimeNormalExitAndTimeout` exercises
/// normal exit and timeout; `testMetadataTimerSurvivesObserverAndGuardianDeath` kills both owners.
/// The fixed shell script arms a same-group timer before starting the command. Both the guardian
/// and timer signal their OWN group, so no cleanup ever targets a recycled numeric PID/PGID.
/// The deadline includes command execution; startup and OS scheduling are not real-time bounded.
/// Contract: commands/descendants remain in the group and do not attack/stop its guardian or timer.
/// Escaped groups are not contained (`testMetadataEscapedOutputHandleDoesNotBlockObservation`).
/// UNVERIFIED HOST CONTRACT: no foreign reaper consumes this child; SIGCHLD is not ignored.
enum MetadataProcess {
    private static let script = #"""
    set +m
    trap 'kill -KILL 0' EXIT
    (
        exec 5<&-
        printf 'armed\n' >&4 || { kill -KILL 0; exit 125; }
        exec 4>&- 3>&- 1>/dev/null 2>/dev/null
        /bin/sleep "$1"
        kill -KILL 0
    ) &
    exec 4>&-
    IFS= read -r armed <&5 || exit 125
    [ "$armed" = armed ] || exit 125
    exec 5<&-
    shift
    while [ "$1" != -- ]; do
        export "$1=${!2}" || exit 125
        unset "$2"
        shift 2
    done
    shift
    # Foreground launch preserves SIGINT/SIGQUIT defaults; an asynchronous job ignores them.
    # Covered by testMetadataOutputAndArgumentBoundaries; the independent timer still owns timeout.
    "$@" 3>&-
    result=$?
    printf '%s\n' "$result" >&3
    exit "$result"
    """#

    static func observe(command: [String], timeout: TimeInterval) -> RuntimeObservation {
        do {
            let output = try DescriptorPipe()
            defer { output.close() }
            let status = try DescriptorPipe()
            defer { status.close() }
            let armed = try DescriptorPipe(nonblocking: false)
            defer { armed.close() }
            let pid = try spawn(command: command, timeout: timeout, output: output.write,
                                status: status.write, armed: armed)
            // A per-child reaper keeps the PID reserved until waitpid completes. It never signals.
            // If the OS cannot terminate the guardian, the caller still returns after its deadline;
            // this thread remains responsible for eventual reaping and owns no output descriptors.
            let reaped = DispatchSemaphore(value: 0)
            Thread {
                var result: Int32 = 0
                var waited: pid_t
                repeat { waited = waitpid(pid, &result, 0) } while waited == -1 && errno == EINTR
                if waited == pid { reaped.signal() }
            }.start()
            output.closeWriter()
            status.closeWriter()
            armed.close()
            let deadline = ContinuousClock.now.advanced(by: .seconds(timeout + 2))
            var bytes = Data(), statusBytes = Data()
            var overflow = false
            var done = false
            repeat {
                try drain(output.read, into: &bytes, limit: 65_536, overflow: &overflow)
                try drain(status.read, into: &statusBytes, limit: 64, overflow: &overflow)
                done = reaped.wait(timeout: .now() + 0.005) == .success
            } while !done && ContinuousClock.now < deadline
            guard done else { return .unavailable("Observation timeout; guardian reap not observed") }
            try drain(output.read, into: &bytes, limit: 65_536, overflow: &overflow)
            try drain(status.read, into: &statusBytes, limit: 64, overflow: &overflow)
            guard !overflow else { return .unavailable("Observation exceeded output limit") }
            guard let statusText = String(data: statusBytes, encoding: .utf8),
                  let code = Int32(statusText.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                return .unavailable("Observation timeout or guardian failure; command status unavailable")
            }
            guard code == 0 else { return .unavailable("Observation command failed: \(code)") }
            guard let text = String(data: bytes, encoding: .utf8) else { return .unavailable("Observation output is not UTF-8") }
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? .unavailable("Observation returned empty output") : RuntimeObservation(value: trimmed)
        } catch { return .unavailable("Observation setup/read failed: \(error)") }
    }

    // Each read pass is capped even if a writer produces indefinitely; EOF is never awaited.
    private static func drain(_ descriptor: Int32, into data: inout Data, limit: Int, overflow: inout Bool) throws {
        var buffer = [UInt8](repeating: 0, count: 4096)
        for _ in 0..<17 {
            let count = Darwin.read(descriptor, &buffer, buffer.count)
            if count == 0 { return }
            if count < 0 {
                if errno == EAGAIN || errno == EINTR { return }
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            let room = max(0, limit - data.count)
            data.append(contentsOf: buffer.prefix(min(count, room)))
            if count > room { overflow = true }
        }
    }

    private static func spawn(command: [String], timeout: TimeInterval, output: Int32,
                              status: Int32, armed: DescriptorPipe) throws -> pid_t {
        var attributes: posix_spawnattr_t?
        try check(posix_spawnattr_init(&attributes))
        defer { posix_spawnattr_destroy(&attributes) }
        var actions: posix_spawn_file_actions_t?
        try check(posix_spawn_file_actions_init(&actions))
        defer { posix_spawn_file_actions_destroy(&actions) }
        var noSignals = sigset_t(), defaults = sigset_t()
        sigemptyset(&noSignals)
        sigfillset(&defaults)
        sigdelset(&defaults, SIGKILL)
        sigdelset(&defaults, SIGSTOP)
        let flags = POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF
        try check(posix_spawnattr_setflags(&attributes, Int16(flags)))
        try check(posix_spawnattr_setpgroup(&attributes, 0))
        try check(posix_spawnattr_setsigmask(&attributes, &noSignals))
        try check(posix_spawnattr_setsigdefault(&attributes, &defaults))
        try check(posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0))
        try check(posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0))
        for (source, destination): (Int32, Int32) in [(output, 1), (status, 3), (armed.write, 4), (armed.read, 5)] {
            try check(posix_spawn_file_actions_adddup2(&actions, source, destination))
        }
        // -p disables importing shell functions/options from the environment; argv stays data.
        var environment = ProcessInfo.processInfo.environment
        var restoredNames: [String] = []
        // Protected /bin/sh strips loader variables. Carry their values in private environment
        // aliases, never argv; also restore original shell-modified values before the command.
        // Quoted indirect expansion restores data without eval. The original
        // target's own protection rules still apply (testMetadataPreservesLoaderEnvironmentWithoutShellEvaluation).
        let shellModifiedVariables: Set<String> = ["IFS", "armed", "PWD", "SHLVL"]
        for name in environment.keys.sorted()
        where name.hasPrefix("DYLD_") || name == "LD_LIBRARY_PATH" || shellModifiedVariables.contains(name) {
            guard name.utf8.allSatisfy({ (65...90).contains($0) || (97...122).contains($0)
                || (48...57).contains($0) || $0 == 95 }) else { throw POSIXError(.EINVAL) }
            var alias = "__VC_METADATA_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
            while environment[alias] != nil { alias += "_" }
            environment[alias] = environment[name]
            restoredNames += [name, alias]
        }
        let arguments = ["/bin/sh", "-p", "-c", script, "metadata-guardian", String(timeout)]
            + restoredNames + ["--"] + command
        let argv = arguments.map { strdup($0) } + [nil]
        defer { argv.forEach { free($0) } }
        let envp = environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { envp.forEach { free($0) } }
        guard argv.dropLast().allSatisfy({ $0 != nil }), envp.dropLast().allSatisfy({ $0 != nil }) else {
            throw POSIXError(.ENOMEM)
        }
        var pid: pid_t = 0
        try check(posix_spawn(&pid, "/bin/sh", &actions, &attributes, argv, envp))
        return pid
    }

    private static func check(_ code: Int32) throws {
        if code != 0 { throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO) }
    }

    private final class DescriptorPipe {
        private(set) var read: Int32
        private(set) var write: Int32
        init(nonblocking: Bool = true) throws {
            var pair: [Int32] = [-1, -1]
            guard pipe(&pair) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            defer { Darwin.close(pair[0]); Darwin.close(pair[1]) }
            // Keep spawn source descriptors above the fixed destinations 0...5.
            read = fcntl(pair[0], F_DUPFD_CLOEXEC, 10)
            write = fcntl(pair[1], F_DUPFD_CLOEXEC, 10)
            let configured = !nonblocking || (read >= 0 && fcntl(read, F_SETFL, O_NONBLOCK) == 0)
            guard read >= 0, write >= 0, configured else {
                let code = errno
                if read >= 0 { Darwin.close(read) }
                if write >= 0 { Darwin.close(write) }
                throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
            }
        }
        func closeWriter() { if write >= 0 { Darwin.close(write); write = -1 } }
        func close() {
            if read >= 0 { Darwin.close(read); read = -1 }
            closeWriter()
        }
    }
}
#endif
