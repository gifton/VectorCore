"""Unit tests for Scripts/ci/check_sdk_name_collisions.py (stdlib only, no toolchain).

The checker's subprocess boundary (`run`) is replaced by a fake that fabricates
symbol graphs, so these tests exercise the pure comparison logic and the exit-2
error paths without Xcode, an SDK or a package build. Nothing here spawns a
process, so no subprocess timeout is needed.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

CHECKER = Path(__file__).resolve().parents[1] / 'check_sdk_name_collisions.py'
spec = importlib.util.spec_from_file_location('check_sdk_name_collisions', CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

TRIPLE = 'arm64-apple-macosx27.0'


def graph(names, kind='swift.struct', access='public'):
    return {'symbols': [{'kind': {'identifier': kind}, 'accessLevel': access, 'pathComponents': [n]}
                        for n in names]}


def completed(returncode=0, stdout='', stderr=''):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class FakeToolchain:
    """Replacement for checker.run. `ours` and `sdk` map module to top-level type names."""

    def __init__(self, ours=None, sdk=None, sdk_path_exit=0, dump_exit=0, write_package_graphs=True,
                 drop_sentinels=(), drop_package_sentinels=()):
        self.ours = {module: list(names) + [module + 'Own'] for module, names in checker.PACKAGE_SENTINELS.items()}
        for module, names in (ours or {}).items():
            self.ours[module] = self.ours[module] + list(names)
        for module, name in drop_package_sentinels:
            self.ours[module].remove(name)
        self.sdk = {module: list(names) for module, names in checker.SENTINELS.items()}
        for module, names in (sdk or {}).items():
            self.sdk[module] = self.sdk.get(module, []) + list(names)
        for module, name in drop_sentinels:
            self.sdk[module].remove(name)
        self.sdk_path_exit = sdk_path_exit
        self.dump_exit = dump_exit
        self.write_package_graphs = write_package_graphs
        self.calls = []

    def __call__(self, argv, timeout, cwd=None):
        self.calls.append(list(argv))
        if argv[:2] == ['xcrun', '--show-sdk-path']:
            return completed(self.sdk_path_exit, '/fake/sdk\n' if self.sdk_path_exit == 0 else '', 'xcrun: error')
        if argv[:2] == ['xcrun', 'swift']:
            return completed(0, json.dumps({'target': {'triple': TRIPLE}}))
        if argv[:2] == ['swift', 'package']:
            scratch = Path(argv[argv.index('--scratch-path') + 1])
            if self.write_package_graphs:
                for module, names in self.ours.items():
                    target = scratch / 'symbolgraph' / module
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f'{module}.symbols.json').write_text(json.dumps(graph(names)))
            return completed(self.dump_exit, '', 'dump output')
        if argv[:2] == ['xcrun', 'swift-symbolgraph-extract']:
            module = argv[argv.index('-module-name') + 1]
            out = Path(argv[argv.index('-output-dir') + 1])
            (out / f'{module}.symbols.json').write_text(json.dumps(graph(self.sdk[module])))
            return completed(0)
        raise AssertionError('unexpected command: ' + repr(argv))


class CheckerTestCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def allowlist(self, *entries):
        path = self.root / 'allowlist.json'
        path.write_text(json.dumps([{'name': n, 'module': m, 'reason': 'test'} for n, m in entries]))
        return path

    def write_sources(self, fake, extra_source=None):
        """A synthetic package whose Sources/<Module> declares exactly the names the fake graphs contain."""
        package = self.root / 'package'
        for module, names in fake.ours.items():
            directory = package / 'Sources' / module
            directory.mkdir(parents=True, exist_ok=True)
            declared = list(names) + list((extra_source or {}).get(module, []))
            lines = [f'public struct {n} {{}}' for n in declared]
            lines += ['    public struct Nested {}', 'struct Internal {}', 'extension X {', '    public enum Inner {}', '}']
            (directory / 'A.swift').write_text('\n'.join(lines) + '\n')
        return package

    def run_main(self, fake, allowlist=None, extra=(), extra_source=None):
        package = self.write_sources(fake, extra_source)
        argv = ['--sdk', '/fake/sdk', '--allowlist', str(allowlist or self.allowlist()),
                '--package-path', str(package), '--scratch-path', str(self.root / 'scratch'), *extra]
        if fake.sdk_path_exit != 0:
            argv = [a for a in argv if a not in ('--sdk', '/fake/sdk')]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(checker, 'run', fake), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = checker.main(argv)
        return code, out.getvalue(), err.getvalue()


class ExitCodeTests(CheckerTestCase):
    def test_clean_run_exits_zero(self):
        code, out, _ = self.run_main(FakeToolchain())
        self.assertEqual(code, 0)
        self.assertIn('OK: no unreviewed SDK name collisions', out)

    def test_unreviewed_collision_exits_one(self):
        code, out, _ = self.run_main(FakeToolchain(ours={'VectorCore': ['Data']}))
        self.assertEqual(code, 1)
        self.assertIn('COLLISION: VectorCore.Data == Foundation.Data', out)

    def test_stdlib_collision_is_informational(self):
        code, out, _ = self.run_main(FakeToolchain(ours={'VectorCore': ['Array']}))
        self.assertEqual(code, 0)
        self.assertIn('info (Swift stdlib, not a failure): VectorCore.Array == Swift.Array', out)

    def test_xcrun_failure_exits_two(self):
        code, out, err = self.run_main(FakeToolchain(sdk_path_exit=1))
        self.assertEqual(code, 2)
        self.assertIn('xcrun --show-sdk-path failed', err)
        self.assertNotIn('OK:', out)

    def test_no_fresh_graphs_exits_two(self):
        fake = FakeToolchain(write_package_graphs=False, dump_exit=1)
        code, _, err = self.run_main(fake)
        self.assertEqual(code, 2)
        self.assertIn('no fresh symbol graph for VectorCore', err)

    def test_graphs_left_by_an_earlier_run_are_not_fresh(self):
        # A graph in the user-supplied parent (an earlier run) must not satisfy a failed dump.
        parent = self.root / 'scratch'
        old = parent / 'run-previous' / 'symbolgraph'
        old.mkdir(parents=True)
        for module in checker.PACKAGE_MODULES:
            (old / f'{module}.symbols.json').write_text(json.dumps(graph(['Data'])))
        fake = FakeToolchain(write_package_graphs=False, dump_exit=1)
        code, _, err = self.run_main(fake)
        self.assertEqual(code, 2)
        self.assertIn('no fresh symbol graph', err)

    def test_nonzero_dump_with_fresh_graphs_still_proceeds(self):
        code, _, err = self.run_main(FakeToolchain(dump_exit=1))
        self.assertEqual(code, 0)
        self.assertIn('warning: dump-symbol-graph exited 1', err)

    def test_missing_sentinel_exits_two(self):
        fake = FakeToolchain(drop_sentinels=[('Foundation', 'AttributedString')])
        code, out, err = self.run_main(fake)
        self.assertEqual(code, 2)
        self.assertIn('extraction self-check failed for Foundation', err)
        self.assertIn('AttributedString', err)
        self.assertNotIn('OK:', out)

    def test_missing_package_sentinel_exits_two(self):
        fake = FakeToolchain(drop_package_sentinels=[('VectorCoreBenchmarking', 'BenchmarkProgressReporter')])
        code, out, err = self.run_main(fake)
        self.assertEqual(code, 2)
        self.assertIn('package extraction self-check failed for VectorCoreBenchmarking', err)
        self.assertIn('BenchmarkProgressReporter', err)
        self.assertNotIn('OK:', out)

    def test_partial_graph_after_failed_dump_exits_two(self):
        fake = FakeToolchain(dump_exit=1, drop_package_sentinels=[('VectorCore', 'DistanceMetric')])
        code, _, err = self.run_main(fake)
        self.assertEqual(code, 2)
        self.assertIn('dump exit 1', err)

    def test_failed_dump_with_every_package_sentinel_passes(self):
        code, out, err = self.run_main(FakeToolchain(dump_exit=1))
        self.assertEqual(code, 0)
        self.assertIn('warning: dump-symbol-graph exited 1', err)
        self.assertIn('OK:', out)

    def test_every_package_module_has_sentinels(self):
        self.assertEqual(set(checker.PACKAGE_SENTINELS), set(checker.PACKAGE_MODULES))
        self.assertTrue(all(checker.PACKAGE_SENTINELS.values()))

    def test_source_declared_name_missing_from_graph_exits_two(self):
        code, out, err = self.run_main(FakeToolchain(), extra_source={'VectorCore': ['Dropped']})
        self.assertEqual(code, 2)
        self.assertIn('source cross-check failed for VectorCore', err)
        self.assertIn("'Dropped'", err)
        self.assertNotIn('OK:', out)

    def test_exact_source_match_passes(self):
        code, out, err = self.run_main(FakeToolchain())
        self.assertEqual(code, 0, err)
        self.assertNotIn('source cross-check', err)

    def test_allowlisted_source_omission_passes(self):
        with mock.patch.object(checker, 'SOURCE_OMISSION_ALLOWLIST', {('VectorCore', 'Dropped'): 'reviewed: test'}):
            code, out, err = self.run_main(FakeToolchain(), extra_source={'VectorCore': ['Dropped']})
        self.assertEqual(code, 0, err)
        self.assertNotIn('WARNING: stale source-omission', err)

    def test_stale_source_omission_entry_warns_without_failing(self):
        with mock.patch.object(checker, 'SOURCE_OMISSION_ALLOWLIST', {('VectorCore', 'Gone'): 'reviewed: test'}):
            code, out, err = self.run_main(FakeToolchain())
        self.assertEqual(code, 0, err)
        self.assertIn('WARNING: stale source-omission allowlist entry VectorCore.Gone', err)

    def test_allowlist_is_keyed_on_module_and_name(self):
        with mock.patch.object(checker, 'SOURCE_OMISSION_ALLOWLIST', {('VectorCoreBenchmarking', 'Dropped'): 'x'}):
            code, _, err = self.run_main(FakeToolchain(), extra_source={'VectorCore': ['Dropped']})
        self.assertEqual(code, 2)

    def test_shipped_allowlist_documents_vectorcoreversion(self):
        reason = checker.SOURCE_OMISSION_ALLOWLIST[('VectorCore', 'VectorCoreVersion')]
        self.assertIn('<Module>Version', reason)

    def test_malformed_allowlist_exits_two(self):
        path = self.root / 'bad.json'
        path.write_text(json.dumps([{'name': 'Logger', 'module': 'os'}]))
        code, _, err = self.run_main(FakeToolchain(), allowlist=path)
        self.assertEqual(code, 2)
        self.assertIn('entries need name, module and reason', err)


class RunBoundaryTests(unittest.TestCase):
    """checker.run converts subprocess failures to ToolError; main maps ToolError to exit 2."""

    def test_timeout_becomes_tool_error(self):
        with mock.patch.object(checker.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['x'], 3)):
            with self.assertRaisesRegex(checker.ToolError, 'timed out after 3s'):
                checker.run(['x'], 3)

    def test_oserror_becomes_tool_error(self):
        with mock.patch.object(checker.subprocess, 'run', side_effect=FileNotFoundError('no such tool')):
            with self.assertRaisesRegex(checker.ToolError, 'cannot run x'):
                checker.run(['x'], 3)

    def test_run_passes_the_timeout_to_subprocess(self):
        with mock.patch.object(checker.subprocess, 'run', return_value=completed()) as patched:
            checker.run(['x'], 7)
        self.assertEqual(patched.call_args.kwargs['timeout'], 7)

    def test_main_exits_two_on_timeout_and_on_oserror(self):
        for failure in [subprocess.TimeoutExpired(['xcrun'], 60), OSError('boom')]:
            with self.subTest(failure=type(failure).__name__):
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.object(checker.subprocess, 'run', side_effect=failure), \
                        contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = checker.main(['--allowlist', str(CHECKER.parent / 'sdk_name_collision_allowlist.json')])
                self.assertEqual(code, 2)
                self.assertIn('error:', err.getvalue())


class AllowlistTests(CheckerTestCase):
    def test_stale_entry_warns_and_does_not_fail(self):
        allowlist = self.allowlist(('Logger', 'os'))
        code, out, _ = self.run_main(FakeToolchain(), allowlist=allowlist)
        self.assertEqual(code, 0)
        self.assertIn('WARNING: stale allowlist entry Logger (os) no longer collides', out)
        self.assertIn('OK: no unreviewed SDK name collisions', out)

    def test_used_entry_is_not_stale(self):
        allowlist = self.allowlist(('Logger', 'os'))
        code, out, _ = self.run_main(FakeToolchain(ours={'VectorCore': ['Logger']}), allowlist=allowlist)
        self.assertEqual(code, 0)
        self.assertIn('allowed: VectorCore.Logger == os.Logger', out)
        self.assertNotIn('WARNING', out)

    def test_entry_is_keyed_on_name_and_sdk_module(self):
        # Logger is allowed for os only; a Foundation.Logger collision is a different key.
        allowlist = self.allowlist(('Logger', 'os'))
        fake = FakeToolchain(ours={'VectorCore': ['Logger']}, sdk={'Foundation': ['Logger']})
        code, out, _ = self.run_main(fake, allowlist=allowlist)
        self.assertEqual(code, 1)
        self.assertIn('COLLISION: VectorCore.Logger == Foundation.Logger', out)
        self.assertIn('allowed: VectorCore.Logger == os.Logger', out)

    def test_entry_for_another_name_does_not_cover_a_collision(self):
        allowlist = self.allowlist(('Other', 'os'))
        code, out, _ = self.run_main(FakeToolchain(ours={'VectorCore': ['Logger']}), allowlist=allowlist)
        self.assertEqual(code, 1)
        self.assertIn('COLLISION: VectorCore.Logger == os.Logger', out)
        self.assertIn('WARNING: stale allowlist entry Other (os)', out)

    def test_one_entry_covers_both_package_modules(self):
        allowlist = self.allowlist(('Logger', 'os'))
        fake = FakeToolchain(ours={'VectorCore': ['Logger'], 'VectorCoreBenchmarking': ['Logger']})
        code, out, _ = self.run_main(fake, allowlist=allowlist)
        self.assertEqual(code, 0)
        self.assertIn('allowed: VectorCore.Logger == os.Logger', out)
        self.assertIn('allowed: VectorCoreBenchmarking.Logger == os.Logger', out)

    def test_shipped_allowlist_is_well_formed(self):
        entries = checker.load_allowlist(CHECKER.parent / 'sdk_name_collision_allowlist.json')
        self.assertTrue(all(isinstance(key, tuple) and len(key) == 2 for key in entries))


class SourceScanTests(CheckerTestCase):
    def scan(self, text):
        directory = self.root / 'Sources' / 'M'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'F.swift').write_text(text)
        return checker.source_public_names(directory)

    def test_top_level_public_and_open_declarations_of_every_kind(self):
        text = ('public struct S {}\npublic final class C {}\nopen class O {}\npublic enum E {}\n'
                'public protocol P {}\npublic typealias T = Int\npublic actor A {}\n'
                '@frozen public struct F {}\n@available(macOS 14, *) public struct V {}\npublic indirect enum I {}\n')
        self.assertEqual(self.scan(text), {'S', 'C', 'O', 'E', 'P', 'T', 'A', 'F', 'V', 'I'})

    def test_nested_internal_and_package_declarations_are_not_top_level_public(self):
        text = ('extension X {\n    public struct Nested {}\n}\ninternal struct I {}\nstruct J {}\n'
                'package struct K {}\nprivate struct L {}\npublic func f() {}\npublic let v = 1\n')
        self.assertEqual(self.scan(text), set())

    def test_scan_recurses_into_subdirectories(self):
        sub = self.root / 'Sources' / 'M' / 'Deep' / 'er'
        sub.mkdir(parents=True)
        (sub / 'G.swift').write_text('public struct Deep {}\n')
        self.assertEqual(checker.source_public_names(self.root / 'Sources' / 'M'), {'Deep'})


class SymbolGraphFilterTests(CheckerTestCase):
    def names(self, payload):
        path = self.root / 'g.json'
        path.write_text(json.dumps(payload))
        return checker.public_top_level_names(path)

    def test_only_public_top_level_type_kinds_count(self):
        payload = {'symbols': [
            {'kind': {'identifier': 'swift.struct'}, 'accessLevel': 'public', 'pathComponents': ['Keep']},
            {'kind': {'identifier': 'swift.class'}, 'accessLevel': 'open', 'pathComponents': ['KeepOpen']},
            {'kind': {'identifier': 'swift.struct'}, 'accessLevel': 'internal', 'pathComponents': ['Hidden']},
            {'kind': {'identifier': 'swift.struct'}, 'accessLevel': 'public', 'pathComponents': ['Outer', 'Nested']},
            {'kind': {'identifier': 'swift.func'}, 'accessLevel': 'public', 'pathComponents': ['free']},
        ]}
        self.assertEqual(self.names(payload), {'Keep', 'KeepOpen'})


if __name__ == '__main__':
    unittest.main()
