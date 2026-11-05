"""verify_body accepts both Swift Testing log renderings and rejects mismatches."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_sanitizers import verify_body

TAIL = '\nTest run with 1 test in 1 suite passed after 0.1 seconds.\n'


class VerifyBodyTests(unittest.TestCase):
    function = {'id': 'S/testF', 'displayName': 'testF()'}
    custom = {'id': 'S/testF', 'displayName': 'custom name'}

    def test_function_name_rendering(self):
        verify_body(self.function, 'Test testF() started.\nTest testF() passed after 1s' + TAIL)

    def test_custom_name_old_and_aka_renderings(self):
        verify_body(self.custom, 'Test "custom name" started.\nTest "custom name" passed after 1s' + TAIL)
        verify_body(self.custom, "Test \"custom name\" (aka 'testF()') started.\n"
                    "Test \"custom name\" (aka 'testF()') passed after 1s" + TAIL)

    def test_wrong_aka_missing_pass_and_skips_are_rejected(self):
        for log in ["Test \"custom name\" (aka 'other()') started.\nTest \"custom name\" (aka 'other()') passed" + TAIL,
                    'Test "custom name" started.' + TAIL,
                    'Test "custom name" started.\nTest "custom name" passed' + TAIL + ' skipped',
                    'Test "custom name" started.\nTest "custom name" passed\nTest run with 2 tests in 1 suite passed']:
            with self.subTest(log=log), self.assertRaises(ValueError):
                verify_body(self.custom, log)

    def test_swift_6_0_run_summary(self):
        # Swift 6.0 (Xcode 16.2) omits "in N suites"; run summaries in the PR #56 CI run 3 rendering.
        body = 'Test testF() started.\nTest testF() passed after 1s\n'
        verify_body(self.function, body + '✔ Test run with 1 test passed after 7.797 seconds.\n')
        for tail in ['✔ Test run with 0 tests passed after 0.006 seconds.\n',
                     '✔ Test run with 2 tests passed after 0.1 seconds.\n',
                     '✘ Test run with 1 test failed after 0.013 seconds with 1 issue.\n']:
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                verify_body(self.function, body + tail)

    def test_cancelled_body_is_rejected_even_when_started(self):
        # Swift Testing 6.4 reports a self-cancelling test as "was cancelled"; that is never a pass.
        cancelled = ('Test "custom name" (aka \'testF()\') started.\n'
                     'Test "custom name" (aka \'testF()\') was cancelled after 0.001 seconds.\n' + TAIL)
        plain = 'Test testF() started.\nTest testF() was cancelled after 0.001 seconds.' + TAIL
        for row, log in [(self.custom, cancelled), (self.function, plain)]:
            with self.subTest(row=row['displayName']), self.assertRaises(ValueError):
                verify_body(row, log)


class DirectSanitizerHostTests(unittest.TestCase):
    """Direct host launch preserves the manifest selection and loader environment."""

    def test_direct_host_configuration_preserves_body_filters_and_sanitizer_options(self):
        import tempfile
        import check_sanitizers as driver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            toolchain, platform = root / 'toolchain/usr', root / 'MacOSX.platform'
            host = toolchain / 'libexec/swift/pm/swiftpm-testing-helper'
            swift = toolchain / 'bin/swift'
            for file in (host, swift):
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            for sanitizer, short in [('address', 'asan'), ('thread', 'tsan')]:
                runtime = toolchain / ('lib/clang/21/lib/darwin/libclang_rt.' + short + '_osx_dynamic.dylib')
                runtime.parent.mkdir(parents=True, exist_ok=True)
                runtime.touch()
                original = {'ASAN_OPTIONS': 'detect_leaks=0:halt_on_error=1',
                            'TSAN_OPTIONS': 'halt_on_error=1', 'Q03_TSAN': '1', 'PATH': '/bin'}
                selected_host, selected_runtime, environment = driver.sanitizer_host_environment(
                    swift, platform, sanitizer, original)
                self.assertEqual(selected_host, host)
                self.assertEqual(selected_runtime, runtime)
                self.assertEqual(environment['DYLD_INSERT_LIBRARIES'], str(runtime))
                self.assertEqual(environment['DYLD_PRINT_LIBRARIES'], '1')
                self.assertIn(str(toolchain / 'lib/swift/macosx/testing'), environment['DYLD_LIBRARY_PATH'].split(':'))
                self.assertIn(str(platform / 'Developer/usr/lib'), environment['DYLD_LIBRARY_PATH'].split(':'))
                self.assertEqual(environment['DYLD_FRAMEWORK_PATH'].split(':'),
                                 [str(platform / 'Developer/Library/Frameworks'),
                                  str(platform / 'Developer/Library/PrivateFrameworks')])
                for key, value in original.items():
                    self.assertEqual(environment[key], value)
                self.assertNotIn('DYLD_INSERT_LIBRARIES', original)
                rows = driver.sanitizer_manifest(driver.ROOT)['sanitizers'][sanitizer]
                for row in rows:
                    bundle = root / 'a bundle.xctest/Contents/MacOS/bundle'
                    command = driver.sanitizer_test_command(host, bundle, row['filter'])
                    self.assertEqual(command, [str(host), '--test-bundle-path', str(bundle), '--filter',
                                               row['filter'], str(bundle), '--testing-library', 'swift-testing'])

    def test_selected_framework_roots_replace_inherited_paths_for_each_sanitizer(self):
        import tempfile
        import check_sanitizers as driver
        for sanitizer, short in [('address', 'asan'), ('thread', 'tsan')]:
            with self.subTest(sanitizer=sanitizer), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                swift = root / 'toolchain/usr/bin/swift'
                host = root / 'toolchain/usr/libexec/swift/pm/swiftpm-testing-helper'
                runtime = root / ('toolchain/usr/lib/clang/17/lib/darwin/libclang_rt.' + short + '_osx_dynamic.dylib')
                for file in [swift, host, runtime]:
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.touch()
                platform = root / 'selected platform/MacOSX.platform'
                public = platform / 'Developer/Library/Frameworks'
                private = platform / 'Developer/Library/PrivateFrameworks'
                _, _, environment = driver.sanitizer_host_environment(
                    swift, platform, sanitizer, {'DYLD_FRAMEWORK_PATH': '/unrelated/developer/Frameworks'})
                self.assertEqual(environment['DYLD_FRAMEWORK_PATH'].split(':'), [str(public), str(private)])

    def test_alias_paths_select_one_canonical_runtime_for_each_sanitizer(self):
        import tempfile
        import check_sanitizers as driver
        for sanitizer, short in [('address', 'asan'), ('thread', 'tsan')]:
            for alias_kind in ['version directory', 'runtime file']:
                with self.subTest(sanitizer=sanitizer, alias_kind=alias_kind), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    swift = root / 'usr/bin/swift'
                    swift.parent.mkdir(parents=True)
                    swift.touch()
                    host = root / 'usr/libexec/swift/pm/swiftpm-testing-helper'
                    host.parent.mkdir(parents=True)
                    host.touch()
                    clang = root / 'usr/lib/clang'
                    runtime = clang / ('17/lib/darwin/libclang_rt.' + short + '_osx_dynamic.dylib')
                    runtime.parent.mkdir(parents=True)
                    runtime.write_bytes(b'one runtime image')
                    alias = clang / ('17.0.0/lib/darwin/' + runtime.name)
                    if alias_kind == 'version directory':
                        (clang / '17.0.0').symlink_to('17', target_is_directory=True)
                    else:
                        alias.parent.mkdir(parents=True)
                        alias.symlink_to(runtime)
                    self.assertTrue(alias.samefile(runtime))
                    _, selected, environment = driver.sanitizer_host_environment(swift, root / 'platform', sanitizer, {})
                    self.assertEqual(selected, runtime)
                    self.assertEqual(environment['DYLD_INSERT_LIBRARIES'], str(runtime))

    def test_missing_or_distinct_canonical_runtimes_are_rejected_for_each_sanitizer(self):
        import tempfile
        import check_sanitizers as driver
        for sanitizer, short in [('address', 'asan'), ('thread', 'tsan')]:
            for contents in [(b'first image', b'second image'), (b'identical bytes', b'identical bytes')]:
                with self.subTest(sanitizer=sanitizer, contents=contents), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    swift = root / 'usr/bin/swift'
                    swift.parent.mkdir(parents=True)
                    swift.touch()
                    host = root / 'usr/libexec/swift/pm/swiftpm-testing-helper'
                    host.parent.mkdir(parents=True)
                    host.touch()
                    with self.assertRaisesRegex(ValueError, 'exactly one.*runtime'):
                        driver.sanitizer_host_environment(swift, root / 'platform', sanitizer, {})
                    for version, content in zip(['17', '17.0.0'], contents):
                        runtime = root / ('usr/lib/clang/' + version + '/lib/darwin/libclang_rt.' + short + '_osx_dynamic.dylib')
                        runtime.parent.mkdir(parents=True)
                        runtime.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, 'exactly one.*runtime'):
                        driver.sanitizer_host_environment(swift, root / 'platform', sanitizer, {})


if __name__ == '__main__':
    unittest.main()
