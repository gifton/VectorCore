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


class SanitizerConfigurationTests(unittest.TestCase):
    def test_configuration_agrees_across_build_discovery_execution_and_evidence(self):
        import json
        import tempfile
        from unittest.mock import patch
        import check_sanitizers as driver
        for sanitizer in ['address', 'thread']:
            for selection, configuration in [([], 'debug'), (['--configuration', 'debug'], 'debug'),
                                              (['--configuration', 'release'], 'release')]:
                with self.subTest(sanitizer=sanitizer, selection=selection), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = root / 'output'
                    bin_path = root / configuration
                    host, runtime = root / 'helper', root / 'runtime'
                    host.write_bytes(b'host')
                    runtime.write_bytes(b'runtime')
                    rows = driver.sanitizer_manifest(driver.ROOT)['sanitizers'][sanitizer]
                    targets = {driver.target_of_filter(row['filter']) for row in rows}
                    bundles = {target: bin_path / (target + '.xctest') for target in targets}
                    for bundle in bundles.values():
                        bundle.parent.mkdir(parents=True, exist_ok=True)
                        bundle.write_bytes(configuration.encode())
                    commands, discovered = [], []

                    def execute(argv, *args, **kwargs):
                        commands.append(list(map(str, argv)))
                        if argv[0] == 'xcrun':
                            return {}, '/selected/MacOSX.platform'
                        if str(argv[0]) == str(host):
                            selector = argv[argv.index('--filter') + 1]
                            row = next(row for row in rows if row['filter'] == selector)
                            label = row['displayName']
                            rendered = label if label.endswith('()') else '"' + label + '"'
                            return {}, ('Test ' + rendered + ' started.\nTest ' + rendered + ' passed' + TAIL
                                        + str(host) + str(argv[2]) + str(runtime))
                        return {}, ''

                    def discover(swift, flags, *args, **kwargs):
                        discovered.extend(flags)
                        return bin_path

                    argv = ['check_sanitizers.py', '--sanitizer', sanitizer, '--scratch-root', str(root / 'scratch'),
                            '--output', str(output), '--jobs', '2', *selection]
                    with patch.object(sys, 'argv', argv), patch.object(driver, 'find_swift', return_value='/swift'), \
                            patch.object(driver, 'run', side_effect=execute), \
                            patch.object(driver, 'resolve_bin_path', side_effect=discover), \
                            patch.object(driver, 'find_test_bundles', return_value=bundles) as find_bundles, \
                            patch.object(driver, 'retain_artifacts', return_value=[]), \
                            patch.object(driver, 'sanitizer_host_environment', return_value=(host, runtime, {})), \
                            patch.object(driver, 'verify_q03', return_value={'verified': True}):
                        driver.main()
                    build = commands[0]
                    self.assertEqual(build[build.index('--configuration') + 1], configuration)
                    self.assertEqual(discovered[discovered.index('--configuration') + 1], configuration)
                    self.assertEqual(build[3:-1], list(map(str, discovered)))
                    self.assertEqual('-enable-testing' in build, configuration == 'release')
                    if configuration == 'release':
                        self.assertEqual(build[build.index('-enable-testing') - 1], '-Xswiftc')
                    find_bundles.assert_called_once()
                    self.assertEqual(find_bundles.call_args.args[0], bin_path)
                    bodies = [argv for argv in commands if argv[0] == str(host)]
                    self.assertEqual(len(bodies), len(rows))
                    for argv in bodies:
                        self.assertIn(Path(argv[2]), bundles.values())
                        self.assertEqual(argv[2], argv[-3])
                    result = json.loads((output / 'result.json').read_text())
                    self.assertEqual(result['configuration'], configuration)
                    self.assertEqual(result['binPath'], str(bin_path))
                    self.assertEqual({v['path'] for v in result['loadedBundles'].values()},
                                     {str(bundle) for bundle in bundles.values()})

    def test_invalid_configuration_rejects_before_any_build_or_output(self):
        import tempfile
        from unittest.mock import patch
        import check_sanitizers as driver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['check_sanitizers.py', '--sanitizer', 'address', '--scratch-root', str(root / 'scratch'),
                    '--output', str(root / 'output'), '--configuration', 'Release']
            with patch.object(sys, 'argv', argv), patch.object(driver, 'find_swift') as lookup:
                with self.assertRaises(SystemExit) as error:
                    driver.main()
            self.assertEqual(error.exception.code, 2)
            lookup.assert_not_called()
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
