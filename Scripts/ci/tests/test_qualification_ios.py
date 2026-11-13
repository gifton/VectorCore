"""App launch is insufficient; require bound, complete numerical execution."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[3]


class IOSQualificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('qualification_ios'), 'iOS qualification harness missing')
        self.mod = importlib.import_module('qualification_ios')
        self.invocation = '38d12cec-bc01-44b4-82e5-39e92adb39ce'
        self.binary = '1' * 64
        self.runtime = {'version': '27.0', 'description': 'Version 27.0 (Build 24A434)'}
        self.case = {'id': 'by-value', 'format': 'buffer-value', 'role': 'supported',
                     'correct': {'first': '3f800000'}, 'observed': {'first': '3f800000'}}
        self.begin = {'invocationId': self.invocation, 'caseId': self.case['id'],
                      'binarySHA256': self.binary, 'runtime': self.runtime,
                      'platform': 'ios-simulator', 'architecture': 'arm64'}
        self.end = {'invocationId': self.invocation, 'caseId': self.case['id'], 'numericalExit': 0}
        self.body = 'BUFFER_VALUE first observed=3f800000\nBUFFER_VALUE_COMPLETE count=1 failures=0\n'

    def output(self, *, begin=None, body=None, end=None):
        return ('VC_IOS_BEGIN ' + json.dumps(self.begin if begin is None else begin) + '\n'
                + (self.body if body is None else body)
                + 'VC_IOS_END ' + json.dumps(self.end if end is None else end) + '\n')

    def parse(self, text):
        return self.mod.parse_app_result(text, self.invocation, self.binary, self.runtime,
                                         'ios-simulator', self.case)

    def test_explicit_destinations_keep_device_and_simulator_separate(self):
        simulator = self.mod.destination_config('platform=iOS Simulator,id=' + self.invocation)
        physical = self.mod.destination_config('platform=iOS,id=00008140-001234567890001E')
        self.assertEqual(simulator['platform'], 'ios-simulator')
        self.assertEqual(simulator['sdk'], 'iphonesimulator')
        self.assertEqual(simulator['target'], 'arm64-apple-ios17.0-simulator')
        self.assertEqual(physical['platform'], 'ios-device')
        self.assertEqual(physical['sdk'], 'iphoneos')
        self.assertEqual(physical['target'], 'arm64-apple-ios17.0')
        for destination in ('', 'booted', 'generic/platform=iOS', 'platform=macOS,id=' + self.invocation,
                            'platform=iOS Simulator,name=iPhone', 'platform=iOS,id=a,id=b'):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.mod.destination_config(destination)

    def test_producer_and_client_modes_are_independent(self):
        for producer in ('debug', 'release'):
            for client in ('debug', 'release'):
                settings = self.mod.configurations(producer, client)
                self.assertEqual(settings['producer'], producer)
                self.assertEqual(settings['clientFlag'], '-O' if client == 'release' else '-Onone')
        for mode in ('Release', '-Ounchecked', ''):
            with self.assertRaises(ValueError):
                self.mod.configurations('release', mode)

    def test_each_execution_gets_a_fresh_invocation(self):
        ids = {self.mod.new_invocation() for _ in range(20)}
        self.assertEqual(len(ids), 20)
        self.assertTrue(all(len(value) == 36 for value in ids))

    def test_complete_app_observation_preserves_numerical_result(self):
        result = self.parse(self.output())
        self.assertTrue(result['numericalCorrect'])
        self.assertEqual(result['observations'], {'first': '3f800000'})
        self.assertEqual(result['numericalExit'], 0)

    def test_stale_invocation_wrong_binary_runtime_or_platform_rejects(self):
        changes = {'invocationId': 'old-run', 'binarySHA256': '2' * 64, 'caseId': 'another-case',
                   'runtime': {'version': '26.0'}, 'platform': 'ios-device', 'architecture': 'x86_64'}
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse(self.output(begin=dict(self.begin, **{field: value})))
        with self.assertRaises(ValueError):
            self.parse(self.output(end=dict(self.end, invocationId='old-run')))

    def test_launch_success_without_test_body_or_completion_rejects(self):
        for text in ('', 'Application launched successfully\n', self.output(body=''),
                     self.output().split('VC_IOS_END ')[0], self.output() + self.output()):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_unknown_duplicate_and_truncated_numerical_rows_reject(self):
        for body in (self.body.replace('first', 'unknown'), self.body.splitlines()[0] + '\n' + self.body,
                     self.body.split('BUFFER_VALUE_COMPLETE')[0]):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.parse(self.output(body=body))

    def test_omitted_write_keeps_incorrect_value_and_exit(self):
        self.case.update(role='negative-control', observed={'first': '40000000'})
        body = 'BUFFER_VALUE first observed=40000000\nBUFFER_VALUE_COMPLETE count=1 failures=1\n'
        result = self.parse(self.output(body=body, end=dict(self.end, numericalExit=1)))
        self.assertFalse(result['numericalCorrect'])
        self.assertEqual(result['numericalExit'], 1)
        self.assertEqual(result['failures'], ['first'])

    def test_main_adapters_preserve_every_helper_body(self):
        paths = ('Scripts/ci/fixtures/compile-contracts/BufferValueConsumer.swift',
                 'Scripts/ci/fixtures/compile-contracts/BufferInoutExposure.swift',
                 'Docs/reviews/2026-11-06-workaround-qualification/ResidualExposure.swift')
        for name in paths:
            original = (ROOT / name).read_text()
            adapted = self.mod.adapt_fixture(original, 'main')
            prefix = original.split('@main', 1)[0]
            with self.subTest(name=name):
                self.assertEqual(adapted[:len(prefix)], prefix)
                self.assertNotIn('@main', adapted)
                self.assertIn('public static func run() -> Int32', adapted)
                self.assertIn('BUFFER_VALUE_COMPLETE', adapted)
                self.mod.verify_adapter(original, adapted, 'main')
                with self.assertRaises(ValueError):
                    self.mod.verify_adapter(original, adapted.replace('@inline(never)', '@inline(__always)', 1), 'main')

    def test_standalone_adapters_preserve_original_helpers_and_print(self):
        paths = [ROOT / 'Docs/reviews/2026-11-04-release-phase-one/compiler/probes' / name
                 for name in ('ByValueReadAfter.swift', 'CopyInoutReturnInside.swift', 'NoWriteNegative.swift',
                              'NonLiteralReadAfter.swift', 'OutOfLineRead.swift')]
        paths += [ROOT / 'Docs/reviews/2026-11-01-compiler-qualification/probes' / name
                  for name in ('ByValueReturnInside.swift', 'LostStoreObserved.swift', 'ReturnInsideInout.swift')]
        for path in paths:
            original = path.read_text()
            adapted = self.mod.adapt_fixture(original, 'standalone')
            prefix, body = original.split('let observed = ', 1)
            with self.subTest(name=path.name):
                self.assertEqual(adapted[:len(prefix)], prefix)
                self.assertIn('let observed = ' + body, adapted)
                self.mod.verify_adapter(original, adapted, 'standalone')
                with self.assertRaises(ValueError):
                    self.mod.verify_adapter(original, adapted + '\n// changed\n', 'standalone')

    def test_xcode_build_requires_both_modules_and_one_app_link(self):
        text = ('    builtin-SwiftDriver -- /Xcode/swiftc -module-name QualificationHost -Onone\n'
                '    builtin-SwiftDriver -- /Xcode/swiftc -module-name BufferValueWitness -Onone\n'
                '    /Xcode/clang -target arm64-apple-ios17.0-simulator -o /out/QualificationHost.app/QualificationHost\n'
                '    /Xcode/libtool -static -o /out/libBufferValueWitness.a\n')
        commands = self.mod.xcode_commands(text)
        self.assertEqual(set(commands['swift']), {'QualificationHost', 'BufferValueWitness'})
        self.assertEqual(commands['link'][0], '/Xcode/clang')
        for invalid in (text.replace('BufferValueWitness', 'Other'), text + text,
                        text.split('    /Xcode/clang')[0], text.replace('-static', '-dynamic')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.mod.xcode_commands(invalid)

    def test_xcode_flags_bind_ios_sdk_target_and_optimization(self):
        identity = {'target': 'arm64-apple-ios17.0-simulator', 'swiftVersion': '6', 'sdk': '/Xcode/iOS.sdk'}
        args = ['/Xcode/swiftc', '-sdk', '/Xcode/iOS.sdk', '-target', identity['target'],
                '-swift-version', '6', '-O', '-I', '/producer/modules']
        self.mod.verify_xcode_swift_flags(args, 'release', identity)
        for old, new in (('arm64-apple-ios17.0-simulator', 'arm64-apple-ios17.0'),
                         ('/Xcode/iOS.sdk', '/Xcode/macOS.sdk'), ('-O', '-Onone'), ('6', '5')):
            altered = [new if arg == old else arg for arg in args]
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.mod.verify_xcode_swift_flags(altered, 'release', identity)
        with self.assertRaises(ValueError):
            self.mod.verify_xcode_swift_flags(args + ['-Ounchecked'], 'release', identity)

    def test_collection_never_reuses_an_existing_output_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                self.mod.collect_ios(ROOT, Path('/not-a-toolchain'),
                    'platform=iOS Simulator,id=' + self.invocation, Path(directory), 'release', 'debug')

    def test_matrix_collection_binds_four_cells_and_rejects_missing_or_mixed_identity(self):
        import copy
        cells = {}
        for producer in ('debug', 'release'):
            for client in ('debug', 'release'):
                key = producer + '-' + client
                cells[key] = {'schemaVersion': 1, 'collectionComplete': True, 'invocationId': self.mod.new_invocation(),
                    'identity': self.begin, 'sourceCommit': 'a' * 40, 'sourceManifest': '/evidence/' + key + '/manifest.json',
                    'commands': {'build': '/evidence/' + key + '/build.json', 'install': '/evidence/' + key + '/install.json',
                                 'launch': '/evidence/' + key + '/launch.json'},
                    'checks': [], 'artifactFiles': [],
                    'producers': {producer: {'buildCommand': 'build', 'objects': ['/evidence/' + key + '/producer.o']}},
                    'cases': [{'id': 'by-value', 'producer': producer, 'client': client, 'command': 'launch',
                               'compileCommand': 'build', 'app': {'caseId': 'by-value', 'installCommand': 'install'}}]}
        merged = self.mod.merge_cells(cells, dict.fromkeys(cells, '1' * 64))
        self.assertEqual(len(merged['cases']), 4)
        self.assertEqual(len(merged['commands']), 12)
        self.assertEqual(merged['cases'][0]['app']['caseId'], 'by-value')
        self.assertEqual(merged['cases'][0]['id'], 'debug-debug/by-value')
        for invalid in ({k: v for k, v in cells.items() if k != 'debug-debug'}, copy.deepcopy(cells)):
            if len(invalid) == 4:
                invalid['debug-debug']['identity']['architecture'] = 'x86_64'
                # deepcopy preserves the shared identity object: replace just this cell.
                invalid['debug-debug']['identity'] = dict(self.begin, architecture='other')
            with self.assertRaises(ValueError):
                self.mod.merge_cells(invalid, dict.fromkeys(invalid, '1' * 64))
        digests = dict.fromkeys(cells, '1' * 64)
        digests['release-debug'] = '2' * 64
        with self.assertRaises(ValueError):
            self.mod.merge_cells(cells, digests)


class IOSBindingTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from ios_qualification_fixture import IOSCaptureFixture
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = IOSCaptureFixture(self.temporary.name)
        baseline = self.check()
        self.assertEqual(baseline['vectorCoreQualification'], 'accepted', baseline['problems'])

    def check(self):
        from compiler_qualification import check
        f = self.fixture
        return check(f.profile_path, f.evidence, f.source)

    def test_shared_checker_accepts_bound_app_results_and_preserves_numerical_failures(self):
        result = self.check()
        self.assertEqual(result['vectorCoreQualification'], 'accepted', result['problems'])
        self.assertEqual(result['compilerProfile'], 'known-failing')
        self.assertFalse(result['releaseAdmitted'])
        self.assertEqual(result['diagnosticFailures'], {'compiler': ['first']})
        self.assertFalse(result['observations']['omitted']['numericalCorrect'])

    def test_repinning_a_missing_app_body_does_not_turn_launch_success_into_acceptance(self):
        f = self.fixture
        Path(f.case_specs[0]['app']['result']).write_text('Application launched successfully\n')
        f.save(repin=True)
        self.assertEqual(self.check()['vectorCoreQualification'], 'rejected')

    def test_producer_target_and_client_mode_are_both_checked_after_repinning(self):
        f = self.fixture
        for command, old, new in (('producer-build', 'arm64-apple-ios17.0-simulator', 'arm64-apple-ios17.0'),
                                  ('app-build', 'Release', 'Debug')):
            original = list(f.commands[command]['argv'])
            altered = [new if v == old else v for v in original]
            f.commands[command]['argv'] = altered
            f.alter_receipt(command, argv=altered)
            f.save(repin=True)
            with self.subTest(command=command):
                self.assertEqual(self.check()['vectorCoreQualification'], 'rejected')
            f.commands[command]['argv'] = original
            f.alter_receipt(command, argv=original)

    def test_missing_or_substituted_link_objects_reject_even_if_repinned(self):
        f = self.fixture
        f.link.remove(str(f.object))
        f.alter_stdout('app-build', f.build_log())
        f.save(repin=True)
        self.assertEqual(self.check()['vectorCoreQualification'], 'rejected')

    def test_lto_output_is_not_an_input_but_forwarded_extra_object_rejects(self):
        f = self.fixture
        f.link += ['-Xlinker', '-object_path_lto', '-Xlinker', str(f.evidence / 'reserved_lto.o')]
        f.alter_stdout('app-build', f.build_log())
        f.save(repin=True)
        result = self.check()
        self.assertEqual(result['vectorCoreQualification'], 'accepted', result['problems'])
        foreign = f.evidence / 'foreign.o'
        foreign.write_bytes(b'unrecorded linked object')
        f.link += ['-Xlinker', str(foreign)]
        f.alter_stdout('app-build', f.build_log())
        f.save(repin=True)
        self.assertEqual(self.check()['vectorCoreQualification'], 'rejected')

    def test_xcode_joined_swift_defines_are_distinct_from_forwarded_clang_defines(self):
        f = self.fixture
        for row in f.case_specs + f.capture_cases:
            row['app']['omittedWrites'] = True
        for args in f.swift_commands.values(): args.append('-DOMIT_WRITES')
        f.alter_stdout('app-build', f.build_log())
        f.save(repin=True)
        result = self.check()
        self.assertEqual(result['vectorCoreQualification'], 'accepted', result['problems'])
        for args in f.swift_commands.values():
            args.remove('-DOMIT_WRITES')
            args += ['-Xcc', '-DOMIT_WRITES']
        f.alter_stdout('app-build', f.build_log())
        f.save(repin=True)
        self.assertEqual(self.check()['vectorCoreQualification'], 'rejected')

    def test_missing_producer_module_search_rejects_even_if_repinned(self):
        f = self.fixture
        args = f.swift_commands['QualificationHost']
        index = args.index(str(f.module.parent))
        del args[index - 1:index + 1]
        f.alter_stdout('app-build', f.build_log())
        f.save(repin=True)
        self.assertEqual(self.check()['vectorCoreQualification'], 'rejected')

    def test_witness_module_substitution_and_changed_adapter_reject(self):
        f = self.fixture
        imported = f.archive.parent / 'BufferValueWitness.swiftmodule/arm64-apple-ios-simulator.swiftmodule'
        original = imported.read_bytes()
        imported.write_bytes(b'unrelated witness module')
        f.save(repin=True)
        self.assertEqual(self.check()['vectorCoreQualification'], 'rejected')
        imported.write_bytes(original)
        f.adapter.write_text(f.adapter.read_text().replace('print("fixture")', 'print("changed")'))
        f.save(repin=True)
        self.assertEqual(self.check()['vectorCoreQualification'], 'rejected')


if __name__ == '__main__':
    unittest.main()
