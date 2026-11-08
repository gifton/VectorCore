"""Behavioral contracts for the candidate-specific native evidence checker.

Fixtures are synthetic raw records; no archived code is executed and no toolchain
is required. Each negative changes an observation, process boundary or binding.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'bounded_native_qualification.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load():
    if not SCRIPT.exists():
        raise AssertionError('draft evidence checker is not implemented')
    spec = importlib.util.spec_from_file_location('bounded_native_qualification', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(values, failures=0):
    return ''.join(f'BUFFER_VALUE {key} observed={value}\n' for key, value in values.items()) + f'BUFFER_VALUE_COMPLETE count={len(values)} failures={failures}\n'


def native_log(release=False):
    issue = ('Test literalArrayStoreIsObservedAfterOutOfLineClosure() recorded an issue at '
             'LostStoreCanaryTests.swift:47:9: Expectation failed: observed == 1\n') if release else ''
    ending = 'failed after 1 seconds with 1 issue.' if release else 'passed after 1 seconds.'
    return (issue + 'Suite SIMDStorageMutationTests passed\n' + ' Test disabled skipped.\n' * 18
            + 'Executed 218 tests, with 28 tests skipped and 0 failures (0 unexpected)\n'
            + f'Test run with 1775 tests in 266 suites {ending}\n')


class NativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.q = load()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_complete_raw_rows_accept_known_failure_without_relabeling_values(self):
        expected = {'good': '3f800000', 'known': '40000000'}
        result = self.q.numerical(rows(expected, 1), 1, expected, 1)
        self.assertEqual(result, expected)

    def test_numerical_inventory_and_exit_are_not_inferred_from_green_summary(self):
        expected = {'good': '3f800000', 'known': '40000000'}
        valid = rows(expected, 1)
        mutations = [valid.replace('40000000', '00000000'),
                     valid.replace('BUFFER_VALUE good observed=3f800000\n', ''),
                     'BUFFER_VALUE good observed=3f800000\n' + valid,
                     valid.replace('known', 'unknown'), valid + 'unexpected\n',
                     valid.replace('failures=1', 'failures=0')]
        for text in mutations:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.q.numerical(text, 1, expected, 1)
        for code in [0, -9, 124, True]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                self.q.numerical(valid, code, expected, 1)

    def test_native_known_canary_is_separate_from_other_failures(self):
        self.q.native_suite(native_log(), 0, 'debug')
        self.q.native_suite(native_log(True), 1, 'release')
        for text, code, mode in [
            (native_log(True).replace('LostStoreCanaryTests.swift:47:9', 'OtherTests.swift:47:9'), 1, 'release'),
            (native_log(True) + 'Test other recorded an issue at Other.swift:2:1\n', 1, 'release'),
            (native_log().replace('1775 tests', '1774 tests'), 0, 'debug'),
            (native_log().replace('0 failures', '1 failures'), 0, 'debug'),
            (native_log().replace('18 skipped', '17 skipped').replace(' Test disabled skipped.\n', '', 1), 0, 'debug'),
            (native_log(True), 0, 'release'),
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.q.native_suite(text, code, mode)

    def command(self):
        exe = self.root / 'program'; exe.write_bytes(b'program')
        out = self.root / 'out'; out.write_bytes(b'hello\n')
        err = self.root / 'err'; err.write_bytes(b'')
        record = {'argv': [str(exe)], 'executable': str(exe), 'executableSHA256': digest(b'program'),
                  'status': 'completed', 'timeout': False, 'exit': 0,
                  'stdout': {'path': str(out), 'sha256': digest(b'hello\n'), 'bytes': 6},
                  'stderr': {'path': str(err), 'sha256': digest(b''), 'bytes': 0}}
        path = self.root / 'command.json'; path.write_text(json.dumps(record))
        return path, record

    def test_command_verifies_streams_binary_completion_and_exit(self):
        path, record = self.command()
        self.assertEqual(self.q.command(path, 0)[1], 'hello\n')
        for key, value in [('timeout', True), ('status', 'timed-out'), ('exit', -9), ('exit', False)]:
            bad = dict(record, **{key: value}); path.write_text(json.dumps(bad))
            with self.subTest(key=key), self.assertRaises(ValueError): self.q.command(path, 0)
        path.write_text(json.dumps(record))
        for file in ['out', 'err', 'program']:
            p = self.root/file; original=p.read_bytes();p.write_bytes(original+b'changed')
            with self.subTest(file=file), self.assertRaises(ValueError): self.q.command(path, 0)
            p.write_bytes(original)

    def test_command_cannot_replace_an_earlier_pinned_receipt(self):
        path, record = self.command()
        evidence = self.q.Evidence(self.root, self.root, self.root)
        first = digest(path.read_bytes())
        evidence.hash(path, first)
        # The new record is internally valid: only an otherwise unused field changes.
        record['deadlineSeconds'] = 15
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            evidence.cmd(path, 0)
        self.assertEqual(evidence.verified[str(path.resolve())], first)

    def test_command_cannot_replace_pinned_stream_or_binary_with_matching_new_record(self):
        for kind in ('stdout', 'stderr', 'executable'):
            with self.subTest(kind=kind):
                path, record = self.command()
                evidence = self.q.Evidence(self.root, self.root, self.root)
                file = Path(record['executable'] if kind == 'executable' else record[kind]['path'])
                first = digest(file.read_bytes())
                evidence.hash(file, first)
                file.write_bytes(file.read_bytes() + b'changed')
                if kind == 'executable':
                    record['executableSHA256'] = digest(file.read_bytes())
                else:
                    record[kind]['sha256'] = digest(file.read_bytes())
                    record[kind]['bytes'] = file.stat().st_size
                path.write_text(json.dumps(record))
                # Register the updated receipt after the earlier artifact pin, so a
                # receipt check alone cannot catch the conflicting artifact claim.
                evidence.hash(path, digest(path.read_bytes()))
                with self.assertRaises(ValueError):
                    evidence.cmd(path, 0)
                self.assertEqual(evidence.verified[str(file.resolve())], first)

    def test_hash_registration_uses_first_canonical_path_expectation(self):
        path, _ = self.command()
        evidence = self.q.Evidence(self.root, self.root, self.root)
        first = digest(path.read_bytes())
        evidence.hash(path, first)
        alias = self.root / 'receipt-alias'
        alias.symlink_to(path)
        evidence.hash(alias, first)
        path.write_text('{"status":"changed"}')
        with self.assertRaises(ValueError):
            evidence.hash(alias, digest(path.read_bytes()))
        self.assertEqual(evidence.verified[str(path.resolve())], first)
        self.assertEqual(len(evidence.verified), 1)

    def test_final_recheck_preserves_command_pins_after_successful_read(self):
        path, record = self.command()
        evidence = self.q.Evidence(self.root, self.root, self.root)
        evidence.hash(path, digest(path.read_bytes()))
        self.assertEqual(evidence.cmd(path, 0)[1], 'hello\n')
        self.assertTrue(callable(getattr(evidence, 'recheck', None)), 'final immutable-pin recheck missing')
        evidence.recheck()
        Path(record['stdout']['path']).write_text('changed after valid command\n')
        with self.assertRaises(ValueError):
            evidence.recheck()

    def test_duplicate_json_keys_cannot_overwrite_rejection(self):
        path=self.root/'duplicate.json';path.write_text('{"exit":1,"exit":0}')
        with self.assertRaises(ValueError): self.q.read_json(path)

    def test_compile_flags_reject_unsupported_optimization_and_extra_driver_flags(self):
        argv=['/bin/swiftc','-swift-version','6','-target','arm64-apple-macosx14.0','-sdk','/sdk',
              '-module-cache-path','/cache','-parse-as-library','-O','/probe.swift','-o','/binary']
        self.q.compile_flags(argv, 'release', '/sdk')
        for bad in [argv+['-Ounchecked'], argv+['-Xfrontend','-disable-sil-perf-optzns'],
                    argv+['-target','x86_64-apple-macosx14.0'],
                    [x.replace('arm64-apple-macosx14.0','arm64-apple-ios17.0') for x in argv],
                    [x.replace('-O','-Onone') for x in argv]]:
            with self.subTest(argv=bad), self.assertRaises(ValueError):self.q.compile_flags(bad,'release','/sdk')

    def test_producer_object_inventory_and_module_must_match_link(self):
        objects={'/build/VectorCore.build/a.o':'abc','/build/VectorCoreC.build/b.o':'def',
                 '/build/Modules/VectorCore.swiftmodule':'ghi',
                 '/build/VectorCoreC.build/module.modulemap':'jkl'}
        argv=['swiftc','-I','/build/Modules','-Xcc','-fmodule-map-file=/build/VectorCoreC.build/module.modulemap','/client.swift','/build/VectorCore.build/a.o','/build/VectorCoreC.build/b.o','-o','/consumer']
        self.q.link_inputs(argv,objects,'/build')
        for bad in [argv+['/other/c.o'], argv+['-I','/other/Modules'],
                    [x.replace('/build/VectorCoreC.build/module.modulemap','/other/module.modulemap') for x in argv],[x for x in argv if x!='/build/VectorCore.build/a.o'],
                    [x.replace('/build/Modules','/other/Modules') for x in argv]]:
            with self.subTest(argv=bad),self.assertRaises(ValueError):self.q.link_inputs(bad,objects,'/build')


    def test_exact_cell_inventory_rejects_duplicate_and_unexpected_modes(self):
        wanted={('debug',False),('release',False)}
        cells=[{'mode':'debug','negative':False},{'mode':'release','negative':False}]
        self.q.inventory(cells,('mode','negative'),wanted)
        for bad in [cells+cells[:1],cells[:1],cells+[{'mode':'unchecked','negative':False}]]:
            with self.subTest(cells=bad),self.assertRaises(ValueError):self.q.inventory(bad,('mode','negative'),wanted)

    def test_host_rejects_different_runtime_even_with_same_deployment_target(self):
        host='ProductName:\tmacOS\nProductVersion:\t27.0.1\nBuildVersion:\t26A434\n'
        self.q.Evidence.host(host)
        for bad in [host.replace('27.0.1','14.0'),host.replace('26A434','26A435'),host.replace('macOS','iOS')]:
            with self.subTest(host=bad),self.assertRaises(ValueError):self.q.Evidence.host(bad)

    def test_data_oracle_ignores_helper_side_effects_and_preloaded_modules(self):
        # The checker must read data, never execute a source helper to obtain expectations.
        # A helper that writes a marker catches accidental reintroduction of dynamic imports.
        self.assertTrue(callable(getattr(self.q, 'consumer_oracles', None)), 'data-only oracle loader missing')
        marker = self.root / 'executed'
        helper = self.root / 'consumer_buffer_value.py'
        helper.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("executed")\nraise RuntimeError("source helper executed")\n')
        profile = json.loads(SCRIPT.with_name('bounded_native_qualification_profile.json').read_text())
        import types
        preloaded = types.ModuleType('consumer_buffer_value')
        def sentinel_oracle(**kwargs):
            marker.write_text('executed preloaded helper')
            raise RuntimeError('preloaded helper executed')
        preloaded.expected_observations = sentinel_oracle
        old = sys.modules.get('consumer_buffer_value')
        sys.path.insert(0, str(self.root))
        try:
            for module in (None, preloaded):
                if module is None: sys.modules.pop('consumer_buffer_value', None)
                else: sys.modules['consumer_buffer_value'] = module
                positive, omitted = self.q.consumer_oracles(profile)
                self.assertEqual(positive['concrete.float.singleton'], '3f800000')
                self.assertEqual(omitted['concrete.float.singleton'], '40000000')
                self.assertEqual(len(positive), 24)
                self.assertEqual(set(positive), set(omitted))
                self.assertFalse(marker.exists())
        finally:
            sys.path.pop(0)
            if old is None: sys.modules.pop('consumer_buffer_value', None)
            else: sys.modules['consumer_buffer_value'] = old

    def test_altered_helper_dependency_is_rejected_without_executing_it(self):
        marker = self.root / 'dependency-executed'
        helper = self.root / 'consumer_smoke.py'
        helper.write_text('pass\n')
        original = digest(helper.read_bytes())
        helper.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("executed")\n')
        evidence = self.q.Evidence(self.root, self.root, self.root)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evidence.mapping({str(helper): original})
        self.assertFalse(marker.exists())

    def test_cli_rejects_incomplete_evidence_and_never_claims_release(self):
        output=self.root/'result.json'
        p=subprocess.run([sys.executable,str(SCRIPT),'--migration-root',str(self.root/'missing'),
                          '--residual-root',str(self.root/'missing2'),'--source-root',str(self.root),
                          '--output',str(output)],capture_output=True,text=True,timeout=15)
        self.assertEqual(p.returncode,1,p.stderr)
        result=json.loads(output.read_text())
        self.assertEqual(result['nativeEvidence'],'rejected')
        self.assertEqual(result['compilerProfile'],'unverified')
        self.assertIs(result['supportPolicyAdopted'],False)
        self.assertIs(result['releaseAdmitted'],False)
        self.assertTrue(result['remainingGates'])
        before=output.read_bytes()
        p=subprocess.run([sys.executable,str(SCRIPT),'--migration-root',str(self.root),
                          '--residual-root',str(self.root),'--source-root',str(self.root),
                          '--output',str(output)],capture_output=True,text=True,timeout=15)
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(output.read_bytes(),before)


if __name__=='__main__':unittest.main()
