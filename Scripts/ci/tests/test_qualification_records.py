"""Integrity failures must reject captures before numerical assessment."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification_fixture import CaptureFixture, sha, write_json


class QualificationRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = CaptureFixture(self.tmp.name)
        self.assertIsNotNone(importlib.util.find_spec('qualification_records'),
                             'candidate-bound record verifier is not implemented')
        self.records = importlib.import_module('qualification_records')

    def verify(self):
        f = self.fixture
        return self.records.verify_capture(f.profile, f.evidence, f.source)

    def test_complete_synthetic_capture_is_verified(self):
        result = self.verify()
        self.assertEqual(set(result['commands']), {'compile', 'supported', 'omitted', 'compiler'})
        self.assertEqual(result['capture']['invocationId'], 'fixture-run-01')

    def test_duplicate_json_keys_reject(self):
        f = self.fixture
        f.capture_path.write_text('{"schemaVersion": 0, "schemaVersion": 1}')
        f.profile['pins'][str(f.capture_path)] = sha(f.capture_path)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.verify()

    def test_unpinned_command_receipt_rejects(self):
        del self.fixture.profile['pins'][self.fixture.commands['supported']['receipt']]
        with self.assertRaises(ValueError):
            self.verify()

    def test_mutated_valid_receipt_cannot_replace_original_pin(self):
        f = self.fixture
        f.alter_receipt('supported', elapsedSeconds=0.2)
        with self.assertRaisesRegex(ValueError, 'hash|digest'):
            self.verify()

    def test_conflicting_pin_through_canonical_alias_rejects(self):
        f = self.fixture
        alias = f.evidence / 'compiler-alias'
        alias.symlink_to(f.compiler)
        f.profile['pins'][str(alias)] = '0' * 64
        with self.assertRaises(ValueError):
            self.verify()

    def test_captured_root_alias_maps_before_canonical_pin_lookup(self):
        import shutil
        f = self.fixture
        alias = f.root / 'captured-alias'
        alias.symlink_to(f.evidence, target_is_directory=True)
        archived = f.root / 'archive'
        shutil.copytree(f.evidence, archived)
        f.profile['captureRoots']['evidence'] = str(alias)
        pins = self.records.Pins(f.profile, archived, f.source)
        self.assertEqual(pins.resolve(str(f.binary)), archived.resolve() / f.binary.name)

    def test_changed_stream_and_matching_receipt_still_reject_old_pins(self):
        self.fixture.alter_stdout('supported', 'forged\n')
        with self.assertRaises(ValueError):
            self.verify()

    def test_input_changed_after_initial_hash_rejects(self):
        f = self.fixture
        real = self.records.read_json
        changed = False
        def reading(path):
            nonlocal changed
            value = real(path)
            if Path(path).resolve() == f.capture_path.resolve() and not changed:
                f.object.write_bytes(b'changed during assessment\n')
                changed = True
            return value
        with patch.object(self.records, 'read_json', side_effect=reading):
            with self.assertRaises(ValueError):
                self.verify()
        self.assertTrue(changed)

    def test_added_compiled_source_rejects(self):
        (self.fixture.source / 'Sources/Unrecorded.swift').write_text('let other = 1\n')
        with self.assertRaises(ValueError):
            self.verify()

    def test_wrong_source_sdk_runtime_or_flags_rejects(self):
        f = self.fixture
        for field, value in [('target', 'x86_64-apple-macosx14.0'),
                             ('sdkSettingsSHA256', '0' * 64),
                             ('runtime', {'version': '14.0', 'build': 'other'})]:
            before = f.capture['identity'][field]
            f.capture['identity'][field] = value
            f.save(repin=True)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify()
            f.capture['identity'][field] = before
        f.save(repin=True)
        record = json.loads(Path(f.commands['compile']['receipt']).read_text())
        record['argv'].append('-Ounchecked')
        write_json(f.commands['compile']['receipt'], record)
        f.save(repin=True)
        with self.assertRaises(ValueError):
            self.verify()

    def test_producer_object_and_module_must_match_compile_binding(self):
        f = self.fixture
        for key in ['objects', 'modules']:
            original = f.capture['producers']['release'][key]
            f.capture['producers']['release'][key] = []
            f.save(repin=True)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify()
            f.capture['producers']['release'][key] = original

    def test_incomplete_or_timeout_record_rejects(self):
        for fields in [{'status': 'running'}, {'timeout': True}, {'exit': None}, {'exit': True}]:
            with self.subTest(fields=fields):
                f = self.fixture
                path = Path(f.commands['supported']['receipt'])
                before = path.read_bytes()
                f.alter_receipt('supported', **fields)
                f.save(repin=True)
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(before)

    def test_profile_paths_cannot_escape_remapped_roots(self):
        f = self.fixture
        f.profile['pins'][str(f.evidence / '..' / 'profile.json')] = sha(f.profile_path)
        with self.assertRaises(ValueError):
            self.verify()

    def test_evidence_helper_is_not_executed(self):
        f = self.fixture
        marker = f.evidence / 'executed'
        (f.evidence / 'consumer_oracles.py').write_text(
            'from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n')
        f.save(repin=True)
        self.verify()
        self.assertFalse(marker.exists())

    def rewrite_compile(self, argv):
        f = self.fixture
        f.commands['compile']['argv'] = argv
        f.alter_receipt('compile', argv=argv)
        f.save(repin=True)

    def test_standalone_probe_binds_its_binary_without_library_products(self):
        f = self.fixture
        producer = {'kind': 'standalone', 'objects': [], 'modules': []}
        f.profile['producers']['release'] = producer
        f.capture['producers']['release'] = producer.copy()
        argv = list(f.commands['compile']['argv'])
        argv.remove(str(f.object))
        index = argv.index('-I')
        del argv[index:index + 2]
        self.rewrite_compile(argv)
        self.verify()

    def test_reviewed_flags_still_must_match_adopted_mode_and_target(self):
        f = self.fixture
        original = list(f.commands['compile']['argv'])
        for flag in ['-Ounchecked', '-Osize', '-Onone']:
            argv = [flag if v == '-O' else v for v in original]
            self.rewrite_compile(argv)
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.verify()
        argv = list(original)
        argv[argv.index('-target') + 1] = 'x86_64-apple-macosx14.0'
        self.rewrite_compile(argv)
        with self.assertRaises(ValueError):
            self.verify()

    def test_even_repinned_wrong_object_or_missing_module_search_rejects(self):
        f = self.fixture
        argv = list(f.commands['compile']['argv'])
        argv.remove(str(f.object))
        self.rewrite_compile(argv)
        with self.assertRaises(ValueError):
            self.verify()
        argv.append(str(f.object))
        index = argv.index('-I')
        del argv[index:index + 2]
        self.rewrite_compile(argv)
        with self.assertRaises(ValueError):
            self.verify()

    def test_separate_witness_is_bound_to_its_compile_and_link(self):
        f = self.fixture
        obj, module = f.evidence / 'witness.o', f.evidence / 'Witness.swiftmodule'
        obj.write_bytes(b'witness object')
        module.write_bytes(b'witness module')
        f.add_command('witness-compile', [str(f.compiler), '-O', '-swift-version', '6',
            '-target', f.identity['target'], '-sdk', str(f.sdk.parent), '-emit-object',
            '-emit-module', '-emit-module-path', str(module), '-o', str(obj)], '', 0,
            executable=f.compiler)
        binding = {'command': 'witness-compile', 'object': str(obj), 'module': str(module)}
        for row in f.case_specs + f.capture_cases:
            row['witness'] = dict(binding)
        f.capture['commands']['witness-compile'] = f.commands['witness-compile']['receipt']
        argv = [*f.commands['compile']['argv'], str(obj)]
        self.rewrite_compile(argv)
        self.verify()
        # Keep the profile and receipt consistent: the independently checked
        # witness output must still be the very object passed to the link.
        f.commands['witness-compile']['argv'][-1] = str(f.object)
        f.alter_receipt('witness-compile', argv=f.commands['witness-compile']['argv'])
        f.save(repin=True)
        with self.assertRaises(ValueError):
            self.verify()

    def test_c_modes_incomplete_or_duplicate_cell_inventory_rejects(self):
        self.assertTrue(hasattr(self.records, 'verify_c_build_modes'), 'C capture verifier missing')
        class Data:
            def __init__(self, result): self.result = result
            def json(self, path): return self.result
        cell = {'name': 'debug-c-off', 'configuration': 'debug', 'cKernels': False}
        for cells in ([], [cell], [cell] * 4):
            pins = Data({'schemaVersion': 1, 'marker': 'V02_BUILD_MODES_COMPLETE', 'cells': cells})
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                self.records.verify_c_build_modes({'result': '/evidence/result.json'}, pins, {}, {})


if __name__ == '__main__':
    unittest.main()
