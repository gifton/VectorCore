"""Collection preserves failures; it cannot manufacture qualification."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class NativeCollectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('capture_compiler_qualification'),
                             'fresh native collector is not implemented')
        self.mod = importlib.import_module('capture_compiler_qualification')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        for name in ('Sources', 'Tests', 'Benchmarks', 'Scripts/ci'):
            (self.source / name).mkdir(parents=True)
        (self.source / 'Package.swift').write_text('// package\n')
        (self.source / 'Sources/Test.swift').write_text('let value = 1\n')
        self.swiftc = self.root / 'toolchain/swiftc'
        self.swiftc.parent.mkdir()
        self.swiftc.write_text('compiler fixture')
        self.swiftc.with_name('swift-frontend').write_text('frontend fixture')
        self.sdk = self.root / 'sdk'
        self.sdk.mkdir()
        (self.sdk / 'SDKSettings.json').write_text('{}')
        self.output = self.root / 'evidence'

    def collector(self):
        return self.mod.NativeCapture(self.source, self.swiftc, self.sdk, self.output, 4)

    def patch_phases(self, collector):
        """External compilers are replaced only at the subprocess/phase boundary."""
        names = ('preflight', 'identity', 'original_probes', 'build_producer',
                 'consumers', 'residual', 'build_modes', 'finish_inventory')
        mocks = {}
        for name in names:
            p = patch.object(collector, name)
            mocks[name] = p.start()
            self.addCleanup(p.stop)
        mocks['identity'].return_value = {'platform': 'macos'}
        mocks['build_producer'].side_effect = lambda mode: self.root / mode
        return mocks

    def test_existing_output_is_never_reused(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('original')
        c = self.collector()
        mocks = self.patch_phases(c)
        with self.assertRaises(FileExistsError):
            c.collect()
        mocks['build_producer'].assert_not_called()
        self.assertEqual(marker.read_text(), 'original')

    def test_wrong_frontend_stops_before_any_build(self):
        c = self.collector()
        with patch.object(c, 'command') as command:
            with self.assertRaisesRegex(ValueError, 'frontend'):
                c.collect()
        command.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_all_producer_client_and_control_cells_are_scheduled(self):
        cells = self.mod.consumer_cells()
        self.assertEqual(set(cells), {(p, c, n) for p in ('debug', 'release')
                         for c in ('debug', 'release') for n in (False, True)})
        self.assertEqual(len(cells), 8)
        c = self.collector()
        mocks = self.patch_phases(c)
        result = c.collect()
        self.assertEqual(json.loads(result.read_text())['collectionComplete'], True)
        self.assertEqual([call.args[0] for call in mocks['build_producer'].call_args_list], ['debug', 'release'])
        self.assertEqual([call.args[0] for call in mocks['consumers'].call_args_list], ['debug', 'release'])
        self.assertEqual([call.args[0] for call in mocks['residual'].call_args_list], ['debug', 'release'])
        mocks['build_modes'].assert_called_once()

    def test_changed_or_added_source_invalidates_collection(self):
        for added in (False, True):
            c = self.collector()
            c.output = self.root / ('added' if added else 'changed')
            mocks = self.patch_phases(c)
            def change():
                name = 'Extra.swift' if added else 'Test.swift'
                (self.source / 'Sources' / name).write_text('let changed = 2\n')
            mocks['build_modes'].side_effect = change
            with self.subTest(added=added), self.assertRaisesRegex(ValueError, 'source'):
                c.collect()
            record = json.loads((c.output / 'capture.json').read_text())
            self.assertFalse(record['collectionComplete'])
            self.assertTrue((c.output / 'source-inputs-after.json').exists())

    def test_failed_producer_stops_dependent_consumers(self):
        c = self.collector()
        mocks = self.patch_phases(c)
        mocks['build_producer'].side_effect = RuntimeError('compile failed')
        with self.assertRaisesRegex(RuntimeError, 'compile failed'):
            c.collect()
        mocks['consumers'].assert_not_called()
        self.assertFalse(json.loads((self.output / 'capture.json').read_text())['collectionComplete'])

    def test_failed_compile_never_launches_binary(self):
        c = self.collector()
        with patch.object(c, 'command', side_effect=RuntimeError('compile failed')) as command:
            with self.assertRaisesRegex(RuntimeError, 'compile failed'):
                c.compile_and_run('cell', [str(self.swiftc), '-O'], self.output / 'consumer')
        self.assertEqual(command.call_count, 1)

    def test_absent_binary_is_not_a_completed_compile(self):
        c = self.collector()
        with patch.object(c, 'command', return_value=({'exit': 0}, '')) as command:
            with self.assertRaises(FileNotFoundError):
                c.compile_and_run('cell', [str(self.swiftc), '-O'], self.output / 'consumer')
        self.assertEqual(command.call_count, 1)

    def test_raw_diagnostic_failure_is_retained_without_promotion(self):
        self.output.mkdir()
        binary = self.output / 'consumer'
        binary.write_bytes(b'fixture')
        c = self.collector()
        with patch.object(c, 'command', side_effect=[({'exit': 0}, ''),
                          ({'exit': 1}, 'BUFFER_VALUE first observed=40000000\n')]) as command:
            record, raw = c.compile_and_run('cell', [str(self.swiftc), '-O'], binary)
        self.assertEqual(record['exit'], 1)
        self.assertIn('40000000', raw)
        self.assertIsNone(command.call_args.kwargs['expected'])

    def test_processes_have_explicit_bounded_deadlines(self):
        c = self.collector()
        with patch.object(self.mod, 'run', return_value=({'exit': 0}, '')) as runner:
            c.command('probe', ['fixture'], timeout=30)
        self.assertEqual(runner.call_args.kwargs['timeout'], 30)
        self.assertEqual(runner.call_args.kwargs['cwd'], self.source.resolve())
        for deadline in (0, -1, float('inf'), float('nan')):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                c.command('probe', ['fixture'], timeout=deadline)

    def test_manifest_has_exact_compiled_inputs(self):
        (self.source / 'Scripts/ci/unrelated.py').write_text('pass\n')
        manifest = self.mod.source_manifest(self.source)
        self.assertEqual(set(manifest), {'Package.swift', 'Sources/Test.swift'})
        self.assertEqual(len(manifest['Package.swift']), 64)

    def test_product_discovery_uses_stdout_despite_deprecation_warning(self):
        c = self.collector()
        self.output.mkdir()
        build = self.output / 'products'
        build.mkdir()
        module = build / 'VectorCore.swiftmodule'
        module.write_text('fixture module')
        stdout = self.output / 'path.stdout'
        stdout.write_text(str(build) + '\n')
        xml = self.output / 'debug-tests-swift-testing.xml'
        xml.write_text('<testsuites/>')
        combined = stdout.read_text() + "warning: '--build-system native' has been deprecated\n"
        with patch.object(c, 'command', side_effect=[({'exit': 0}, ''),
                ({'stdout': {'path': str(stdout)}}, combined), ({'exit': 0}, '')]), \
             patch.object(self.mod.products, 'link_objects', return_value=[build / 'Library.o']), \
             patch.object(self.mod.products, 'module_search_path', return_value=build), \
             patch.object(self.mod.products, 'module_map', return_value=build / 'module.modulemap'), \
             patch.object(self.mod.products, 'find_test_bundle', return_value=build / 'tests'):
            self.assertEqual(c.build_producer('debug'), build)
        self.assertEqual(c.capture['checks'][0]['xunit'], str(xml))


if __name__ == '__main__':
    unittest.main()
