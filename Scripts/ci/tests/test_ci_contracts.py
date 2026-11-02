"""Independent body requirements and real target/lint policy mutations."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('ci_contracts', ROOT / 'Scripts/ci/check_ci_contracts.py')
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class CIContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ci-contracts-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        paths = ['Docs/executed-ci-contracts.md', 'Package.swift', '.swiftlint.yml', 'Sources/.swiftlint.yml',
                 'Scripts/ci/sanitizer_bodies.json', 'Scripts/ci/sanitizer_expected_bodies.txt']
        data = json.loads((ROOT / paths[-2]).read_text())
        paths += list({row['source'] for rows in data['sanitizers'].values() for row in rows})
        paths += ['Tests/BenchmarkContractTests/' + name for name in POLICY.BENCHMARK_TEST_SOURCES]
        for relative in paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, path)

    def test_actual_target_membership_and_selected_bodies(self):
        result = POLICY.validate(ROOT)
        self.assertEqual(result['sanitizerBodies'], {'address': 77, 'thread': 39})
        self.assertEqual(len(result['benchmarkTestSources']), 4)

    def test_deleting_each_required_body_cannot_shrink_expectation(self):
        path = self.root / 'Scripts/ci/sanitizer_bodies.json'
        original = path.read_text()
        data = json.loads(original)
        for mode, rows in data['sanitizers'].items():
            for index, row in enumerate(rows):
                mutated = json.loads(original)
                mutated['sanitizers'][mode].pop(index)
                path.write_text(json.dumps(mutated))
                with self.subTest(mode=mode, body=row['id']), self.assertRaises(ValueError):
                    POLICY.validate(self.root)
        path.write_text(original)

    def test_missing_body_definition_and_empty_or_widened_filter_rejected(self):
        path = self.root / 'Scripts/ci/sanitizer_bodies.json'
        original = path.read_text()
        for value in ['', '.*', 'MemoryPoolTests']:
            data = json.loads(original)
            data['sanitizers']['address'][0]['filter'] = value
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                POLICY.validate(self.root)
        path.write_text(original)
        source = self.root / 'Tests/ComprehensiveTests/MemoryPoolTests.swift'
        source.write_text(source.read_text().replace('func testAcquire_ReturnsBufferWithCountAndAlignment(', 'func removedBody('))
        with self.assertRaises(ValueError):
            POLICY.validate(self.root)

    def test_benchmark_registration_and_source_exclusion_mutations(self):
        package = self.root / 'Package.swift'
        original = package.read_text()
        for changed in [original.replace('name: "BenchmarkContractTests"', 'name: "DisabledBenchmarkContractTests"'),
                        original.replace('dependencies: ["VectorCoreBenchSupport", "VectorCoreBenchmarking"]',
                                         'dependencies: ["VectorCoreBenchSupport", "VectorCoreBenchmarking"], exclude: ["RunnerContractTests.swift"]')]:
            package.write_text(changed)
            with self.assertRaises(ValueError):
                POLICY.validate(self.root)
        package.write_text(original)
        (self.root / 'Tests/BenchmarkContractTests/RunnerContractTests.swift').unlink()
        with self.assertRaises(ValueError):
            POLICY.validate(self.root)

    def test_lint_roots_and_unexecuted_analysis_mutations(self):
        for relative in ['.swiftlint.yml', 'Sources/.swiftlint.yml']:
            path = self.root / relative
            original = path.read_text()
            for changed in [original.replace('  - Benchmarks\n', ''), original + '\nanalyzer_rules:\n  - unused_import\n']:
                path.write_text(changed)
                with self.assertRaises(ValueError):
                    POLICY.validate(self.root)
            path.write_text(original)
