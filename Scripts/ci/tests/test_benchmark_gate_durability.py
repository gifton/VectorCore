"""Bounded content-validation fixtures for the Python benchmark drivers; no Swift build."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVERS = (HERE.parent / 'check_benchmark_build_modes.py', HERE / 'test_benchmark_contracts.py')
REPORTING = HERE / 'test_benchmark_reporting.py'


def load(path):
    spec = importlib.util.spec_from_file_location('gate_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BeyondFixture(Exception):
    """End a synthetic cell before dispatcher compilation, after the report validation under test."""


def probe_modes(case, root):
    modes = load(DRIVERS[0])
    if case == 'command':
        return modes.assert_isolated(['clang', '-o', str(root / 'real' / 'dispatch.o')], [root / 'real'], 'fixture')
    if case == 'compiler-log':
        # Duplicate actual commands are invalid, even if they name the right source.
        source = root / 'dispatch.c'
        return modes.clang_command(f'clang -c {source} -o a.o\nclang -c {source} -o b.o', source)
    if case == 'empty-artifacts':
        return modes.artifacts(root, root / 'retained')
    if case == 'record-overwrite':
        (root / 'fixture.json').write_text('original')
        return modes.run([sys.executable, '-c', 'pass'], root, 'fixture')

    executable = root / 'debug' / 'vectorcore-bench'
    executable.parent.mkdir()
    executable.write_text('synthetic product')

    def run(argv, directory, name, **kwargs):
        if name == 'build':
            return ''
        if name != 'benchmark-absent':
            raise BeyondFixture()
        report = {'metadata': {'coreConfiguration': 'debug', 'benchmarkConfiguration': 'debug',
                               'flags': {'useCKernels': False}, 'selectedCaseIDs': ['dist.euclidean.512.optimized']},
                  'results': [{'iterations': 2}], 'failures': []}
        if case == 'report':
            report['metadata']['flags']['useCKernels'] = True
        Path(argv[-1]).write_text(json.dumps(report))
        return '' if case == 'banner' else 'configuration: debug\n[WARN debug build'

    # Discovery and launch are synthetic; cell's log/report checks execute unchanged.
    with patch.object(modes, 'run', run), patch.object(modes, 'find_executable', return_value=executable), \
         patch.object(modes.products, 'find_executable', return_value=executable), \
         patch.object(modes.products, 'detect_layout', return_value='native'), \
         patch.object(modes, 'artifacts', return_value=[{'fixture': True}]):
        modes.cell('debug', False, root / 'scratch', root, 1)


def probe_contracts(case, root):
    contracts = load(DRIVERS[1])

    def run(argv, directory, name, **kwargs):
        if name == 'compile':
            return ''
        if name.startswith('compare-'):
            (directory / (name + '.stderr')).write_text('')
            return ''  # Required comparator log fragments are deliberately absent.
        if Path(str(argv[0])).name.startswith('python'):
            if case == 'reporting-log':
                return ''  # Required reporting completion evidence is deliberately absent.
            output = Path(argv[-1]); output.mkdir()
            (output / 'result.json').write_text(json.dumps({'marker': 'V02_REPORTING_COMPLETE', 'cases': [{}, {}]}))
            return 'V02_REPORTING_COMPLETE 2'
        stderr = 'finite invalid threshold' if case != 'command' else 'unrelated diagnostic'
        if '--format' in argv:
            if '--strict-correctness' in argv:
                report = {'metadata': {'benchmarkConfiguration': 'release'}, 'failures': [],
                          'results': [{'correctness': {'status': 'invalid' if case == 'report' else 'valid', 'samples': 3}}]}
                Path(argv[-1]).write_text(json.dumps(report))
                stdout = '' if case == 'banner' else 'configuration: release'
                (directory / (name + '.stderr')).write_text('')
                return stdout
            stderr = 'Failed to write'
        if '--ab-only' in argv:
            stderr = '--ab-only incompatible suite'
        (directory / (name + '.stderr')).write_text(stderr)
        return ''

    helper = SimpleNamespace(run=run, write_json=lambda path, value: path.write_text(json.dumps(value)))
    if case == 'comparator-log':
        return contracts.comparator_contracts(helper, root)
    # main imports the real capture helper dynamically; replace only that launch boundary.
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    with patch.object(contracts.importlib.util, 'spec_from_file_location', return_value=spec), \
         patch.object(contracts.importlib.util, 'module_from_spec', return_value=helper), \
         patch.object(contracts, 'comparator_contracts', return_value=0), \
         patch.object(sys, 'argv', ['contracts', str(root / 'binary'), '--output', str(root)]):
        contracts.main()


def probe_reporting(case, root):
    reporting = load(REPORTING)

    def run(argv, directory, name, **kwargs):
        if name == 'compile':
            Path(argv[-1]).write_text('synthetic comparator')
        # The first regression comparison must print its hand-derived delta.
        return ''

    with patch.object(reporting.helper, 'run', run), \
         patch.object(sys, 'argv', ['reporting', '--output', str(root / 'reporting')]):
        reporting.main()


class BenchmarkGateDurability(unittest.TestCase):
    def run_probe(self, driver, case, optimization):
        return subprocess.run([sys.executable, '-B', *optimization, str(Path(__file__).resolve()),
                               '--probe', driver, case], capture_output=True, text=True, timeout=15)

    def test_negative_command_log_and_report_fixtures_in_all_python_modes(self):
        cases = {'modes': ['command', 'compiler-log', 'empty-artifacts', 'record-overwrite', 'banner', 'report'],
                 'contracts': ['command', 'banner', 'report', 'comparator-log', 'reporting-log'],
                 'reporting': ['comparison-log']}
        for optimization in [[], ['-O'], ['-OO']]:
            for driver, fixtures in cases.items():
                for case in fixtures:
                    with self.subTest(optimization=optimization, driver=driver, case=case):
                        result = self.run_probe(driver, case, optimization)
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn('AssertionError', result.stderr, result.stderr)

    def test_valid_content_reaches_the_next_stage_in_all_python_modes(self):
        for optimization in [[], ['-O'], ['-OO']]:
            for driver in ['modes', 'contracts']:
                with self.subTest(optimization=optimization, driver=driver):
                    result = self.run_probe(driver, 'valid', optimization)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runtime_asserts_cannot_return_in_the_benchmark_drivers(self):
        for path in (*DRIVERS, REPORTING):
            with self.subTest(driver=path.name):
                assertions = [node.lineno for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Assert)]
                self.assertEqual(assertions, [], f'{path}: optimization removes assert gates at {assertions}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--probe':
        with tempfile.TemporaryDirectory() as directory:
            try:
                {'modes': probe_modes, 'contracts': probe_contracts, 'reporting': probe_reporting}[sys.argv[2]](sys.argv[3], Path(directory))
            except BeyondFixture:
                pass
    else:
        unittest.main()
