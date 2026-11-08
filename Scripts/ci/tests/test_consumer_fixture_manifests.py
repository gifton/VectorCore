"""F011 retained-input checks and D02 orchestration regressions.

Fixture discovery reads the drivers' actual declarations and fixture-relative path
expressions. It supports literal names and tuple-loop names, and fails for an
unrecognized fixture-relative expression; this is not general Python dataflow.
The mocked compiler tests exercise the real D02 main function's decisions; actual
compiler/link/SILGen/symbolgraph validation is a separate driver execution.
"""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D02 = load('f011_d02', 'Scripts/ci/tests/D02RunConsumers.py')
SMOKE = load('f011_smoke', 'Scripts/ci/consumer_smoke.py')
DRIVERS = {
    'Scripts/ci/consumer_buffer_value.py': 'Scripts/ci/fixtures/compile-contracts',
    'Scripts/ci/tests/D02RunConsumers.py': 'Scripts/ci/tests/D02',
    'Scripts/ci/tests/D02CContract.py': 'Scripts/ci/tests/D02C',
    'Scripts/ci/fixtures/B01_public_driver.py': 'Scripts/ci/fixtures',
    'Scripts/ci/fixtures/L03_public_driver.py': 'Scripts/ci/fixtures',
    'Scripts/ci/fixtures/R05_public_driver.py': 'Scripts/ci/fixtures',
    'Scripts/ci/check_benchmark_build_modes.py': 'Scripts/ci/fixtures/V02',
}


def literal_fixture_names(source):
    tree = ast.parse(source)
    bindings = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, (ast.Tuple, ast.List)):
            continue
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        for index, target in enumerate(node.target.elts):
            if not isinstance(target, ast.Name):
                continue
            values = set()
            for item in node.iter.elts:
                if isinstance(item, (ast.Tuple, ast.List)) and len(item.elts) > index:
                    value = item.elts[index]
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        values.add(value.value)
            if values:
                bindings.setdefault(target.id, set()).update(values)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name) \
                and node.left.id in ('FIXTURES', 'fixture'):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                names.add(node.right.value)
            elif isinstance(node.right, ast.Name) and node.right.id in bindings:
                names.update(bindings[node.right.id])
            else:
                raise AssertionError('unreviewed fixture expression: ' + ast.unparse(node))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'tsv_paths':
            names.add(ast.literal_eval(node.args[0]))
    return names


def retained_inputs():
    result = {}
    for driver, directory in DRIVERS.items():
        names = literal_fixture_names((ROOT / driver).read_text())
        if driver.endswith('/D02RunConsumers.py'):
            names.update(D02.TABLE)
        if not names:
            raise AssertionError('empty fixture inventory: ' + driver)
        result[driver] = {str(Path(directory) / name) for name in names}
    result['Scripts/ci/consumer_smoke.py'] = {row['path'] for row in SMOKE.CONTRACTS}
    return result


def missing_inputs(root, inventory):
    return sorted((driver, path) for driver, paths in inventory.items() for path in paths if not (root / path).is_file())


class ConsumerFixtureManifests(unittest.TestCase):
    def test_retained_drivers_have_all_referenced_fixtures(self):
        self.assertEqual(missing_inputs(ROOT, retained_inputs()), [])
        self.assertEqual(set(D02.TABLE), {p.name for p in D02.FIXTURES.glob('*.swift')})
        discovered = {str(p.relative_to(ROOT)) for p in (ROOT / 'Scripts/ci/fixtures').glob('*_public_driver.py')}
        discovered.update(str(p.relative_to(ROOT)) for p in (ROOT / 'Scripts/ci/tests').glob('*RunConsumers.py'))
        self.assertEqual(discovered, {p for p in DRIVERS if p.endswith('_public_driver.py') or p.endswith('RunConsumers.py')})

    def test_removing_each_referenced_fixture_is_detected(self):
        inventory = retained_inputs()
        paths = {path for group in inventory.values() for path in group}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(missing_inputs(root, inventory), [])
            for name in sorted(paths):
                with self.subTest(path=name):
                    (root / name).unlink()
                    expected = sorted((driver, name) for driver, names in inventory.items() if name in names)
                    self.assertEqual(missing_inputs(root, inventory), expected)
                    (root / name).touch()

    def test_fixture_expression_changes_require_review(self):
        self.assertEqual(literal_fixture_names("fixture / 'one.swift'\nfor lang, src in [('c','two.c'),('cpp','three.cpp')]:\n FIXTURES / src"),
                         {'one.swift', 'two.c', 'three.cpp'})
        with self.assertRaisesRegex(AssertionError, 'unreviewed fixture expression'):
            literal_fixture_names('FIXTURES / choose_runtime_name()')

    def test_d02_manifest_errors_precede_snapshot_and_output_creation(self):
        for missing, extra in [('Factory.swift', None), (None, 'Unlisted.swift'),
                               ('gated-public-base.tsv', None), ('protocol-members-base.tsv', None)]:
            with self.subTest(missing=missing, extra=extra), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                fixtures = directory / 'fixtures'
                fixtures.mkdir()
                for name in [*D02.TABLE, 'gated-public-base.tsv', 'protocol-members-base.tsv']:
                    if name != missing:
                        (fixtures / name).touch()
                if extra:
                    (fixtures / extra).touch()
                output = directory / 'output'
                with mock.patch.object(D02, 'FIXTURES', fixtures), \
                        mock.patch.object(D02.products, 'snapshot_consumer_inputs') as snapshot, \
                        mock.patch.object(D02.subprocess, 'run') as run, \
                        mock.patch.object(D02.subprocess, 'Popen') as popen, \
                        mock.patch.object(D02.signal, 'signal'), \
                        mock.patch.object(sys, 'argv', ['D02RunConsumers.py', '/unused/build', str(output), 'final']):
                    with self.assertRaisesRegex(SystemExit, missing or extra):
                        D02.main()
                    snapshot.assert_not_called()
                    run.assert_not_called()
                    popen.assert_not_called()
                    self.assertFalse(output.exists())

    def test_d02_preserves_mode_and_restored_fixture_contracts(self):
        self.assertEqual({tuple(sorted(row)) for row in D02.TABLE.values()}, {('baseline', 'final')})
        self.assertEqual(D02.TABLE['StoragePositive.swift'], {
            'baseline': ('fails', "does not conform to protocol 'StaticDimension'"),
            'final': ('positive', 'D02_STORAGE_POSITIVE_COMPLETE')})
        self.assertEqual(D02.TABLE['DeviceSupport.swift'], {
            'baseline': ('fails', "has no member 'isSupportedByCore'"),
            'final': ('positive', 'D02_DEVICE_SUPPORT_COMPLETE')})
        self.assertEqual(D02.TABLE['Factory.swift'], {
            mode: ('positive', 'D02_FACTORY_COMPLETE') for mode in ('baseline', 'final')})


class D02DriverOrchestration(unittest.TestCase):
    def run_driver(self, *, silgen=0, graph=0, logger_lines=(9, 10), logger_status=1, unbound=(), fallbacks=(), mode='final'):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixtures = directory / 'fixtures'
            fixtures.mkdir()
            for name in ('Reachability.swift', 'LoggerNegative.swift', 'gated-public-base.tsv', 'protocol-members-base.tsv'):
                shutil.copy2(D02.FIXTURES / name, fixtures / name)
            output = directory / 'output'
            modulemap = directory / 'module.modulemap'
            modulemap.write_text('')
            snapshot = {'modules': directory, 'modulemap': modulemap, 'objects': [], 'records': []}
            rows = [line.split('\t')[0] for line in (fixtures / 'protocol-members-base.tsv').read_text().splitlines()
                    if line and not line.startswith(('#', 'path\t'))]
            table = {name: D02.TABLE[name] for name in ('Reachability.swift', 'LoggerNegative.swift')}
            # Only Logger's identical baseline/final diagnostic contract is under test in baseline mode.
            if mode == 'baseline':
                table['Reachability.swift'] = {'baseline': ('positive', 'D02_REACHABILITY_COMPLETE'),
                                               'final': ('positive', 'D02_REACHABILITY_COMPLETE')}

            class CompilerProcess:
                pid = 123456
                def __init__(self, command, stdout, **kwargs):
                    self.returncode = 0
                    text = ''
                    if '-emit-silgen' in command:
                        self.returncode = silgen
                        if silgen == 0:
                            Path(command[command.index('-o') + 1]).write_text('// controlled SILGen output\n')
                    elif 'swift-symbolgraph-extract' in command:
                        self.returncode = graph
                        if graph == 0:
                            folder = Path(command[command.index('-output-dir') + 1])
                            (folder / 'VectorCore.symbols.json').write_text(json.dumps(
                                {'symbols': [{'pathComponents': row.split('.', 1)} for row in rows]}))
                    elif '-typecheck' in command and str(command[-1]).endswith('LoggerNegative.swift'):
                        self.returncode = logger_status
                        text = ''.join(f'{command[-1]}:{line}:1: error: ' +
                                       ("type 'VectorCore' has no member 'Logger'" if line == 10 else 'ambiguous Logger') + '\n'
                                       for line in logger_lines)
                    elif '-o' in command:
                        Path(command[command.index('-o') + 1]).write_bytes(b'controlled linked artifact')
                    else:
                        text = 'D02_REACHABILITY_COMPLETE\n'
                    stdout.write(text.encode())
                def wait(self, timeout=None):
                    return self.returncode
                def poll(self):
                    return self.returncode

            def tool(command, **kwargs):
                if 'swift-demangle' in command:
                    return SimpleNamespace(returncode=0, stdout='')
                return SimpleNamespace(returncode=0, stdout='/controlled/toolchain/path\n')

            printed = io.StringIO()
            with mock.patch.object(D02, 'FIXTURES', fixtures), mock.patch.object(D02, 'TABLE', table), \
                    mock.patch.object(D02.products, 'snapshot_consumer_inputs', return_value=snapshot), \
                    mock.patch.object(D02.subprocess, 'Popen', CompilerProcess), mock.patch.object(D02.subprocess, 'run', tool), \
                    mock.patch.object(D02.signal, 'signal'), mock.patch.object(D02, 'binding_failures', return_value=(list(unbound), list(fallbacks))), \
                    mock.patch.object(sys, 'argv', ['D02RunConsumers.py', '/unused/build', str(output), mode]), \
                    contextlib.redirect_stdout(printed):
                status = D02.main()
            return status, printed.getvalue(), json.loads((output / 'provenance.json').read_text())

    def test_binding_status_uses_its_own_stage_result(self):
        for silgen, graph, expected in [(1, 0, 'FAILED'), (0, 1, 'passed'), (0, 0, 'passed')]:
            with self.subTest(silgen=silgen, graph=graph):
                status, printed, record = self.run_driver(silgen=silgen, graph=graph)
                self.assertIn('binding check ' + expected, printed)
                self.assertEqual(record['bindingCheck']['silgenExit'], silgen)
                self.assertEqual(record['memberBackstop']['extractExit'], graph)
                self.assertEqual(status, int(silgen != 0 or graph != 0))
        for kwargs in ({'unbound': ['Vector.isFinite']}, {'fallbacks': ['VectorProtocol.isFinite']}):
            status, printed, record = self.run_driver(**kwargs)
            self.assertEqual(status, 1)
            self.assertIn('binding check FAILED', printed)
            self.assertTrue(record['problems'])

    def test_logger_required_and_optional_diagnostics(self):
        for mode in ('baseline', 'final'):
            for lines, compiler_exit, expected in [((10,), 1, True), ((9, 10), 1, True), ((9,), 1, False),
                                                  ((9, 9, 10), 1, False), ((8, 10), 1, False),
                                                  ((10, 10), 1, False), ((10,), 0, False), ((10,), -5, False)]:
                with self.subTest(mode=mode, lines=lines, compiler_exit=compiler_exit):
                    status, _, record = self.run_driver(mode=mode, logger_lines=lines, logger_status=compiler_exit)
                    self.assertEqual(record['results']['LoggerNegative.swift']['passed'], expected)
                    self.assertEqual(status, 0 if expected else 1)


if __name__ == '__main__':
    unittest.main()
