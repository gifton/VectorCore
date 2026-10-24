"""check_benchmark_build_modes.py discovery and compile-command rewriting under both SwiftPM layouts.

Synthetic trees only; no toolchain. Complements test_swiftpm_products.py. Importing the
driver this way is the same contract the V04 benchmark tests rely on.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('build_modes_under_test', HERE.parent / 'check_benchmark_build_modes.py')
modes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modes)


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x')
    return path


class ExecutableDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_native_and_swiftbuild_products_are_found_by_configuration(self):
        native = touch(self.root / 'n' / 'arm64-apple-macosx' / 'debug' / 'vectorcore-bench')
        swift = touch(self.root / 's' / 'out' / 'Products' / 'Release' / 'vectorcore-bench')
        self.assertEqual(modes.find_executable(self.root / 'n', 'debug'), native)
        self.assertEqual(modes.find_executable(self.root / 's', 'release'), swift)

    def test_wrong_configuration_missing_and_ambiguous_products_fail(self):
        touch(self.root / 'arm64' / 'debug' / 'vectorcore-bench')
        with self.assertRaises(AssertionError):
            modes.find_executable(self.root, 'release')
        with self.assertRaises(AssertionError):
            modes.find_executable(self.root / 'absent', 'debug')
        touch(self.root / 'out' / 'Products' / 'Debug' / 'vectorcore-bench')
        with self.assertRaises(AssertionError):
            modes.find_executable(self.root, 'debug')


class ReplacementCommandTests(unittest.TestCase):
    def test_swiftbuild_side_outputs_are_redirected_away_from_the_real_build(self):
        original = ['clang', '-c', '/src/vc_dispatch.c', '-o', '/build/Objects-normal/arm64/vc_dispatch.o',
                    '-MF', '/build/vc_dispatch.d', '-MT', 'dependencies',
                    '--serialize-diagnostics', '/build/vc_dispatch.dia',
                    '-index-unit-output-path', '/build/vc_dispatch.o', '-index-store-path', '/build/index']
        output = Path('/work/dispatcher-renamed.o')
        command = modes.replacement_command(original, Path('/work/renamed.c'), output)
        self.assertEqual(command[command.index('-c') + 1], '/work/renamed.c')
        self.assertEqual(command[command.index('-o') + 1], str(output))
        for flag in ['-MF', '--serialize-diagnostics', '-index-unit-output-path', '-index-store-path']:
            self.assertTrue(command[command.index(flag) + 1].startswith('/work/'), flag)
        self.assertTrue(all('/build/' not in part for part in command))
        self.assertIn('/build/vc_dispatch.d', original)  # input list is not mutated

    def test_native_command_without_swiftbuild_flags_is_unchanged_apart_from_paths(self):
        original = ['clang', '-c', '/src/a.c', '-o', '/b/a.c.o', '-MF', '/b/a.d']
        command = modes.replacement_command(original, Path('/w/x.c'), Path('/w/x.o'))
        self.assertEqual(command, ['clang', '-c', '/w/x.c', '-o', '/w/x.o', '-MF', '/w/x.d'])


class DriverFlagAliasTests(unittest.TestCase):
    """The old flags (--artifacts for B01/L03, --products for R05) remain accepted beside --bin-path."""

    def help_text(self, name):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-B', str(HERE.parent / 'fixtures' / name), '--help'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_old_flag_spellings_are_aliases_of_bin_path(self):
        for name, old in [('B01_public_driver.py', '--artifacts'), ('L03_public_driver.py', '--artifacts'),
                          ('R05_public_driver.py', '--products')]:
            with self.subTest(driver=name):
                text = self.help_text(name)
                self.assertIn('--bin-path', text)
                self.assertIn(old, text)


class IsolationTests(unittest.TestCase):
    def test_arguments_resolving_under_real_object_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / 'Objects-normal' / 'arm64'
            root.mkdir(parents=True)
            modes.assert_isolated(['clang', '-o', str(Path(name) / 'elsewhere.o')], [root], 'ok')
            for argument in [str(root / 'vc_dispatch.o'), '-MF=' + str(root / 'vc_dispatch.d'), str(root)]:
                with self.subTest(argument=argument), self.assertRaises(AssertionError):
                    modes.assert_isolated(['clang', argument], [root], 'bad')

    def test_snapshot_detects_a_changed_side_file(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'a.o').write_bytes(b'1')
            (root / 'a.d').write_bytes(b'2')
            before = modes.snapshot([root])
            self.assertEqual(set(before), {str(root / 'a.o'), str(root / 'a.d')})
            (root / 'a.d').write_bytes(b'3')
            self.assertNotEqual(modes.snapshot([root]), before)


if __name__ == '__main__':
    unittest.main()
