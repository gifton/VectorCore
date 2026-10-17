"""Product discovery on synthetic SwiftPM output trees (no toolchain needed)."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import swiftpm_products as products
from swiftpm_products import ProductError


def make_bundle(root, name):
    path = Path(root) / (name + '.xctest') / 'Contents' / 'MacOS' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\xcf\xfa')
    return path


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_native_layout_resolves_every_target_to_the_aggregate_bundle(self):
        bin_path = self.root / 'arm64-apple-macosx' / 'debug'
        aggregate = make_bundle(bin_path, 'VectorCorePackageTests')
        for target in ['ComprehensiveTests', 'MinimalTests']:
            self.assertEqual(products.find_test_bundle(bin_path, target), aggregate)

    def test_swiftbuild_layout_selects_the_bundle_of_each_target(self):
        bin_path = self.root / 'out' / 'Products' / 'Debug'
        expected = {t: make_bundle(bin_path, t)
                    for t in ['BenchmarkContractTests', 'ComprehensiveTests', 'MinimalTests']}
        self.assertEqual(products.find_test_bundles(bin_path, ['MinimalTests', 'ComprehensiveTests', 'MinimalTests']),
                         {t: expected[t] for t in ['ComprehensiveTests', 'MinimalTests']})

    def test_missing_product_is_an_error(self):
        bin_path = self.root / 'Debug'
        make_bundle(bin_path, 'MinimalTests')
        with self.assertRaisesRegex(ProductError, 'No test bundle for ComprehensiveTests'):
            products.find_test_bundle(bin_path, 'ComprehensiveTests')
        with self.assertRaisesRegex(ProductError, 'No test bundle'):
            products.find_test_bundle(self.root / 'absent', 'MinimalTests')
        with self.assertRaisesRegex(ProductError, 'not found'):
            products.find_executable(bin_path, 'vectorcore-bench')

    def test_bundle_without_its_executable_is_not_a_candidate(self):
        bin_path = self.root / 'Debug'
        (bin_path / 'MinimalTests.xctest' / 'Contents' / 'MacOS').mkdir(parents=True)
        with self.assertRaisesRegex(ProductError, 'No test bundle'):
            products.find_test_bundle(bin_path, 'MinimalTests')

    def test_two_candidate_bundles_for_one_target_are_an_error(self):
        bin_path = self.root / 'Debug'
        make_bundle(bin_path, 'MinimalTests')
        make_bundle(bin_path / 'nested', 'MinimalTests')
        with self.assertRaisesRegex(ProductError, 'Ambiguous'):
            products.find_test_bundle(bin_path, 'MinimalTests')

    def test_per_target_and_aggregate_bundles_together_are_ambiguous(self):
        bin_path = self.root / 'Debug'
        make_bundle(bin_path, 'MinimalTests')
        make_bundle(bin_path, 'VectorCorePackageTests')
        with self.assertRaisesRegex(ProductError, 'Ambiguous'):
            products.find_test_bundle(bin_path, 'MinimalTests')

    def test_configuration_directory_case_is_taken_from_the_reported_bin_path(self):
        for index, configuration in enumerate(['Debug', 'debug', 'Release']):
            with self.subTest(configuration=configuration):
                bin_path = self.root / str(index) / configuration
                tool = bin_path / 'vectorcore-bench'
                tool.parent.mkdir(parents=True)
                tool.write_bytes(b'x')
                bundle = make_bundle(bin_path, 'ComprehensiveTests')
                self.assertEqual(products.find_executable(bin_path, 'vectorcore-bench'), tool)
                self.assertEqual(products.find_test_bundle(bin_path, 'ComprehensiveTests'), bundle)

    def test_search_is_confined_to_the_reported_bin_path(self):
        other = make_bundle(self.root / 'other' / 'debug', 'ComprehensiveTests')
        bin_path = self.root / 'bin' / 'Debug'
        bundle = make_bundle(bin_path, 'ComprehensiveTests')
        self.assertNotEqual(other, bundle)
        self.assertEqual(products.find_test_bundle(bin_path, 'ComprehensiveTests'), bundle)

    def test_executable_must_be_a_regular_file_not_a_symlink(self):
        bin_path = self.root / 'Debug'
        bin_path.mkdir()
        (bin_path / 'real').write_bytes(b'x')
        (bin_path / 'vectorcore-bench').symlink_to(bin_path / 'real')
        with self.assertRaises(ProductError):
            products.find_executable(bin_path, 'vectorcore-bench')


class SmallHelperTests(unittest.TestCase):
    def test_bin_path_parsing_takes_the_last_line_and_requires_absolute(self):
        self.assertEqual(products.parse_bin_path('warning: x\n/a/b/Debug\n'), Path('/a/b/Debug'))
        for text in ['', '\n', 'relative/path\n']:
            with self.subTest(text=text), self.assertRaises(ProductError):
                products.parse_bin_path(text)

    def test_build_system_flag_is_absent_by_default_and_validated(self):
        self.assertEqual(products.build_system_flags(None), [])
        self.assertEqual(products.build_system_flags('native'), ['--build-system', 'native'])
        self.assertEqual(products.build_system_flags('swiftbuild'), ['--build-system', 'swiftbuild'])
        with self.assertRaises(ProductError):
            products.build_system_flags('make')

    def test_filter_target_extraction(self):
        self.assertEqual(products.target_of_filter(r'^ComprehensiveTests\.Suite/test\(\)'), 'ComprehensiveTests')
        with self.assertRaises(ProductError):
            products.target_of_filter('Suite/test')

    def test_log_association_accepts_reported_or_resolved_spelling(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'x'
            path.write_bytes(b'')
            self.assertTrue(products.in_log(path, 'loaded ' + str(path)))
            self.assertTrue(products.in_log(path, 'loaded ' + str(path.resolve())))
            self.assertFalse(products.in_log(path, 'loaded /elsewhere'))

    def test_no_rebuild_check_rejects_build_output_in_test_logs(self):
        products.check_no_rebuild('Test run started.\nTest passed\n', 'x')
        for log in ['[3/9] Compiling Foo.swift\n', '[9/9] Linking X\n', 'Build complete! (1.0s)\n', 'Compiling Foo\n']:
            with self.subTest(log=log), self.assertRaises(ProductError):
                products.check_no_rebuild(log, 'x')


def touch(path, data=b'o'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_native(root, config='debug'):
    bin_path = Path(root) / 'arm64-apple-macosx' / config
    for name in ['a.swift.o', 'b.swift.o']:
        touch(bin_path / 'VectorCore.build' / name)
    touch(bin_path / 'VectorCoreC.build' / 'common' / 'vc_dispatch.c.o')
    touch(bin_path / 'VectorCoreC.build' / 'arm64' / 'vc_arm64.c.o')
    touch(bin_path / 'VectorCoreC.build' / 'module.modulemap', b'module VectorCoreC {}')
    touch(bin_path / 'Modules' / 'VectorCore.swiftmodule')
    return bin_path


def make_swiftbuild(root, config='Debug', architectures=('arm64',)):
    out = Path(root) / 'out'
    bin_path = out / 'Products' / config
    touch(bin_path / 'VectorCore.o')
    touch(bin_path / 'VectorCoreC.o')
    touch(bin_path / 'VectorCore.swiftmodule' / 'arm64-apple-macos.swiftmodule')
    touch(out / 'Intermediates.noindex' / 'GeneratedModuleMaps' / 'VectorCoreC.modulemap', b'module VectorCoreC {}')
    for target, names in [('VectorCore', ['a.o', 'b.o']), ('VectorCoreC', ['vc_dispatch.o', 'vc_arm64.o'])]:
        for arch in architectures:
            for name in names:
                touch(out / 'Intermediates.noindex' / 'VectorCore.build' / config / f'{target}-t.build'
                      / 'Objects-normal' / arch / name)
    return bin_path


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_native_layout_objects_modules_and_module_map(self):
        bin_path = make_native(self.root)
        self.assertEqual(products.detect_layout(bin_path), 'native')
        self.assertEqual([p.name for p in products.source_objects(bin_path, 'VectorCoreC')], ['vc_arm64.c.o', 'vc_dispatch.c.o'])
        self.assertEqual(products.link_objects(bin_path, ['VectorCore', 'VectorCoreC']),
                         products.source_objects(bin_path, 'VectorCore') + products.source_objects(bin_path, 'VectorCoreC'))
        self.assertEqual(products.module_search_path(bin_path), bin_path / 'Modules')
        self.assertEqual(products.module_map(bin_path, 'VectorCoreC'), bin_path / 'VectorCoreC.build' / 'module.modulemap')

    def test_swiftbuild_layout_merges_objects_and_has_no_module_modulemap(self):
        bin_path = make_swiftbuild(self.root)
        self.assertEqual(products.detect_layout(bin_path), 'swiftbuild')
        self.assertEqual(products.link_objects(bin_path, ['VectorCore', 'VectorCoreC']),
                         [bin_path / 'VectorCore.o', bin_path / 'VectorCoreC.o'])
        self.assertEqual([p.name for p in products.source_objects(bin_path, 'VectorCoreC')], ['vc_arm64.o', 'vc_dispatch.o'])
        self.assertEqual(products.module_search_path(bin_path), bin_path)
        self.assertEqual(products.module_map(bin_path, 'VectorCoreC').name, 'VectorCoreC.modulemap')
        self.assertFalse(list(bin_path.parent.parent.rglob('module.modulemap')))

    def test_release_configuration_name_is_followed(self):
        bin_path = make_swiftbuild(self.root, config='Release')
        self.assertEqual([p.name for p in products.source_objects(bin_path, 'VectorCore')], ['a.o', 'b.o'])

    def test_unrecognized_and_ambiguous_layouts_are_errors(self):
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaisesRegex(ProductError, 'Unrecognized'):
            products.detect_layout(empty)
        with self.assertRaisesRegex(ProductError, 'Unrecognized'):
            products.detect_layout(self.root / 'absent')
        bin_path = make_swiftbuild(self.root)
        (bin_path / 'VectorCore.build').mkdir()
        with self.assertRaisesRegex(ProductError, 'Ambiguous SwiftPM layout'):
            products.detect_layout(bin_path)

    def test_missing_pieces_are_errors_in_both_layouts(self):
        native = make_native(self.root / 'n')
        (native / 'VectorCoreC.build' / 'module.modulemap').unlink()
        with self.assertRaisesRegex(ProductError, 'Module map'):
            products.module_map(native, 'VectorCoreC')
        with self.assertRaisesRegex(ProductError, 'No objects'):
            products.source_objects(native, 'Absent')
        swift = make_swiftbuild(self.root / 's')
        (swift / 'VectorCoreC.o').unlink()
        with self.assertRaisesRegex(ProductError, 'Merged object'):
            products.link_objects(swift, ['VectorCoreC'])
        with self.assertRaisesRegex(ProductError, 'No object directory'):
            products.source_objects(swift, 'Absent')

    def test_swiftbuild_with_two_architectures_is_ambiguous(self):
        bin_path = make_swiftbuild(self.root, architectures=('arm64', 'x86_64'))
        with self.assertRaisesRegex(ProductError, 'exactly one architecture'):
            products.source_objects(bin_path, 'VectorCore')

    def test_swiftbuild_with_two_package_object_roots_is_ambiguous(self):
        bin_path = make_swiftbuild(self.root)
        touch(bin_path.parent.parent / 'Intermediates.noindex' / 'Other.build' / 'Debug' / 'VectorCore-t.build'
              / 'Objects-normal' / 'arm64' / 'a.o')
        with self.assertRaisesRegex(ProductError, 'Ambiguous object directories'):
            products.source_objects(bin_path, 'VectorCore')


if __name__ == '__main__':
    unittest.main()
