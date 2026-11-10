#!/usr/bin/env python3
"""Collect fresh native evidence. Completion is not compiler/release acceptance.

Every child has an owned process group and explicit deadline (v03_process).
NativeCollectionTests cover fresh directories, source drift, scheduling and
failure propagation. Numerical oracles belong to separately reviewed profiles.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

import swiftpm_products as products
from qualification_records import require, sha
from v03_process import run, test_environment, write_json

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = '448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f'
TARGET = 'arm64-apple-macosx14.0'
MODES = ('debug', 'release')


def source_manifest(root):
    paths = [root / 'Package.swift']
    for name in ('Sources', 'Tests', 'Benchmarks'):
        require((root / name).is_dir(), 'missing source directory: ' + name)
        paths.extend(p for p in (root / name).rglob('*') if p.is_file())
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def consumer_cells():
    return [(producer, client, omitted) for producer in MODES for client in MODES for omitted in (False, True)]


class NativeCapture:
    def __init__(self, source_root, swiftc, sdk, output, jobs):
        self.source, self.swiftc = Path(source_root).resolve(), Path(swiftc).absolute()
        self.sdk, self.output = Path(sdk).resolve(), Path(output).resolve()
        require(type(jobs) is int and 1 <= jobs <= 6, 'jobs must be in 1...6')
        self.jobs = jobs
        self.env = test_environment()
        self.env.update(SWIFT_EXEC=str(self.swiftc), SDKROOT=str(self.sdk),
                        PATH=str(self.swiftc.parent) + ':' + self.env.get('PATH', ''))
        self.capture = {'schemaVersion': 1, 'invocationId': str(uuid.uuid4()),
                        'collectionComplete': False, 'commands': {}, 'cases': [],
                        'checks': [], 'producers': {}, 'artifactFiles': []}

    def command(self, name, argv, *, timeout, expected=(0,)):
        require(math.isfinite(timeout) and 0 < timeout <= 7200, 'invalid process deadline')
        record, text = run(argv, self.output, name, cwd=self.source, environment=self.env,
                           timeout=timeout, expected=expected)
        self.capture['commands'][name] = str(self.output / (name + '.json'))
        return record, text

    def preflight(self):
        require(sha(self.swiftc.with_name('swift-frontend')) == FRONTEND, 'frontend identity differs')
        require(self.source == ROOT, 'collector must run from the selected source checkout')
        require(self.sdk.is_relative_to(self.swiftc.parents[4]), 'SDK and compiler developer directories differ')
        require((self.sdk / 'SDKSettings.json').is_file(), 'missing SDK settings')
        self.env['DEVELOPER_DIR'] = str(self.swiftc.parents[4])

    def identity(self):
        _, xcode = self.command('xcode-version', [self.swiftc.parents[4] / 'usr/bin/xcodebuild', '-version'], timeout=30)
        _, swift = self.command('compiler-version', [self.swiftc, '--version'], timeout=30)
        _, version = self.command('host-version', ['/usr/bin/sw_vers', '-productVersion'], timeout=10)
        _, build = self.command('host-build', ['/usr/bin/sw_vers', '-buildVersion'], timeout=10)
        _, revision = self.command('source-revision', ['git', 'rev-parse', 'HEAD'], timeout=10)
        self.command('source-state', ['git', 'status', '--porcelain=v1'], timeout=10)
        require(xcode.strip() == 'Xcode 27.2\nBuild version 27B52', 'Xcode tuple differs')
        require(swift.startswith('Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)'),
                'compiler version differs')
        require(platform.machine() == 'arm64', 'native architecture differs')
        self.capture['sourceCommit'] = revision.strip()
        return {'xcodeBuild': '27B52', 'compilerVersion': '6.4.2', 'frontendSHA256': FRONTEND,
                'sdkSettingsSHA256': sha(self.sdk / 'SDKSettings.json'), 'sdk': str(self.sdk),
                'platform': 'macos', 'architecture': 'arm64', 'target': TARGET, 'swiftVersion': '6',
                'runtime': {'version': version.strip(), 'build': build.strip()}}

    def compile_and_run(self, name, argv, binary):
        self.command(name + '-compile', argv, timeout=120)
        if not binary.is_file():
            raise FileNotFoundError(binary)
        return self.command(name + '-run', [binary], timeout=30, expected=None)

    def original_probes(self):
        driver = self.source / 'Docs/reviews/2026-11-04-release-phase-one/compiler/run.py'
        directory = self.output / 'original-probes'
        self.command('original-probe-driver', [sys.executable, '-B', driver, '--swiftc', self.swiftc,
                     '--sdk', self.sdk, '--frontend-sha256', FRONTEND, '--output', directory],
                     timeout=1800, expected=(0, 1))
        summary = json.loads((directory / 'summary.json').read_text())
        require(len(summary['cells']) == 16, 'incomplete original probe collection')
        self.capture['producers']['standalone'] = {'kind': 'standalone', 'objects': [], 'modules': []}
        for row in summary['cells']:
            require(row['compileExit'] == 0 and row.get('runExit') == 0, 'original probe setup/execution failed')
            label = Path(row['probe']).stem + '-' + row['configuration']
            self.case('original/' + label, 'lost-store', 'standalone', row['configuration'],
                      'original-probes/' + label + '-compile', 'original-probes/' + label + '-run',
                      directory / label)

    def build_producer(self, mode):
        flags = ['--build-system', 'native', '--scratch-path', self.output / 'build', '--sdk', self.sdk,
                 '--triple', TARGET, '-c', mode, '-j', str(self.jobs)]
        swift = self.swiftc.with_name('swift')
        xml = self.output / (mode + '-tests.xml')
        self.command(mode + '-full-suite', [swift, 'test', *flags, '--xunit-output', xml],
                     timeout=1800, expected=(0, 1))
        path_record, _ = self.command(mode + '-bin-path', [swift, 'build', *flags, '--show-bin-path'], timeout=120)
        build = products.parse_bin_path(Path(path_record['stdout']['path']).read_text())
        objects = products.link_objects(build, ['VectorCore', 'VectorCoreC'])
        module_root = products.module_search_path(build)
        modules = [p for p in module_root.rglob('*') if p.is_file() and p.name.startswith('VectorCore.')]
        require(bool(modules), 'missing producer modules')
        modulemap = products.module_map(build, 'VectorCoreC')
        bundle = products.find_test_bundle(build, 'VectorCorePackageTests')
        self.command(mode + '-test-list', [swift, 'test', *flags, '--skip-build', 'list'], timeout=120)
        self.capture['producers'][mode] = {'kind': 'library', 'objects': list(map(str, objects)),
            'modules': list(map(str, modules)), 'modulemap': str(modulemap),
            'buildCommand': mode + '-full-suite', 'testBinary': str(bundle)}
        self.capture['checks'].append({'id': mode + '-suite', 'format': 'native-suite',
            'command': mode + '-full-suite', 'listCommand': mode + '-test-list',
            'producer': mode, 'binary': str(bundle),
            'xunit': str(xml.with_name(xml.stem + '-swift-testing.xml'))})
        return build

    def case(self, name, format, producer, client, compile_id, run_id, binary, **extra):
        self.capture['cases'].append({'id': name, 'format': format, 'producer': producer,
            'client': client, 'compileCommand': compile_id, 'command': run_id, 'binary': str(binary), **extra})

    def consumers(self, producer, build):
        driver = self.source / 'Scripts/ci/consumer_buffer_value.py'
        directory = self.output / ('consumers-' + producer)
        self.command('consumer-driver-' + producer, [sys.executable, '-B', driver,
            '--swiftc', self.swiftc, '--frontend-sha256', FRONTEND, '--sdk', self.sdk,
            '--bin-path', build, '--library-configuration', producer, '--output', directory], timeout=1800)
        summary = json.loads((directory / 'summary.json').read_text())
        require(summary['validationPassed'] is True, 'by-value/contract collection failed')
        for p, client, omitted in consumer_cells():
            if p != producer:
                continue
            label = client + ('-omitted-write' if omitted else '-by-value')
            folder = directory / label
            prefix = folder.relative_to(self.output).as_posix()
            self.case(prefix, 'buffer-value', producer, client, prefix + '/consumer-compile',
                prefix + '/consumer-run', folder / 'consumer',
                witness={'command': prefix + '/witness-compile', 'object': str(folder / 'witness.o'),
                         'module': str(folder / 'BufferValueWitness.swiftmodule')})
        for client in MODES:
            prefix = f'consumers-{producer}/{client}-legacy-exposure'
            self.case(prefix, 'buffer-value', producer, client, prefix + '/compile', prefix + '/run',
                      self.output / prefix / 'consumer')
        for contract in summary['migrationContracts']:
            name = f"consumers-{producer}/{contract['clientMode']}-{contract['id']}"
            self.capture['checks'].append({'id': name, 'format': 'migration-contract', 'command': name,
                'producer': producer, 'client': contract['clientMode'], 'contract': contract['id']})

    def residual(self, producer, build):
        original = self.source / 'Docs/reviews/2026-11-06-workaround-qualification/ResidualExposure.swift'
        source = self.output / 'ResidualExposure.swift'
        if not source.exists():
            shutil.copy2(original, source)
        require(sha(source) == sha(original), 'residual source copy differs')
        common = [self.swiftc, '-swift-version', '6', '-target', TARGET, '-sdk', self.sdk,
            '-module-cache-path', self.output / 'module-cache', '-I', products.module_search_path(build),
            '-Xcc', '-fmodule-map-file=' + str(products.module_map(build, 'VectorCoreC')),
            '-Xcc', '-I' + str(self.source / 'Sources/VectorCoreC/include'), '-parse-as-library']
        objects = products.link_objects(build, ['VectorCore', 'VectorCoreC'])
        for p, client, omitted in consumer_cells():
            if p != producer:
                continue
            name = f'residual-{producer}-{client}' + ('-omitted' if omitted else '-positive')
            binary = self.output / name
            flags = ['-O' if client == 'release' else '-Onone'] + (['-D', 'OMIT_WRITES'] if omitted else [])
            self.compile_and_run(name, [*common, *flags, source, *objects, '-framework', 'Accelerate', '-o', binary], binary)
            self.case(name, 'buffer-value', producer, client, name + '-compile', name + '-run', binary)

    def build_modes(self):
        self.command('build-modes-driver', [sys.executable, '-B', self.source / 'Scripts/ci/check_benchmark_build_modes.py',
            '--scratch-root', self.output / 'c-build', '--output', self.output / 'c-modes', '--jobs', str(self.jobs)],
            timeout=7200)
        self.capture['checks'].append({'id': 'c-build-modes', 'format': 'c-build-modes',
            'command': 'build-modes-driver', 'result': str(self.output / 'c-modes/result.json')})

    def finish_inventory(self):
        # Retain every v03 receipt, including nested drivers. The C route driver
        # has its own older receipt schema; it is bound and checked separately.
        artifacts = set()
        for directory in (self.output, self.output / 'original-probes',
                          self.output / 'consumers-debug', self.output / 'consumers-release'):
            if not directory.exists():
                continue
            paths = directory.glob('*') if directory == self.output else directory.rglob('*')
            for path in paths:
                if not path.is_file() or 'module-cache' in path.parts:
                    continue
                artifacts.add(str(path))
                if path.suffix != '.json' or path.name == 'capture.json':
                    continue
                value = json.loads(path.read_text())
                if type(value) is dict and 'executableSHA256' in value and 'argv' in value:
                    key = path.relative_to(self.output).with_suffix('').as_posix()
                    self.capture['commands'][key] = str(path)
                    artifacts.update((value['executable'], value['stdout']['path'], value['stderr']['path']))
        for producer in self.capture['producers'].values():
            artifacts.update(producer['objects'] + producer['modules'])
            for field in ('modulemap', 'testBinary'):
                if field in producer:
                    artifacts.add(producer[field])
        for case in self.capture['cases']:
            artifacts.add(case['binary'])
        for path in (self.output / 'c-modes').rglob('*'):
            if path.is_file():
                artifacts.add(str(path))
        for path in (self.source / 'Scripts').rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                artifacts.add(str(path))
        artifacts.update((str(self.swiftc.with_name('swift-frontend')), str(self.sdk / 'SDKSettings.json')))
        artifacts.discard(str(self.output / 'capture.json'))
        self.capture['artifactFiles'] = sorted(artifacts)

    def collect(self):
        if self.output.exists():
            raise FileExistsError(self.output)
        self.preflight()
        self.output.mkdir(parents=True, exist_ok=False)
        manifest = source_manifest(self.source)
        write_json(self.output / 'source-inputs.json', manifest)
        self.capture['sourceManifest'] = str(self.output / 'source-inputs.json')
        capture_path = self.output / 'capture.json'
        write_json(capture_path, self.capture)
        try:
            self.capture['identity'] = self.identity()
            self.original_probes()
            for mode in MODES:
                print('Collecting ' + mode + ' producer and clients', flush=True)
                build = self.build_producer(mode)
                self.consumers(mode, build)
                self.residual(mode, build)
                write_json(capture_path, self.capture)
            self.build_modes()
            after = source_manifest(self.source)
            write_json(self.output / 'source-inputs-after.json', after)
            require(after == manifest, 'compiled source inputs changed during collection')
            self.finish_inventory()
            self.capture['collectionComplete'] = True
        except BaseException as error:
            self.capture['collectionProblem'] = str(error)
            raise
        finally:
            write_json(capture_path, self.capture)
        return capture_path


def collect_native(source_root: Path, swiftc: Path, sdk: Path, output: Path, jobs: int) -> Path:
    return NativeCapture(source_root, swiftc, sdk, output, jobs).collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'swiftc', 'sdk', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frontend-sha256', required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.frontend_sha256 != FRONTEND:
        parser.error('frontend identity is not the adopted Xcode 27.2 profile')
    path = collect_native(args.source_root, args.swiftc, args.sdk, args.output, args.jobs)
    print(str(path) + '; collectionComplete=true; qualification=unassessed; releaseAdmitted=false')


if __name__ == '__main__':
    main()
