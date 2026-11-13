"""Synthetic iOS capture for binding tests; no synthetic program is executed."""
import copy
import json
from pathlib import Path
import shlex
import uuid

from qualification_fixture import CaptureFixture, sha, write_json
from qualification_ios import adapt_fixture, verify_adapter


class IOSCaptureFixture(CaptureFixture):
    def __init__(self, root):
        super().__init__(root)
        self.identity.update(platform='ios-simulator', target='arm64-apple-ios17.0-simulator',
                             runtime={'version': '27.0', 'description': 'Version 27.0 (Build 24A434)'},
                             sdk=str(self.sdk.parent))
        self.capture['identity'] = copy.deepcopy(self.identity)
        self.profile['identity'] = copy.deepcopy(self.identity)
        self.tool = {}
        for name in ('swift', 'swiftc', 'swift-frontend', 'xcodebuild', 'clang', 'libtool', 'xcrun'):
            path = self.evidence / 'tools' / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(self.compiler.read_bytes())
            self.tool[name] = str(path)
        self.profile['identityFiles']['frontendSHA256'] = self.tool['swift-frontend']
        self.binary = self.evidence / 'QualificationHost.app/QualificationHost'
        self.binary.parent.mkdir()
        self.binary.write_text('synthetic app')
        self.producer = self.capture['producers']['release']
        self.producer.update(kind='library', buildCommand='producer-build')
        self.profile['producers'] = copy.deepcopy(self.capture['producers'])
        self.destination = 'platform=iOS Simulator,id=38d12cec-bc01-44b4-82e5-39e92adb39ce'
        self.original = self.evidence / 'original.swift'
        self.original.write_text('@main\nprivate struct Probe {\n    static func main() {\n'
                                 '        print("fixture")\n        exit(0)\n    }\n}\n')
        self.adapter = self.evidence / 'ProbeAdapter.swift'
        self.adapter.write_text(adapt_fixture(self.original.read_text(), 'main'))
        self.host = self.evidence / 'App.swift'
        self.host.write_text('synthetic app host')
        self.witness = self.evidence / 'BufferValueWitness.swift'
        self.witness.write_text('synthetic witness')
        template = self.source / 'Scripts/ci/fixtures'
        (template / 'compiler-qualification-ios').mkdir(parents=True)
        (template / 'compiler-qualification-ios/App.swift').write_bytes(self.host.read_bytes())
        (template / 'compile-contracts').mkdir()
        (template / 'compile-contracts/BufferValueWitness.swift').write_bytes(self.witness.read_bytes())
        self.archive = self.evidence / 'Products/libBufferValueWitness.a'
        self.archive.parent.mkdir()
        self.archive.write_bytes(b'witness archive')
        self.witness_module = self.evidence / 'Objects/BufferValueWitness.swiftmodule'
        self.witness_module.parent.mkdir()
        self.witness_module.write_bytes(b'witness module')
        imported = self.archive.parent / 'BufferValueWitness.swiftmodule/arm64-apple-ios-simulator.swiftmodule'
        imported.parent.mkdir()
        imported.write_bytes(self.witness_module.read_bytes())
        self.swift_commands = {}
        lists = {}
        for name, sources in (('QualificationHost', [self.host, self.adapter]), ('BufferValueWitness', [self.witness])):
            listing = self.evidence / (name + '.SwiftFileList')
            listing.write_text('\n'.join(map(str, sources)) + '\n')
            outputs = {}
            objects = []
            for source in sources:
                obj = self.evidence / 'Objects' / (source.stem + '.o')
                obj.write_bytes(source.read_bytes())
                outputs[str(source)] = {'object': str(obj)}
                objects.append(obj)
            outputmap = self.evidence / (name + '.OutputFileMap.json')
            write_json(outputmap, outputs)
            linklist = self.evidence / (name + '.LinkFileList')
            linklist.write_text('\n'.join(map(str, objects)) + '\n')
            lists[name] = linklist
            self.swift_commands[name] = [self.tool['swiftc'], '-module-name', name, '-O',
                '-target', self.identity['target'], '-swift-version', '6', '-sdk', str(self.sdk.parent),
                '@' + str(listing), '-output-file-map', str(outputmap), '-emit-module-path',
                str(self.witness_module), '-I', str(self.module.parent), '-I', str(self.archive.parent)]
        self.link = [self.tool['clang'], '-target', self.identity['target'], '-isysroot', str(self.sdk.parent),
            '-filelist', str(lists['QualificationHost']), str(self.object), '-L' + str(self.archive.parent),
            '-lBufferValueWitness', '-o', str(self.binary)]
        self.archive_argv = [self.tool['libtool'], '-static', '-arch_only', 'arm64',
            '-syslibroot', str(self.sdk.parent), '-filelist', str(lists['BufferValueWitness']), '-o', str(self.archive)]
        self.add_command('producer-build', [self.tool['swift'], 'build', '--build-system', 'native',
            '--target', 'VectorCore', '--triple', self.identity['target'], '--sdk', str(self.sdk.parent),
            '-c', 'release'], '', 0, executable=self.tool['swift'])
        self.add_command('app-build', [self.tool['xcodebuild'], '-configuration', 'Release', '-sdk', str(self.sdk.parent),
            '-destination', self.destination, '-scheme', 'QualificationHost', 'build'], self.build_log(), 0,
            executable=self.tool['xcodebuild'])
        self.add_command('install', [self.tool['xcrun'], 'simctl', 'install', self.destination.split('id=')[1],
            str(self.binary.parent)], '', 0, executable=self.tool['xcrun'])
        for spec, captured in zip(self.case_specs, self.capture_cases):
            invocation = str(uuid.uuid4())
            result = self.evidence / (spec['id'] + '.records')
            old = json.loads(Path(self.commands[spec['id']]['receipt']).read_text())
            body = Path(old['stdout']['path']).read_text()
            begin = {'invocationId': invocation, 'caseId': spec['id'], 'binarySHA256': sha(self.binary),
                     'runtime': self.identity['runtime'], 'platform': 'ios-simulator', 'architecture': 'arm64'}
            end = {'invocationId': invocation, 'caseId': spec['id'], 'numericalExit': old['exit']}
            result.write_text('VC_IOS_BEGIN ' + json.dumps(begin) + '\n' + body + 'VC_IOS_END ' + json.dumps(end) + '\n')
            app = {'kind': 'xcode-app', 'invocationId': invocation, 'caseId': spec['id'], 'result': str(result),
                'adapter': str(self.adapter), 'original': str(self.original), 'adapterKind': 'main',
                'adapterIdentity': verify_adapter(self.original.read_text(), self.adapter.read_text(), 'main'),
                'omittedWrites': False, 'hostSource': str(self.host), 'witnessSource': str(self.witness),
                'witnessArchive': str(self.archive), 'witnessModule': str(self.witness_module),
                'destination': self.destination, 'installCommand': 'install'}
            for row in (spec, captured):
                row.update(binary=str(self.binary), compileCommand='app-build', app=app)
            self.add_command(spec['command'], [self.tool['xcrun'], 'simctl', 'launch', '--terminate-running-process',
                self.destination.split('id=')[1], 'org.vectorcore.qualification.Host', '--invocation', invocation,
                '--case', spec['id']], 'Application launched successfully\n', 0, executable=self.tool['xcrun'])
        self.capture['commands'] = {k: v['receipt'] for k, v in self.commands.items()}
        self.save(repin=True)

    def build_log(self):
        return ''.join('    builtin-SwiftDriver -- ' + shlex.join(args) + '\n'
                       for args in self.swift_commands.values()) + '    ' + shlex.join(self.link) + '\n' + '    ' + shlex.join(self.archive_argv) + '\n'

    def save(self, *, repin=False):
        super().save(repin=repin)
        if repin:
            self.profile['pins'].update({str(p): sha(p) for p in self.source.rglob('*') if p.is_file()})
            write_json(self.profile_path, self.profile)
