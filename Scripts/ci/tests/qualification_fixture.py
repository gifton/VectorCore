"""Small synthetic captures; expected observations are literal, not collected."""
from pathlib import Path
import copy
import hashlib
import json

POLICY = 'vc-0.4.0-xcode27.2-bounded-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


class CaptureFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.source = self.root / 'source'
        self.evidence = self.root / 'evidence'
        self.source.mkdir()
        self.evidence.mkdir()
        for name in ('Sources', 'Tests', 'Benchmarks'):
            (self.source / name).mkdir()
        (self.source / 'Package.swift').write_text('// fixture package\n')
        (self.source / 'Sources/Library.swift').write_text('let fixture: Int = 1\n')
        self.compiler = self.evidence / 'compiler'
        self.compiler.write_text('synthetic compiler identity; never executed\n')
        self.binary = self.evidence / 'consumer'
        self.binary.write_text('synthetic consumer identity; never executed\n')
        self.object = self.evidence / 'Library.o'
        self.object.write_bytes(b'fixture object\n')
        self.module = self.evidence / 'Library.swiftmodule'
        self.module.write_bytes(b'fixture module\n')
        self.sdk = self.evidence / 'SDKSettings.json'
        write_json(self.sdk, {'Version': '27.2'})
        self.manifest = self.evidence / 'source-inputs.json'
        write_json(self.manifest, {str(p.relative_to(self.source)): sha(p)
                                  for p in self.source.rglob('*') if p.is_file()})
        self.identity = {'xcodeBuild': '27B52', 'compilerVersion': '6.4.2',
                         'frontendSHA256': sha(self.compiler), 'sdkSettingsSHA256': sha(self.sdk),
                         'platform': 'macos', 'architecture': 'arm64',
                         'target': 'arm64-apple-macosx14.0', 'swiftVersion': '6',
                         'runtime': {'version': '27.0.1', 'build': '26A434'}}
        self.commands = {}
        self.case_specs = []
        self.capture_cases = []
        self.profile_path = self.root / 'profile.json'
        self.capture_path = self.evidence / 'capture.json'
        # This fixture covers three declared groups, not the whole VectorCore matrix.
        self.add_command('compile', [str(self.compiler), '-O', '-swift-version', '6',
            '-target', self.identity['target'], '-sdk', str(self.sdk.parent),
            str(self.object), '-I', str(self.module.parent), '-o', str(self.binary)], '', 0,
            executable=self.compiler)
        self.add_case('supported', 'supported', '1', '1')
        self.add_case('omitted', 'negative-control', '1', '2', negative_for='supported')
        self.add_case('compiler', 'diagnostic', '1', '2', compiler_probe=True)
        self.capture = {'schemaVersion': 1, 'invocationId': 'fixture-run-01',
                        'collectionComplete': True, 'identity': copy.deepcopy(self.identity),
                        'sourceManifest': str(self.manifest),
                        'commands': {k: str(v['receipt']) for k, v in self.commands.items()},
                        'cases': self.capture_cases,
                        'producers': {'release': {'objects': [str(self.object)],
                                                  'modules': [str(self.module)]}}}
        self.profile = {'schemaVersion': 1, 'policyId': POLICY,
                        'captureRoots': {'evidence': str(self.evidence), 'source': str(self.source)},
                        'sourceManifestSHA256': sha(self.manifest),
                        'identity': copy.deepcopy(self.identity),
                        'scope': ['synthetic fixture: macos/arm64/release'],
                        'commands': self.commands, 'cases': self.case_specs,
                        'producers': copy.deepcopy(self.capture['producers']),
                        'identityFiles': {'frontendSHA256': str(self.compiler),
                                          'sdkSettingsSHA256': str(self.sdk)},
                        'pins': {}}
        self.save(repin=True)

    def add_command(self, name, argv, text, code, *, executable=None):
        stdout = self.evidence / (name + '.stdout')
        stderr = self.evidence / (name + '.stderr')
        stdout.write_text(text)
        stderr.write_text('')
        executable = executable or self.binary
        receipt = self.evidence / (name + '.json')
        record = {'argv': argv, 'cwd': str(self.source), 'status': 'completed',
                  'deadlineSeconds': 15, 'cleanupDeadlineSeconds': 2,
                  'timeout': False, 'executable': str(executable),
                  'executableSHA256': sha(executable), 'environment': {},
                  'elapsedSeconds': 0.01, 'exit': code,
                  'stdout': {'path': str(stdout), 'sha256': sha(stdout), 'bytes': stdout.stat().st_size},
                  'stderr': {'path': str(stderr), 'sha256': sha(stderr), 'bytes': 0}}
        write_json(receipt, record)
        self.commands[name] = {'receipt': str(receipt), 'argv': argv, 'cwd': str(self.source),
                               'environment': {}, 'exit': code}

    def add_case(self, name, role, correct, observed, *, negative_for=None, compiler_probe=False):
        count = int(correct != observed)
        text = f'BUFFER_VALUE first observed={observed}\nBUFFER_VALUE_COMPLETE count=1 failures={count}\n'
        self.add_command(name, [str(self.binary), name], text, count)
        spec = {'id': name, 'format': 'buffer-value', 'role': role, 'command': name,
                'producer': 'release', 'client': 'release', 'compileCommand': 'compile',
                'binary': str(self.binary), 'correct': {'first': correct},
                'observed': {'first': observed}, 'compilerProbe': compiler_probe}
        if negative_for:
            spec['negativeFor'] = negative_for
        self.case_specs.append(spec)
        self.capture_cases.append({k: v for k, v in spec.items()
                                   if k not in ('correct', 'observed', 'role', 'negativeFor', 'compilerProbe')})

    def save(self, *, repin=False):
        write_json(self.capture_path, self.capture)
        if repin:
            # Test fixture author pins the synthetic capture independently of verifier logic.
            self.profile['pins'] = {str(p): sha(p) for p in self.evidence.rglob('*') if p.is_file()}
        write_json(self.profile_path, self.profile)

    def alter_receipt(self, name, **fields):
        path = Path(self.commands[name]['receipt'])
        record = json.loads(path.read_text())
        record.update(fields)
        write_json(path, record)

    def alter_stdout(self, name, text, *, code=None):
        path = Path(self.commands[name]['receipt'])
        record = json.loads(path.read_text())
        stream = Path(record['stdout']['path'])
        stream.write_text(text)
        record['stdout'].update(sha256=sha(stream), bytes=stream.stat().st_size)
        if code is not None:
            record['exit'] = code
        write_json(path, record)
