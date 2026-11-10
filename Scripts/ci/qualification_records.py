"""Read candidate-bound captures using an explicitly reviewed data profile.

The profile is the trust anchor. Hash checks establish consistency of retained
files, not authenticity of a capture on a hostile filesystem. Evidence files are
never imported or executed. Canonical digest conflicts and mid-read changes are
covered by QualificationRecordTests.
"""
import hashlib
import json
import re
from pathlib import Path

POLICY = 'vc-0.4.0-xcode27.2-bounded-v1'
CASE_METADATA = ('id', 'format', 'command', 'producer', 'client', 'compileCommand', 'binary')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same(left, right):
    # JSON encoding distinguishes booleans from integers in identities/records.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


class Pins:
    def __init__(self, profile, evidence_root, source_root):
        captures = profile['captureRoots']
        require(type(captures) is dict and set(captures) == {'evidence', 'source'}, 'capture roots')
        self.roots = {'evidence': Path(evidence_root).resolve(), 'source': Path(source_root).resolve()}
        self.captures = {}
        for kind, value in captures.items():
            path = self.absolute(value)
            self.captures[kind] = path
        a, b = self.captures.values()
        require(not a.is_relative_to(b) and not b.is_relative_to(a), 'overlapping capture roots')
        self.pins = {}
        entries = profile['pins']
        require(type(entries) is dict and bool(entries), 'empty pin inventory')
        for name, digest in entries.items():
            require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest), 'invalid digest')
            path = self.resolve(name)
            require(path not in self.pins or self.pins[path] == digest, 'conflicting canonical digest: ' + str(path))
            self.pins.setdefault(path, digest)
        self.recheck()

    @staticmethod
    def absolute(value):
        require(type(value) is str, 'path is not a string')
        path = Path(value)
        require(path.is_absolute() and '..' not in path.parts, 'invalid absolute path: ' + value)
        return path

    def resolve(self, value):
        path = self.absolute(value)
        for kind, captured in self.captures.items():
            if path.is_relative_to(captured):
                return (self.roots[kind] / path.relative_to(captured)).resolve()
        return path.resolve()

    def checked(self, value):
        path = self.resolve(str(value))
        require(path in self.pins, 'file has no prior pin: ' + str(path))
        require(sha(path) == self.pins[path], 'hash mismatch: ' + str(path))
        return path

    def json(self, value):
        path = self.checked(value)
        result = read_json(path)
        self.checked(value)
        return result

    def recheck(self):
        for path, digest in self.pins.items():
            require(sha(path) == digest, 'hash mismatch: ' + str(path))


def named(rows, label):
    require(type(rows) is list and bool(rows), 'empty ' + label)
    result = {}
    for row in rows:
        require(type(row) is dict, 'invalid ' + label + ' row')
        name = row['id']
        require(type(name) is str and re.fullmatch(r'[a-zA-Z0-9_.:/-]+', name), 'invalid case id')
        require(name not in result, 'duplicate ' + label + ': ' + name)
        result[name] = row
    return result


def verify_sources(profile, capture, pins):
    path = pins.checked(capture['sourceManifest'])
    require(sha(path) == profile['sourceManifestSHA256'], 'source manifest digest differs')
    sources = pins.json(capture['sourceManifest'])
    require(type(sources) is dict and bool(sources), 'empty source inventory')
    root = pins.roots['source']
    actual = {'Package.swift'}
    for folder in ('Sources', 'Tests', 'Benchmarks'):
        directory = root / folder
        require(directory.is_dir(), 'missing compiled input root: ' + folder)
        actual.update(str(p.relative_to(root)) for p in directory.rglob('*') if p.is_file())
    require(actual == set(sources), 'compiled source inventory differs')
    for name, digest in sources.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'invalid source path')
        source = (root / relative).resolve()
        require(source.is_relative_to(root), 'source escapes root')
        require(sha(source) == digest, 'source hash mismatch: ' + name)
        # Source pins join the same first-registration-wins registry.
        require(source not in pins.pins or pins.pins[source] == digest, 'conflicting source digest')
        pins.pins.setdefault(source, digest)


def verify_command(expected, receipt, pins):
    record = pins.json(receipt)
    require(record['status'] == 'completed' and record['timeout'] is False, 'incomplete command')
    require(type(record['exit']) is int and type(expected['exit']) is int
            and record['exit'] == expected['exit'], 'unexpected command exit')
    for key in ('argv', 'cwd', 'environment'):
        require(same(record[key], expected[key]), 'command ' + key + ' differs')
    argv = record['argv']
    require(type(argv) is list and bool(argv) and all(type(v) is str for v in argv), 'invalid argv')
    executable = pins.checked(record['executable'])
    require(sha(executable) == record['executableSHA256'], 'executable digest differs')
    if Path(argv[0]).is_absolute():
        require(pins.resolve(argv[0]) == executable, 'argv executable differs')
    else:
        require(Path(argv[0]).name == argv[0] and executable.name == argv[0], 'unresolved executable name')
    streams = {}
    for name in ('stdout', 'stderr'):
        info = record[name]
        path = pins.checked(info['path'])
        data = path.read_bytes()
        require(type(info['bytes']) is int and info['bytes'] == len(data)
                and hashlib.sha256(data).hexdigest() == info['sha256'], 'stream identity differs')
        streams[name] = data.decode('utf-8')
    return {'record': record, **streams, 'text': streams['stdout'] + streams['stderr']}


def verify_bindings(profile, capture, commands, cases, pins):
    producers = capture['producers']
    require(same(producers, profile['producers']) and type(producers) is dict and bool(producers),
            'producer inventory differs')
    for producer in producers.values():
        for key in ('objects', 'modules'):
            paths = producer[key]
            require(type(paths) is list and bool(paths), 'empty producer ' + key)
            resolved = [pins.checked(path) for path in paths]
            require(len(resolved) == len(set(resolved)), 'duplicate producer ' + key)
    for case in cases.values():
        require(case['producer'] in producers and case['client'] in ('debug', 'release'), 'unknown configuration')
        producer = producers[case['producer']]
        binary = pins.checked(case['binary'])
        compile_record = commands[case['compileCommand']]['record']
        require(compile_record['exit'] == 0, 'failed consumer compilation')
        argv = compile_record['argv']
        objects = [pins.resolve(value) for value in argv if value.endswith('.o')]
        require(len(objects) == len(set(objects)) and set(objects) == {pins.resolve(v) for v in producer['objects']},
                'linked producer object inventory differs')
        require(argv.count('-o') == 1 and pins.resolve(argv[argv.index('-o') + 1]) == binary, 'consumer output differs')
        searches = [pins.resolve(argv[i + 1]) for i, value in enumerate(argv[:-1]) if value == '-I']
        for module in producer['modules']:
            require(any(pins.resolve(module).is_relative_to(search) for search in searches), 'producer module not selected')
        run_record = commands[case['command']]['record']
        require(pins.resolve(run_record['executable']) == binary, 'executed consumer differs')


def verify_capture(profile: dict, evidence_root: Path, source_root: Path) -> dict:
    require(type(profile) is dict and type(profile['schemaVersion']) is int
            and profile['schemaVersion'] == 1 and profile['policyId'] == POLICY, 'unsupported profile')
    require(type(profile['scope']) is list and bool(profile['scope'])
            and all(type(v) is str and v for v in profile['scope']), 'empty scope')
    pins = Pins(profile, evidence_root, source_root)
    capture_path = str(pins.captures['evidence'] / 'capture.json')
    capture = pins.json(capture_path)
    require(type(capture['schemaVersion']) is int and capture['schemaVersion'] == 1, 'capture schema')
    require(capture['collectionComplete'] is True and type(capture['invocationId']) is str
            and bool(capture['invocationId']), 'incomplete capture')
    require(same(capture['identity'], profile['identity']), 'compiler/platform identity differs')
    identity = capture['identity']
    require(identity['xcodeBuild'] == '27B52' and identity['compilerVersion'] == '6.4.2', 'unreviewed Xcode tuple')
    require(identity['swiftVersion'] == '6' and identity['architecture'] == 'arm64', 'unreviewed language/architecture')
    for field in ('platform', 'target', 'runtime'):
        require(bool(identity[field]), 'missing identity: ' + field)
    require(set(profile['identityFiles']) == {'frontendSHA256', 'sdkSettingsSHA256'}, 'identity file inventory')
    for field, path in profile['identityFiles'].items():
        require(sha(pins.checked(path)) == identity[field], 'identity file differs: ' + field)
    verify_sources(profile, capture, pins)
    require(type(capture['commands']) is dict and bool(capture['commands'])
            and set(capture['commands']) == set(profile['commands']), 'command inventory differs')
    commands = {}
    for name, expected in profile['commands'].items():
        receipt = capture['commands'][name]
        require(receipt == expected['receipt'], 'command receipt mapping differs')
        commands[name] = verify_command(expected, receipt, pins)
    cases = named(capture['cases'], 'captured cases')
    wanted = named(profile['cases'], 'profile cases')
    require(set(cases) == set(wanted), 'case inventory differs')
    for name, spec in wanted.items():
        require(same(cases[name], {k: spec[k] for k in CASE_METADATA}), 'case binding differs: ' + name)
    verify_bindings(profile, capture, commands, cases, pins)
    pins.recheck()
    return {'capture': capture, 'commands': commands, 'verifiedFiles': len(pins.pins),
            'sourceManifestSHA256': profile['sourceManifestSHA256']}
