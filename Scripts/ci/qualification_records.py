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
OPTIONAL_CASE_METADATA = ('witness', 'app')


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
            path = self.absolute(value).resolve()
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
        path = self.absolute(value).resolve()
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


def option(argv, flag):
    require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), 'missing/duplicate ' + flag)
    return argv[argv.index(flag) + 1]


def compile_flags(argv, mode, identity, profile, pins):
    require(option(argv, '-swift-version') == identity['swiftVersion'], 'language mode differs')
    require(option(argv, '-target') == identity['target'], 'target differs')
    require(pins.resolve(option(argv, '-sdk')) == pins.resolve(profile['identityFiles']['sdkSettingsSHA256']).parent,
            'SDK selection differs')
    require([arg for arg in argv if arg.startswith('-O')] == ['-Onone' if mode == 'debug' else '-O'],
            'unsupported optimization')
    allowed = {'-swift-version', '-target', '-sdk', '-module-cache-path', '-I', '-Xcc', '-parse-as-library',
               '-O', '-Onone', '-o', '-D', '-module-name', '-emit-module', '-emit-module-path', '-emit-object',
               '-emit-sil', '-emit-ir', '-typecheck', '-framework'}
    for arg in argv[1:]:
        require(not arg.startswith('-') or arg in allowed or arg.startswith(('-fmodule-map-file=', '-I/')),
                'unsupported compiler flag: ' + arg)


def verify_bindings(profile, capture, commands, cases, pins):
    app_observations = {}
    producers = capture['producers']
    require(same(producers, profile['producers']) and type(producers) is dict and bool(producers),
            'producer inventory differs')
    for producer in producers.values():
        kind = producer.get('kind', 'library')
        require(kind in ('standalone', 'library'), 'unknown producer kind')
        for key in ('objects', 'modules'):
            paths = producer[key]
            require(type(paths) is list and bool(paths) == (kind == 'library'), 'producer ' + key + ' inventory')
            resolved = [pins.checked(path) for path in paths]
            require(len(resolved) == len(set(resolved)), 'duplicate producer ' + key)
    for case in cases.values():
        require(case['producer'] in producers and case['client'] in ('debug', 'release'), 'unknown configuration')
        producer = producers[case['producer']]
        binary = pins.checked(case['binary'])
        compile_record = commands[case['compileCommand']]['record']
        require(compile_record['exit'] == 0, 'failed consumer compilation')
        if 'app' in case:
            from qualification_ios import verify_ios_case
            app_observations[case['id']] = verify_ios_case(case, producer, capture['identity'], profile, commands, pins)
            continue
        argv = compile_record['argv']
        compile_flags(argv, case['client'], capture['identity'], profile, pins)
        extra_objects, extra_modules = [], []
        if 'witness' in case:
            witness = case['witness']
            require(set(witness) == {'command', 'object', 'module'}, 'witness binding fields')
            obj, module = pins.checked(witness['object']), pins.checked(witness['module'])
            emitted = commands[witness['command']]['record']
            require(emitted['exit'] == 0, 'witness compilation failed')
            compile_flags(emitted['argv'], case['client'], capture['identity'], profile, pins)
            require(pins.resolve(option(emitted['argv'], '-o')) == obj
                    and pins.resolve(option(emitted['argv'], '-emit-module-path')) == module,
                    'witness output differs')
            require('-emit-object' in emitted['argv'] and '-emit-module' in emitted['argv'], 'witness emission missing')
            extra_objects.append(obj)
            extra_modules.append(module)
        objects = [pins.resolve(value) for value in argv if value.endswith('.o')]
        require(len(objects) == len(set(objects))
                and set(objects) == {pins.resolve(v) for v in producer['objects']} | set(extra_objects),
                'linked producer object inventory differs')
        require(argv.count('-o') == 1 and pins.resolve(argv[argv.index('-o') + 1]) == binary, 'consumer output differs')
        searches = [pins.resolve(argv[i + 1]) for i, value in enumerate(argv[:-1]) if value == '-I']
        for module in [*producer['modules'], *map(str, extra_modules)]:
            require(any(pins.resolve(module).is_relative_to(search) for search in searches), 'producer module not selected')
        if 'modulemap' in producer:
            modulemap = pins.checked(producer['modulemap'])
            maps = [pins.resolve(arg.split('=', 1)[1]) for arg in argv if arg.startswith('-fmodule-map-file=')]
            require(maps == [modulemap], 'producer module map differs')
        run_record = commands[case['command']]['record']
        require(pins.resolve(run_record['executable']) == binary, 'executed consumer differs')
    invocations = [row['app']['invocationId'] for row in cases.values() if 'app' in row]
    require(len(set(invocations)) == len(invocations), 'reused app invocation')
    return app_observations


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
        fields = (*CASE_METADATA, *(k for k in OPTIONAL_CASE_METADATA if k in spec))
        require(same(cases[name], {k: spec[k] for k in fields}), 'case binding differs: ' + name)
    app_observations = verify_bindings(profile, capture, commands, cases, pins)
    checks = verify_checks(profile, capture, commands, pins)
    pins.recheck()
    return {'capture': capture, 'commands': commands, 'checks': checks, 'appObservations': app_observations,
            'verifiedFiles': len(pins.pins),
            'sourceManifestSHA256': profile['sourceManifestSHA256']}


def verify_checks(profile, capture, commands, pins):
    actual, wanted = capture.get('checks', []), profile.get('checks', [])
    require(type(actual) is list and type(wanted) is list, 'invalid auxiliary checks')
    if not actual and not wanted:
        return []
    captured, reviewed = named(actual, 'captured checks'), named(wanted, 'reviewed checks')
    require(set(captured) == set(reviewed), 'auxiliary check inventory differs')
    results = []
    for name, spec in reviewed.items():
        require(same(captured[name], {k: v for k, v in spec.items() if k != 'expected'}), 'auxiliary binding differs')
        kind = spec['format']
        require(kind in ('native-suite', 'migration-contract', 'c-build-modes'), 'unsupported auxiliary format: ' + str(kind))
        command = commands[spec['command']]
        result = {'spec': spec, 'command': command}
        if kind == 'native-suite':
            producer = capture['producers'][spec['producer']]
            require(producer['buildCommand'] == spec['command'] and producer['testBinary'] == spec['binary'],
                    'native suite producer differs')
            pins.checked(spec['binary'])
            argv = command['record']['argv']
            require('test' in argv and not any(v in argv for v in ('--filter', '--skip', '--specifier', '--skip-build')),
                    'native suite selection differs')
            result['xml'] = pins.checked(spec['xunit']).read_text()
            result['listing'] = commands[spec['listCommand']]['stdout']
        elif kind == 'migration-contract':
            compile_flags(command['record']['argv'], spec['client'], capture['identity'], profile, pins)
        else:
            require(command['record']['exit'] == 0, 'C matrix driver failed')
            result['cells'] = verify_c_build_modes(spec, pins, capture['identity'])
        results.append(result)
    return results


def verify_c_build_modes(spec, pins, identity):
    from native_qualification import assess_routes, route_link_binding
    result = pins.json(spec['result'])
    require(result['schemaVersion'] == 1 and result['marker'] == 'V02_BUILD_MODES_COMPLETE', 'C matrix incomplete')
    cells = result['cells']
    require(type(cells) is list and len(cells) == 4, 'C matrix cell inventory differs')
    require({(row['configuration'], row['cKernels']) for row in cells}
            == {(m, c) for m in ('debug', 'release') for c in (False, True)}, 'C matrix cell inventory differs')
    root = Path(spec['result']).parent

    def digest(path, expected):
        require(sha(pins.checked(path)) == expected, 'C artifact identity differs')

    def receipt(directory, name, code):
        record = pins.json(str(directory / (name + '.json')))
        require(record['status'] == 'completed' and record['timeout'] is False
                and type(record['exit']) is int and record['exit'] == code, 'C command incomplete/failed: ' + name)
        require(pins.resolve(record['cwd']) == pins.roots['source'], 'C source checkout differs')
        environment = record['environment']
        require(pins.resolve(environment['SDKROOT']) == pins.resolve(identity['sdk']), 'C SDK selection differs')
        executable = record['argv'][0]
        if not Path(executable).is_absolute():
            require(executable in ('swift', 'swiftc'), 'unresolved C matrix tool')
            executable = str(Path(environment['PATH'].split(':')[0]) / executable)
        pins.checked(executable)
        text = {}
        for name in ('stdout', 'stderr'):
            stream = record[name]
            path = pins.checked(stream['path'])
            data = path.read_bytes()
            require(hashlib.sha256(data).hexdigest() == stream['sha256'] and len(data) == stream['bytes'],
                    'C command stream differs')
            text[name] = data.decode('utf-8')
        return record, text

    for row in cells:
        mode, enabled = row['configuration'], row['cKernels']
        require(type(enabled) is bool, 'C flag type differs')
        name = mode + ('-c-on' if enabled else '-c-off')
        require(row['name'] == name and row['routeRows'] == 36 and row['directCTails'] == 10, 'C cell scope differs')
        directory = root / name
        require(Path(row['artifactManifest']) == directory / 'artifacts.json'
                and Path(row['linkProvenance']) == directory / 'link-provenance.json', 'C cell artifact binding differs')
        artifacts = pins.json(row['artifactManifest'])
        require(bool(artifacts) and len({a['buildPath'] for a in artifacts}) == len(artifacts), 'C artifact inventory differs')
        for artifact in artifacts:
            digest(artifact['buildPath'], artifact['sha256'])
            digest(artifact['retainedPath'], artifact['sha256'])
        build, _ = receipt(directory, 'build', 0)
        require(build['argv'][1] == 'build' and option(build['argv'], '-c') == mode
                and ('-DVC_USE_C_KERNELS' in build['argv']) == enabled, 'C producer configuration differs')
        provenance = pins.json(row['linkProvenance'])
        for key in ('originalDispatcher', 'originalSource', 'derivativeSource', 'consumerSource', 'shimSource', 'consumer'):
            item = provenance[key]
            digest(item['path'], item['sha256'])
        for item in provenance['linkedObjects']:
            digest(item['path'], item['sha256'])
        require(row['consumerSHA256'] == provenance['consumer']['sha256'], 'C consumer identity differs')
        link, _ = receipt(directory, 'link-public-routes', 0)
        executed, streams = receipt(directory, 'public-routes', 0)
        route_link_binding(link, executed, provenance, mode, enabled)
        assess_routes(streams['stdout'], enabled)
        require(streams['stderr'] == '', 'unexpected C route diagnostics')
        for request in ('absent', 'matching'):
            receipt(directory, 'benchmark-' + request, 0)
            benchmark = pins.json(str(directory / (request + '-report.json')))
            metadata = benchmark['metadata']
            require(metadata['coreConfiguration'] == mode and metadata['benchmarkConfiguration'] == mode
                    and metadata['flags']['useCKernels'] is enabled
                    and metadata['flags'].get('requestedCKernels') is (enabled if request == 'matching' else None),
                    'C benchmark product/request differs')
            require(metadata['selectedCaseIDs'] == ['dist.euclidean.512.optimized'] and benchmark['failures'] == []
                    and len(benchmark['results']) == 1 and benchmark['results'][0]['iterations'] == 2,
                    'C benchmark correctness/completion differs')
        _, rejected = receipt(directory, 'benchmark-mismatch', 1)
        require('does not match compiled' in rejected['stderr'], 'C mismatched request was not rejected')
    return [{'configuration': row['configuration'], 'cKernels': row['cKernels'], 'routes': 36, 'tails': 10} for row in cells]
