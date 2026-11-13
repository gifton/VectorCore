#!/usr/bin/env python3
"""App-hosted qualification plumbing. Launch success is not numerical acceptance.

IOSQualificationTests covers destination separation, preserved fixture helpers,
fresh invocation binding and complete numerical records. Adapted fixtures are
new app-hosted compositions; their results do not replace native diagnostics.
"""
import hashlib
import json
import re
import uuid
from pathlib import Path
import shlex

from compiler_qualification import observations
from qualification_records import require, same, unique


def destination_config(destination):
    require(type(destination) is str, 'destination must be explicit')
    fields = {}
    for token in destination.split(','):
        parts = token.split('=', 1)
        require(len(parts) == 2 and parts[0] not in fields, 'ambiguous destination')
        fields[parts[0]] = parts[1]
    require(set(fields) == {'platform', 'id'} and fields['platform'] in ('iOS', 'iOS Simulator'),
            'explicit iOS device or simulator identifier required')
    require(re.fullmatch(r'[a-fA-F0-9-]{16,40}', fields['id']) is not None, 'invalid destination identifier')
    simulator = fields['platform'] == 'iOS Simulator'
    if simulator:
        require(str(uuid.UUID(fields['id'])) == fields['id'].lower(), 'invalid simulator UUID')
    return {'identifier': fields['id'], 'platform': 'ios-simulator' if simulator else 'ios-device',
            'sdk': 'iphonesimulator' if simulator else 'iphoneos',
            'target': 'arm64-apple-ios17.0' + ('-simulator' if simulator else '')}


def configurations(producer, client):
    require(producer in ('debug', 'release') and client in ('debug', 'release'), 'unsupported optimization mode')
    return {'producer': producer, 'client': client, 'clientFlag': '-O' if client == 'release' else '-Onone'}


def new_invocation():
    return str(uuid.uuid4())


def adapt_fixture(original, kind):
    if kind == 'main':
        marker = re.search(r'@main\s+private struct [A-Za-z0-9_]+\s*\{', original)
        require(marker is not None and original.count('@main') == 1, 'unexpected fixture entry point')
        prefix, entry = original[:marker.start()], original[marker.end():]
        require(entry.count('static func main() {') == 1, 'unexpected fixture main signature')
        require(len(re.findall(r'        exit\(([^\n]+)\)\n', entry)) == 1, 'unexpected fixture termination')
        entry = entry.replace('static func main() {', 'public static func run() -> Int32 {', 1)
        entry = re.sub(r'        exit\(([^\n]+)\)\n', r'        return \1\n', entry, count=1)
        return prefix + 'public enum HostedProbe {' + entry
    require(kind == 'standalone', 'unknown fixture adapter')
    require(original.count('let observed = ') == 1 and '@main' not in original, 'unexpected standalone entry point')
    prefix, body = original.split('let observed = ', 1)
    return (prefix + 'public enum HostedProbe {\npublic static func run() -> Int32 {\n'
            + 'let observed = ' + body + '\nreturn 0\n}\n}\n')


def verify_adapter(original, adapted, kind):
    require(adapted == adapt_fixture(original, kind), 'fixture adapter changed helper or numerical body')
    prefix = original.split('@main' if kind == 'main' else 'let observed = ', 1)[0]
    require(adapted.startswith(prefix), 'fixture helper prefix changed')
    return {'originalSHA256': hashlib.sha256(original.encode()).hexdigest(),
            'adapterSHA256': hashlib.sha256(adapted.encode()).hexdigest(),
            'preservedPrefixSHA256': hashlib.sha256(prefix.encode()).hexdigest()}


def read_app_frame(text, invocation_id, binary_sha256, runtime, platform, case_id):
    require(str(uuid.UUID(invocation_id)) == invocation_id, 'invalid invocation identity')
    require(re.fullmatch(r'[0-9a-f]{64}', binary_sha256) is not None, 'invalid binary digest')
    require(platform in ('ios-simulator', 'ios-device'), 'unsupported app platform')
    lines = text.splitlines()
    require(len(lines) >= 3 and lines[0].startswith('VC_IOS_BEGIN ') and lines[-1].startswith('VC_IOS_END '),
            'app body or completion missing')
    require(sum(line.startswith('VC_IOS_BEGIN ') for line in lines) == 1
            and sum(line.startswith('VC_IOS_END ') for line in lines) == 1, 'duplicate app execution frame')
    begin = json.loads(lines[0][len('VC_IOS_BEGIN '):], object_pairs_hook=unique)
    end = json.loads(lines[-1][len('VC_IOS_END '):], object_pairs_hook=unique)
    expected = {'invocationId': invocation_id, 'caseId': case_id, 'binarySHA256': binary_sha256,
                'runtime': runtime, 'platform': platform, 'architecture': 'arm64'}
    require(same(begin, expected), 'app invocation, binary or runtime differs')
    require(set(end) == {'invocationId', 'caseId', 'numericalExit'} and end['invocationId'] == invocation_id
            and end['caseId'] == case_id and type(end['numericalExit']) is int and end['numericalExit'] in (0, 1),
            'app completion identity/exit differs')
    return begin, '\n'.join(lines[1:-1]) + '\n', end['numericalExit']


def parse_app_result(text, invocation_id, binary_sha256, runtime, platform, case):
    begin, body, code = read_app_frame(text, invocation_id, binary_sha256, runtime, platform, case['id'])
    result = observations(case, {'text': body, 'record': {'exit': code}})
    return {**result, 'numericalExit': code, 'appIdentity': begin}


def xcode_commands(text):
    """Extract the actual two Swift driver invocations, app link and witness archive.

    IOSQualificationTests rejects missing/duplicate modules and links. Receipts
    retain the complete build log, including the commands ignored here.
    """
    from qualification_records import option
    swift, links, archives = {}, [], []
    for line in text.splitlines():
        if 'builtin-SwiftDriver -- ' in line:
            argv = shlex.split(line.split('builtin-SwiftDriver -- ', 1)[1])
            name = option(argv, '-module-name')
            require(name in ('QualificationHost', 'BufferValueWitness') and name not in swift,
                    'unexpected or duplicate Swift module')
            swift[name] = argv
        elif re.match(r'^\s+/\S*/clang\s', line):
            argv = shlex.split(line)
            if '-o' in argv and option(argv, '-o').endswith('/QualificationHost.app/QualificationHost'):
                links.append(argv)
        elif re.match(r'^\s+/\S*/libtool\s', line):
            argv = shlex.split(line)
            if '-o' in argv and option(argv, '-o').endswith('/libBufferValueWitness.a') and '-static' in argv:
                archives.append(argv)
    require(set(swift) == {'QualificationHost', 'BufferValueWitness'} and len(links) == len(archives) == 1,
            'incomplete or ambiguous Xcode build commands')
    return {'swift': swift, 'link': links[0], 'archive': archives[0]}


def verify_xcode_swift_flags(argv, mode, identity):
    from qualification_records import option
    require(option(argv, '-target') == identity['target'], 'app target differs')
    require(Path(option(argv, '-sdk')).resolve() == Path(identity['sdk']).resolve(), 'app SDK differs')
    require(option(argv, '-swift-version') == identity['swiftVersion'], 'app language differs')
    require([v for v in argv if v.startswith('-O')] == ['-O' if mode == 'release' else '-Onone'],
            'app optimization differs')


def verify_ios_case(case, producer, identity, profile, commands, pins):
    """Bind an app observation to its producer, separate witness, link and launch.

    IOSBindingTests exercises this through the shared compiler checker with
    repinned SDK/target/mode, object, import, adapter and result substitutions.
    """
    from qualification_records import option, sha, verify_swift_tool
    app = case['app']
    require(app['kind'] == 'xcode-app', 'unknown app binding')
    destination = destination_config(app['destination'])
    require(identity['platform'] == destination['platform'] and identity['target'] == destination['target'],
            'app platform/target differs from destination')
    require(type(app['omittedWrites']) is bool, 'invalid omitted-write mode')
    binary = pins.checked(case['binary'])
    original, adapter = pins.checked(app['original']), pins.checked(app['adapter'])
    require(same(verify_adapter(original.read_text(), adapter.read_text(), app['adapterKind']), app['adapterIdentity']),
            'adapter identity differs')
    root = pins.captures['source']
    host, witness = pins.checked(app['hostSource']), pins.checked(app['witnessSource'])
    require(sha(host) == sha(pins.checked(str(root / 'Scripts/ci/fixtures/compiler-qualification-ios/App.swift'))),
            'app host differs from reviewed source')
    require(sha(witness) == sha(pins.checked(str(root / 'Scripts/ci/fixtures/compile-contracts/BufferValueWitness.swift'))),
            'witness source differs from reviewed source')
    producer_record = commands[producer['buildCommand']]['record']
    args = producer_record['argv']
    require(producer_record['exit'] == 0 and args[1] == 'build' and option(args, '--target') == 'VectorCore'
            and option(args, '-c') == producer.get('configuration', case['producer'])
            and option(args, '--triple') == identity['target']
            and pins.resolve(option(args, '--sdk')) == pins.resolve(identity['sdk']), 'iOS producer build differs')
    require(not any(arg in args for arg in ('-Xswiftc', '-Xcc', '-Xlinker', '--skip-build')), 'producer flag override')
    verify_swift_tool(args, 'swift', profile, pins, producer_record['environment'])
    build = commands[case['compileCommand']]
    args = build['record']['argv']
    require(option(args, '-configuration') == case['client'].title()
            and option(args, '-destination') == app['destination']
            and option(args, '-scheme') == 'QualificationHost' and 'build' in args
            and pins.resolve(option(args, '-sdk')) == pins.resolve(identity['sdk']), 'app build configuration differs')
    built = xcode_commands(build['text'])
    emitted_objects = {}
    for name, argv in built['swift'].items():
        verify_swift_tool(argv, 'swiftc', profile, pins)
        verify_xcode_swift_flags(argv, case['client'], identity)
        defines = []
        for index, value in enumerate(argv):
            if index > 0 and argv[index - 1] == '-Xcc':
                continue
            if value == '-D':
                require(index + 1 < len(argv), 'missing Swift condition')
                defines.append(argv[index + 1])
            elif value.startswith('-D'):
                defines.append(value[2:])
        require(defines.count('OMIT_WRITES') == int(app['omittedWrites']), 'app/witness control compilation differs')
        searches = [pins.resolve(argv[i + 1]) for i, v in enumerate(argv[:-1]) if v == '-I']
        for module in producer['modules']:
            require(any(pins.resolve(module).is_relative_to(search) for search in searches), 'iOS producer module not selected')
        if 'modulemap' in producer:
            maps = [pins.resolve(v.split('=', 1)[1]) for v in argv if v.startswith('-fmodule-map-file=')]
            require(maps == [pins.checked(producer['modulemap'])], 'iOS producer module map differs')
        lists = [v[1:] for v in argv if v.startswith('@')]
        require(len(lists) == 1, 'Swift source filelist differs')
        sources = [pins.checked(p) for p in shlex.split(pins.checked(lists[0]).read_text())]
        wanted = [host, adapter] if name == 'QualificationHost' else [witness]
        require(len(sources) == len(wanted) and set(sources) == set(wanted), 'app/witness compiled source inventory differs')
        outputs = pins.json(option(argv, '-output-file-map'))
        require({pins.resolve(p) for p in outputs if p} == set(sources), 'Swift output map source differs')
        emitted_objects[name] = [pins.checked(v['object']) for k, v in outputs.items() if k]
    link, archive = built['link'], built['archive']
    pins.checked(link[0])
    pins.checked(archive[0])
    require(option(link, '-target') == identity['target']
            and pins.resolve(option(link, '-isysroot')) == pins.resolve(identity['sdk'])
            and pins.resolve(option(link, '-o')) == binary and '-dynamiclib' not in link, 'app link identity differs')
    lto_outputs = set()
    for index, arg in enumerate(link):
        if arg == '-object_path_lto':
            require(index > 0 and link[index - 1] == '-Xlinker'
                    and index + 2 < len(link) and link[index + 1] == '-Xlinker', 'malformed LTO output option')
            lto_outputs.add(index + 2)
    linked_producer = [pins.checked(arg) for index, arg in enumerate(link)
                       if arg.endswith('.o') and index not in lto_outputs]
    require(len(linked_producer) == len(producer['objects'])
            and set(linked_producer) == {pins.checked(p) for p in producer['objects']}, 'linked iOS producer differs')
    for argv, name in ((link, 'QualificationHost'), (archive, 'BufferValueWitness')):
        linked = [pins.checked(p) for p in shlex.split(pins.checked(option(argv, '-filelist')).read_text())]
        require(len(linked) == len(emitted_objects[name]) and set(linked) == set(emitted_objects[name]),
                'app/witness object binding differs')
    library, module = pins.checked(app['witnessArchive']), pins.checked(app['witnessModule'])
    require(option(archive, '-arch_only') == 'arm64'
            and pins.resolve(option(archive, '-syslibroot')) == pins.resolve(identity['sdk'])
            and pins.resolve(option(archive, '-o')) == library, 'witness archive binding differs')
    require(pins.resolve(option(built['swift']['BufferValueWitness'], '-emit-module-path')) == module,
            'witness emitted module differs')
    # Xcode copies the emitted module into its product directory before the
    # dependent target compiles. Bind both copies and the host's search path.
    suffix = '-simulator' if identity['platform'] == 'ios-simulator' else ''
    imported = library.parent / 'BufferValueWitness.swiftmodule' / ('arm64-apple-ios' + suffix + '.swiftmodule')
    require(sha(pins.checked(str(imported))) == sha(module), 'imported witness module differs')
    host_args = built['swift']['QualificationHost']
    searches = [pins.resolve(host_args[i + 1]) for i, v in enumerate(host_args[:-1]) if v == '-I']
    require(library.parent in searches and link.count('-lBufferValueWitness') == 1
            and library.parent in [pins.resolve(v[2:]) for v in link if v.startswith('-L/')],
            'witness library/module not selected')
    install = commands[app['installCommand']]['record']
    launch = commands[case['command']]['record']
    require(install['exit'] == launch['exit'] == 0, 'app lifecycle command failed')
    bundle, identifier = 'org.vectorcore.qualification.Host', destination['identifier']
    if identity['platform'] == 'ios-simulator':
        install_args = ['simctl', 'install', identifier, str(Path(case['binary']).parent)]
        launch_args = ['simctl', 'launch', '--terminate-running-process', identifier, bundle]
    else:
        install_args = ['devicectl', 'device', 'install', 'app', '--device', identifier, str(Path(case['binary']).parent)]
        launch_args = ['devicectl', 'device', 'process', 'launch', '--device', identifier, '--terminate-existing', bundle]
    require(install['argv'][1:] == install_args and launch['argv'][1:] == [*launch_args,
            '--invocation', app['invocationId'], '--case', app['caseId']], 'installed/launched app differs')
    result = pins.checked(app['result']).read_text()
    _, body, numerical_exit = read_app_frame(result, app['invocationId'], sha(binary), identity['runtime'],
                                             identity['platform'], app['caseId'])
    return {'text': body, 'record': {'exit': numerical_exit}}


PROBES = (
    ('by-value', 'Scripts/ci/fixtures/compile-contracts/BufferValueConsumer.swift', 'main', False),
    ('by-value-omitted', 'Scripts/ci/fixtures/compile-contracts/BufferValueConsumer.swift', 'main', True),
    ('exposure', 'Scripts/ci/fixtures/compile-contracts/BufferInoutExposure.swift', 'main', False),
    ('residual', 'Docs/reviews/2026-11-06-workaround-qualification/ResidualExposure.swift', 'main', False),
    ('residual-omitted', 'Docs/reviews/2026-11-06-workaround-qualification/ResidualExposure.swift', 'main', True),
) + tuple((name, f'Docs/reviews/{folder}/probes/{name}.swift', 'standalone', False)
          for folder, names in (
              ('2026-11-04-release-phase-one/compiler', ('ByValueReadAfter', 'CopyInoutReturnInside',
                  'NoWriteNegative', 'NonLiteralReadAfter', 'OutOfLineRead')),
              ('2026-11-01-compiler-qualification', ('ByValueReturnInside', 'LostStoreObserved', 'ReturnInsideInout')))
          for name in names)


def collect_ios(source_root: Path, developer_dir: Path, destination: str, output: Path,
                producer_mode: str, client_mode: str, *, team=None, allow_device_install=False) -> Path:
    """Collect one fresh producer/client cell; observations remain unreviewed."""
    import shutil
    import time
    import swiftpm_products as products
    from capture_compiler_qualification import NativeCapture, source_manifest
    from qualification_records import option, sha
    from v03_process import write_json

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    settings = destination_config(destination)
    configurations(producer_mode, client_mode)
    physical = settings['platform'] == 'ios-device'
    require(not physical or (allow_device_install and team and re.fullmatch('[A-Z0-9]{10}', team)),
            'physical installation requires explicit authorization and development team')
    developer = Path(developer_dir).resolve()
    swiftc = developer / 'Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc'
    sdk = developer / ('Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk' if physical else
                       'Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk')
    collector = NativeCapture(source_root, swiftc, sdk, output, 4)
    collector.preflight()
    # The package manifest executes on macOS; the --sdk/--triple flags select
    # iOS for the producer. Global SDKROOT would break the host manifest build.
    collector.env.pop('SDKROOT', None)
    output.mkdir(parents=True, exist_ok=False)
    capture = collector.capture
    build_tools = set()
    before = source_manifest(collector.source)
    write_json(output / 'source-inputs.json', before)
    capture['sourceManifest'] = str(output / 'source-inputs.json')
    write_json(output / 'capture.json', capture)
    xcrun = ['/usr/bin/xcrun']

    def command(name, args, timeout=60, expected=(0,)):
        return collector.command(name, args, timeout=timeout, expected=expected)

    def stdout(record):
        return Path(record['stdout']['path']).read_text()

    try:
        identity = collector.identity()
        identity.update(platform=settings['platform'], target=settings['target'])
        if physical:
            inventory = output / 'devices.json'
            command('device-inventory', [*xcrun, 'devicectl', 'list', 'devices', '--timeout', '15',
                    '--json-output', inventory], 30)
            devices = json.loads(inventory.read_text())['result']['devices']
            matches = [row for row in devices if row['identifier'] == settings['identifier']
                       or row.get('hardwareProperties', {}).get('udid') == settings['identifier']]
            require(len(matches) == 1, 'selected physical device not found')
            device = matches[0]
            hardware, properties = device['hardwareProperties'], device['deviceProperties']
            require(hardware['reality'] == 'physical', 'selected device is not physical')
            identity['deviceModel'] = hardware['productType']
            runtime = {'version': properties['osVersionNumber'],
                       'description': f"Version {properties['osVersionNumber']} (Build {properties['osBuildUpdate']})"}
        else:
            record, _ = command('simulator-devices', [*xcrun, 'simctl', 'list', 'devices', 'available', '-j'])
            devices = json.loads(stdout(record))['devices']
            matches = [(runtime_id, row) for runtime_id, rows in devices.items() for row in rows
                       if row['udid'].lower() == settings['identifier'].lower()]
            require(len(matches) == 1, 'selected simulator not found')
            runtime_id, device = matches[0]
            record, _ = command('simulator-runtimes', [*xcrun, 'simctl', 'list', 'runtimes', '-j'])
            runtimes = json.loads(stdout(record))['runtimes']
            matches = [r for r in runtimes if r['identifier'] == runtime_id and r['isAvailable']]
            require(len(matches) == 1, 'selected simulator runtime unavailable')
            runtime_info = matches[0]
            runtime = {'version': runtime_info['version'],
                       'description': f"Version {runtime_info['version']} (Build {runtime_info['buildversion']})"}
            if device['state'] != 'Booted':
                command('simulator-boot', [*xcrun, 'simctl', 'boot', settings['identifier']])
            command('simulator-bootstatus', [*xcrun, 'simctl', 'bootstatus', settings['identifier'], '-b'], 180)
        identity['runtime'] = runtime
        capture['identity'] = identity
        capture['destination'] = destination
        flags = ['--build-system', 'native', '--scratch-path', output / 'producer', '--sdk', sdk,
                 '--triple', settings['target'], '-c', producer_mode, '-j', '4']
        command('producer-build', [swiftc.with_name('swift'), 'build', *flags, '--target', 'VectorCore'], 1200)
        record, _ = command('producer-bin-path', [swiftc.with_name('swift'), 'build', *flags, '--show-bin-path'])
        build = products.parse_bin_path(stdout(record))
        objects = products.link_objects(build, ['VectorCore', 'VectorCoreC'])
        modules_root = products.module_search_path(build)
        modules = [p for p in modules_root.rglob('*') if p.is_file() and p.name.startswith('VectorCore.')]
        require(bool(modules), 'missing iOS producer module')
        modulemap = products.module_map(build, 'VectorCoreC')
        capture['producers'][producer_mode] = {'kind': 'library', 'configuration': producer_mode,
            'objects': list(map(str, objects)),
            'modules': list(map(str, modules)), 'modulemap': str(modulemap), 'buildCommand': 'producer-build'}
        template = collector.source / 'Scripts/ci/fixtures/compiler-qualification-ios'
        for name, original_path, kind, omitted in PROBES:
            print(f'Collecting iOS {producer_mode}/{client_mode}: {name}', flush=True)
            directory = output / 'cases' / name
            host = directory / 'host'
            host.mkdir(parents=True, exist_ok=False)
            shutil.copytree(template / 'QualificationHost.xcodeproj', host / 'QualificationHost.xcodeproj',
                            ignore=shutil.ignore_patterns('xcuserdata', 'project.xcworkspace'))
            shutil.copy2(template / 'App.swift', host / 'App.swift')
            original = collector.source / original_path
            adapter = host / 'ProbeAdapter.swift'
            adapter.write_text(adapt_fixture(original.read_text(), kind))
            adapter_identity = verify_adapter(original.read_text(), adapter.read_text(), kind)
            witness_source = collector.source / 'Scripts/ci/fixtures/compile-contracts/BufferValueWitness.swift'
            shutil.copy2(witness_source, host / 'BufferValueWitness.swift')
            args = [developer / 'usr/bin/xcodebuild', '-project', host / 'QualificationHost.xcodeproj',
                    '-scheme', 'QualificationHost', '-configuration', client_mode.title(),
                    '-sdk', sdk, '-destination', destination, '-derivedDataPath', directory / 'derived',
                    'build', 'ARCHS=arm64', 'ONLY_ACTIVE_ARCH=YES',
                    'VC_PRODUCER_MODULES=' + str(modules_root),
                    'VC_MODULE_FLAGS=' + shlex.join(['-Xcc', '-fmodule-map-file=' + str(modulemap),
                         '-Xcc', '-I' + str(collector.source / 'Sources/VectorCoreC/include')]),
                    'VC_PRODUCER_OBJECT_FLAGS=' + shlex.join(list(map(str, objects))),
                    'VC_PROBE_CONDITIONS=' + ('OMIT_WRITES' if omitted else '')]
            if physical:
                args += ['DEVELOPMENT_TEAM=' + team, '-allowProvisioningUpdates']
            record, log = command(name + '-build', args, 600)
            built = xcode_commands(log)
            build_tools.update(argv[0] for argv in [*built['swift'].values(), built['link'], built['archive']])
            for argv in built['swift'].values():
                verify_xcode_swift_flags(argv, client_mode, identity)
            binary = Path(option(built['link'], '-o'))
            require(binary.is_file(), 'app binary missing')
            invocation = new_invocation()
            app = binary.parent
            bundle = 'org.vectorcore.qualification.Host'
            if physical:
                command(name + '-install', [*xcrun, 'devicectl', 'device', 'install', 'app',
                        '--device', settings['identifier'], app], 120)
                command(name + '-launch', [*xcrun, 'devicectl', 'device', 'process', 'launch',
                        '--device', settings['identifier'], '--terminate-existing', bundle,
                        '--invocation', invocation, '--case', name])
            else:
                command(name + '-install', [*xcrun, 'simctl', 'install', settings['identifier'], app])
                command(name + '-launch', [*xcrun, 'simctl', 'launch', '--terminate-running-process',
                        settings['identifier'], bundle, '--invocation', invocation, '--case', name])
            result = directory / 'app.records'
            if physical:
                # Each retrieval has its own receipt/path, even when the app is
                # still starting. Only a complete matching frame is retained.
                complete = False
                for attempt in range(30):
                    destination_file = directory / f'copy-{attempt}.records'
                    command(name + f'-retrieve-{attempt}', [*xcrun, 'devicectl', 'device', 'copy', 'from',
                            '--device', settings['identifier'], '--domain-type', 'appDataContainer',
                            '--domain-identifier', bundle, '--source', f'Documents/{invocation}.records',
                            '--destination', destination_file], 20, expected=None)
                    if destination_file.is_file() and 'VC_IOS_END ' in destination_file.read_text():
                        shutil.copy2(destination_file, result)
                        complete = True
                        break
                    time.sleep(0.5)
                require(complete, 'physical app result incomplete')
            else:
                record, _ = command(name + '-container', [*xcrun, 'simctl', 'get_app_container',
                                    settings['identifier'], bundle, 'data'])
                remote = Path(stdout(record).strip()) / 'Documents' / (invocation + '.records')
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if remote.is_file() and 'VC_IOS_END ' in remote.read_text():
                        break
                    time.sleep(0.1)
                require(remote.is_file(), 'app never wrote numerical records')
                command(name + '-retrieve', ['/bin/cp', remote, result])
            read_app_frame(result.read_text(), invocation, sha(binary), runtime, settings['platform'], name)
            binding = {'kind': 'xcode-app', 'invocationId': invocation, 'caseId': name, 'result': str(result),
                'adapter': str(adapter), 'original': str(original), 'adapterKind': kind,
                'adapterIdentity': adapter_identity, 'omittedWrites': omitted,
                'hostSource': str(host / 'App.swift'), 'witnessSource': str(host / 'BufferValueWitness.swift'),
                'witnessArchive': option(built['archive'], '-o'),
                'witnessModule': option(built['swift']['BufferValueWitness'], '-emit-module-path'),
                'destination': destination, 'installCommand': name + '-install'}
            collector.case(name, 'lost-store' if kind == 'standalone' else 'buffer-value',
                producer_mode, client_mode, name + '-build', name + '-launch', binary, app=binding)
            write_json(output / 'capture.json', capture)
        after = source_manifest(collector.source)
        write_json(output / 'source-inputs-after.json', after)
        require(before == after, 'compiled source inputs changed during collection')
        artifacts = {str(p) for p in output.rglob('*') if p.is_file()
                     and 'ModuleCache.noindex' not in p.parts and 'module-cache' not in p.parts}
        for receipt in capture['commands'].values():
            artifacts.add(json.loads(Path(receipt).read_text())['executable'])
        artifacts.update(str(p) for p in (collector.source / 'Scripts/ci').rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts)
        artifacts.update(str(collector.source / p) for _, p, _, _ in PROBES)
        artifacts.update((str(swiftc.with_name('swift-frontend')), str(sdk.resolve() / 'SDKSettings.json')))
        artifacts.update(build_tools)
        artifacts.discard(str(output / 'capture.json'))
        capture['artifactFiles'] = sorted(artifacts)
        capture['collectionComplete'] = True
    except BaseException as error:
        capture['collectionError'] = str(error)
        raise
    finally:
        write_json(output / 'capture.json', capture)
    return output / 'capture.json'


def merge_cells(cells, manifest_digests):
    """Combine all four completed cells without changing their original receipts."""
    import copy
    expected = {p + '-' + c for p in ('debug', 'release') for c in ('debug', 'release')}
    require(set(cells) == expected and set(manifest_digests) == expected, 'incomplete iOS matrix')
    require(len(set(manifest_digests.values())) == 1, 'matrix compiled source inputs differ')
    first = cells['debug-debug']
    require(all(c['collectionComplete'] is True and same(c['identity'], first['identity'])
                and c['sourceCommit'] == first['sourceCommit'] for c in cells.values()), 'matrix identities differ')
    require(len({c['invocationId'] for c in cells.values()}) == 4, 'reused matrix capture')
    merged = {'schemaVersion': 1, 'invocationId': new_invocation(), 'collectionComplete': True,
        'identity': copy.deepcopy(first['identity']), 'sourceCommit': first['sourceCommit'],
        'sourceManifest': first['sourceManifest'], 'commands': {}, 'producers': {}, 'cases': [],
        'checks': [], 'artifactFiles': sorted({path for c in cells.values() for path in c['artifactFiles']})}
    for key, capture in sorted(cells.items()):
        producer_mode, client_mode = key.split('-')
        require(set(capture['producers']) == {producer_mode} and not capture['checks'], 'unexpected iOS cell inventory')
        producer = copy.deepcopy(capture['producers'][producer_mode])
        producer['configuration'] = producer_mode
        producer['buildCommand'] = key + '/' + producer['buildCommand']
        merged['producers'][key] = producer
        merged['commands'].update({key + '/' + name: receipt for name, receipt in capture['commands'].items()})
        for row in capture['cases']:
            require(row['producer'] == producer_mode and row['client'] == client_mode, 'cell configurations differ')
            case = copy.deepcopy(row)
            case['id'] = key + '/' + row['id']
            case['producer'] = key
            for field in ('command', 'compileCommand'):
                case[field] = key + '/' + row[field]
            case['app']['installCommand'] = key + '/' + row['app']['installCommand']
            merged['cases'].append(case)
    return merged


def collect_ios_matrix(source_root, developer_dir, destination, output):
    from qualification_records import read_json, sha
    from v03_process import write_json
    require(destination_config(destination)['platform'] == 'ios-simulator', 'matrix requires a simulator')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cells, manifests, paths = {}, {}, {}
    for producer in ('debug', 'release'):
        for client in ('debug', 'release'):
            key = producer + '-' + client
            path = collect_ios(source_root, developer_dir, destination, output / key, producer, client)
            cells[key] = read_json(path)
            manifests[key] = sha(cells[key]['sourceManifest'])
            paths[key] = str(path)
    merged = merge_cells(cells, manifests)
    merged['cellCaptures'] = paths
    merged['artifactFiles'] = sorted(set(merged['artifactFiles']) | set(paths.values()))
    write_json(output / 'capture.json', merged)
    return output / 'capture.json'


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'developer-dir', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--producer-mode', choices=('debug', 'release'))
    parser.add_argument('--client-mode', choices=('debug', 'release'))
    parser.add_argument('--matrix', action='store_true')
    parser.add_argument('--team')
    parser.add_argument('--allow-device-install', action='store_true')
    args = parser.parse_args()
    if args.matrix:
        require(not (args.producer_mode or args.client_mode or args.team or args.allow_device_install),
                'matrix cannot override individual modes or device options')
        path = collect_ios_matrix(args.source_root, args.developer_dir, args.destination, args.output)
    else:
        path = collect_ios(args.source_root, args.developer_dir, args.destination, args.output,
                           args.producer_mode, args.client_mode, team=args.team,
                           allow_device_install=args.allow_device_install)
    print('Collection complete; qualification requires a reviewed profile: ' + str(path))


if __name__ == '__main__':
    main()
