#!/usr/bin/env python3
"""Verify one retained native evidence selection; never admit a compiler or release.

No subprocesses, builds or historical Python imports are performed. The adjacent
profile pins receipt identities, not mutable summary success flags. Paths recorded
under the three capture roots are remapped to the supplied roots; external paths
are limited to files explicitly named in pinned input/command records. New runs
or candidates require review of a new profile. Hashes establish integrity, not
independent authentication of the original capture or proof of compiler safety.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

FRONTEND = '448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f'
SDK_HASH = '82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6'
SOURCE_HASH = 'f2869bf68bb9286d33f2107566a2fb53d6913f95c76dced68d9affde448bd6eb'
TARGET = 'arm64-apple-macosx14.0'
MODES = ('debug', 'release')
PROBES = ('ByValueReadAfter', 'ByValueReturnInside', 'CopyInoutReturnInside',
          'LostStoreObserved', 'NoWriteNegative', 'NonLiteralReadAfter', 'OutOfLineRead', 'ReturnInsideInout')
COMPILER_FAILURES = {'CopyInoutReturnInside', 'LostStoreObserved', 'ReturnInsideInout'}
RESIDUAL_CASES = ('new.concrete.value', 'new.concrete.copied-inout', 'new.inside.copied-inout',
                  'new.aggregate.inout', 'new.generic.value', 'new.generic.copied-inout',
                  'custom.value-bridge', 'custom.inout-bridge', 'optimized.storage.stdlib-inout',
                  'optimized.storage.value', 'optimized.storage.copied-inout', 'optimized.float.value',
                  'optimized.float.copied-inout', 'fixed.value', 'fixed.copied-inout',
                  'dynamic.value', 'dynamic.copied-inout')
RESIDUAL_FAILURES = {'new.concrete.copied-inout', 'new.inside.copied-inout', 'new.aggregate.inout'}
REMAINING = ['User-approved support boundary and admission policy', 'Old CI compiler tuples and hosted exact-candidate acceptance',
             'Optional C kernels and architecture-dependent routes', 'Platform/device and minimum-OS runtime coverage',
             'Sanitizer and controlled performance qualification', 'Downstream release-reference resolution and release materials',
             'Arbitrary downstream callbacks, custom conformers and generic/inlinable compositions are not certified']


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
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def command(path, expected_exit, resolve=Path):
    record = read_json(path)
    require(record['status'] == 'completed' and record['timeout'] is False, f'incomplete command: {path}')
    require(type(record['exit']) is int and record['exit'] == expected_exit, f'command exit: {path}')
    require(sha(resolve(record['executable'])) == record['executableSHA256'], f'executable changed: {path}')
    text = ''
    for stream in ('stdout', 'stderr'):
        info = record[stream]
        file = resolve(info['path'])
        data = file.read_bytes()
        require(hashlib.sha256(data).hexdigest() == info['sha256'] and len(data) == info['bytes'],
                f'command {stream} changed: {path}')
        text += data.decode('utf-8')
    return record, text


def numerical(text, code, observations, failures):
    """Require exact inventory/values and a final complete record; test_numerical_inventory..."""
    found = {}
    lines = text.splitlines()
    require(bool(lines), 'empty numerical output')
    for line in lines[:-1]:
        match = re.fullmatch(r'BUFFER_VALUE ([a-z0-9.-]+) observed=(\S+)', line)
        require(match is not None, 'malformed numerical row')
        name, value = match.groups()
        require(name not in found, 'duplicate numerical case: ' + name)
        found[name] = value
    require(found == observations, 'numerical observations differ from exact oracle')
    require(lines[-1] == f'BUFFER_VALUE_COMPLETE count={len(observations)} failures={failures}',
            'numerical completion differs')
    require(type(code) is int and code == int(failures > 0), 'numerical process exit differs')
    return found


def native_suite(text, code, mode):
    require(mode in MODES, 'unsupported native mode')
    failed = mode == 'release'
    issues = [line for line in text.splitlines() if 'recorded an issue at' in line]
    require(len(issues) == int(failed), 'additional/missing native issues')
    if failed:
        require('literalArrayStoreIsObservedAfterOutOfLineClosure() recorded an issue at LostStoreCanaryTests.swift:47:9:' in issues[0],
                'unexpected native issue')
    require(type(code) is int and code == int(failed), 'native suite exit')
    totals = [(int(a), int(b or 0), int(c)) for a, b, c in re.findall(
        r'Executed (\d+) tests?, with (?:(\d+) tests? skipped and )?(\d+) failures?', text)]
    require(bool(totals) and max(totals) == (218, 28, 0) and all(c == 0 for _, _, c in totals), 'XCTest totals/failures')
    runs = re.findall(r'Test run with (\d+) tests? in \d+ suites? (passed|failed) after[^\n]*', text)
    main = [(int(n), result) for n, result in runs if int(n) > 1]
    require(main == [(1775, 'failed' if failed else 'passed')], 'Swift Testing totals/outcome')
    require(sum(' Test ' in line and line.endswith(' skipped.') for line in text.splitlines()) == 18, 'Swift Testing skips')
    require('Suite SIMDStorageMutationTests passed' in text, 'new API suite did not pass')


def compile_flags(argv, mode, sdk):
    require(mode in MODES, 'unsupported compilation mode')
    for flag, value in [('-swift-version', '6'), ('-target', TARGET), ('-sdk', sdk)]:
        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, 'unsupported ' + flag)
    optimizations = [a for a in argv if a.startswith('-O')]
    require(optimizations == ['-Onone' if mode == 'debug' else '-O'], 'unsupported optimization')
    allowed = {'-swift-version', '-target', '-sdk', '-module-cache-path', '-I', '-Xcc', '-parse-as-library',
               '-O', '-Onone', '-o', '-D', '-module-name', '-emit-module', '-emit-module-path', '-emit-object',
               '-emit-sil', '-emit-ir', '-typecheck', '-framework'}
    for arg in argv[1:]:
        require(not arg.startswith('-') or arg in allowed or arg.startswith(('-fmodule-map-file=', '-I/')),
                'unsupported compiler flag: ' + arg)


def link_inputs(argv, inputs, build, extra_objects=()):
    build = Path(build)
    objects = {p for p in inputs if Path(p).suffix == '.o' and Path(p).is_relative_to(build)}
    linked = [a for a in argv if a.endswith('.o')]
    require(bool(objects) and len(linked) == len(set(linked)) and set(linked) == objects | set(extra_objects),
            'producer object inventory differs from linked objects')
    module = str(build / 'Modules/VectorCore.swiftmodule')
    searches = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == '-I']
    expected_searches = [str(build / 'Modules')] + sorted({str(Path(p).parent) for p in extra_objects})
    require(module in inputs and searches == expected_searches, 'producer module search path differs')
    modulemap = str(build / 'VectorCoreC.build/module.modulemap')
    maps = [arg for arg in argv if arg.startswith('-fmodule-map-file=')]
    require(modulemap in inputs and maps == ['-fmodule-map-file=' + modulemap], 'producer modulemap differs')


def consumer_oracles(profile):
    """Read literal expected rows; never execute a helper from the source/evidence roots."""
    data = profile['consumerObservations']
    positive, omitted = data['positive'], data['omitted']
    require(type(positive) is dict and type(omitted) is dict and len(positive) == 24
            and positive.keys() == omitted.keys(), 'consumer oracle inventory')
    require(all(type(value) is str and re.fullmatch(r'[0-9a-f,]+', value)
                for table in (positive, omitted) for value in table.values()), 'consumer oracle value syntax')
    require(sum(positive[key] != omitted[key] for key in positive) == 19, 'omitted-write oracle differs')
    return positive, omitted


def inventory(rows, keys, expected):
    actual = [tuple(row[key] for key in keys) for row in rows]
    require(len(actual) == len(set(actual)) and set(actual) == expected, 'missing/duplicate/unknown evidence cells')


class Evidence:
    def __init__(self, migration, residual, source):
        self.roots = {'migration': Path(migration).resolve(), 'residual': Path(residual).resolve(), 'source': Path(source).resolve()}
        self.profile = read_json(Path(__file__).with_name('bounded_native_qualification_profile.json'))
        self.verified = {}

    def resolve(self, value):
        path = Path(value)
        require(path.is_absolute() and '..' not in path.parts, 'invalid recorded absolute path')
        for kind, captured in self.profile['captureRoots'].items():
            if path.is_relative_to(captured):
                return self.roots[kind] / path.relative_to(captured)
        return path

    def hash(self, path, expected):
        # The first expectation wins, including when another spelling reaches the
        # same file. Covered by test_hash_registration_uses_first_canonical_path_expectation.
        path = Path(path).resolve()
        key = str(path)
        require(key not in self.verified or self.verified[key] == expected,
                'conflicting digest expectation: ' + key)
        require(sha(path) == expected, 'hash mismatch: ' + key)
        self.verified.setdefault(key, expected)

    def recheck(self):
        for path, expected in self.verified.items():
            require(sha(path) == expected, 'input changed during verification: ' + path)

    def mapping(self, entries, base=None):
        require(type(entries) is dict and bool(entries), 'empty hash inventory')
        for name, digest in entries.items():
            if base is None:
                path = self.resolve(name)
            else:
                require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'invalid artifact path')
                path = base / name
            self.hash(path, digest)

    def cmd(self, path, code):
        path = Path(path).resolve()
        expected = self.verified.get(str(path))
        require(expected is not None, 'command receipt has no prior pin: ' + str(path))
        self.hash(path, expected)
        record, text = command(path, code, self.resolve)
        # Parsing a valid record does not authorize replacing an earlier manifest
        # expectation. Recheck the original receipt pin after consuming the record.
        self.hash(path, expected)
        self.hash(self.resolve(record['executable']), record['executableSHA256'])
        for stream in ('stdout', 'stderr'):
            info = record[stream]
            self.hash(self.resolve(info['path']), info['sha256'])
        return record, text

    def anchors(self):
        for kind, entries in self.profile['anchors'].items():
            self.mapping(entries, self.roots[kind])

    def sources(self):
        m, r, s = (self.roots[k] for k in ('migration', 'residual', 'source'))
        self.hash(m/'compiled-source-inputs.json', SOURCE_HASH)
        self.hash(r/'compiled-source-inputs.json', SOURCE_HASH)
        frozen = read_json(m/'compiled-source-inputs.json')
        require(len(frozen) == 389, 'compiled source count')
        self.mapping(frozen, s)
        # Restrict enumeration to actual compiled-input roots, never evidence archives.
        actual = {'Package.swift'}
        for folder in ('Sources', 'Tests', 'Benchmarks'):
            actual.update(str(p.relative_to(s)) for p in (s/folder).rglob('*') if p.is_file())
        require(actual == set(frozen), 'compiled source inventory differs')
        identity = read_json(m/'xcode-27.2/identity.json')
        require(identity['frontendSHA256'] == FRONTEND and identity['sdkSettingsSHA256'] == SDK_HASH
                and identity['sourceInputs'] == frozen, 'unsupported producer identity')

    def compiler(self):
        folder = self.roots['migration']/'xcode-27.2/compiler-controls'
        summary = read_json(folder/'summary.json')
        self.mapping(summary['inputs']); self.mapping(summary['artifacts'], folder)
        require(summary['target'] == TARGET, 'compiler target differs')
        sdk = summary['sdk']
        require(summary['inputs'][sdk+'/SDKSettings.json'] == SDK_HASH, 'compiler SDK differs')
        frontend = [p for p in summary['inputs'] if Path(p).name == 'swift-frontend']
        require(len(frontend) == 1 and summary['inputs'][frontend[0]] == FRONTEND, 'compiler frontend differs')
        self.frontend, self.sdk = frontend[0], sdk
        self.swiftc = str(Path(self.frontend).with_name('swiftc'))
        _, host = self.cmd(folder/'host.json', 0)
        self.host(host)
        inventory(summary['cells'], ('probe','configuration'), {(p+'.swift',m) for p in PROBES for m in MODES})
        for cell in summary['cells']:
            probe, mode = Path(cell['probe']).stem, cell['configuration']
            stem = probe+'-'+mode
            compiled, _ = self.cmd(folder/(stem+'-compile.json'), 0)
            compile_flags(compiled['argv'], mode, sdk)
            require(compiled['argv'][0] == self.swiftc and compiled['executableSHA256'] == FRONTEND, 'compiler command identity')
            executed, text = self.cmd(folder/(stem+'-run.json'), 0)
            binary = folder/stem
            require(self.resolve(executed['argv'][0]) == binary and len(executed['argv']) == 1, 'compiler executable binding')
            self.hash(binary, cell['binarySHA256'])
            stale = probe == 'NoWriteNegative' or (mode == 'release' and probe in COMPILER_FAILURES)
            value, bits = ('2.0','40000000') if stale else ('1.0','3f800000')
            require(text == f'LOST_STORE observed={value} bits={bits} expected=1.0\n', 'compiler numerical profile differs: '+stem)
        return {'positiveCells':14,'correctPositiveCells':11,'knownFailures':sorted(COMPILER_FAILURES),'negativeControls':2}

    @staticmethod
    def host(text):
        values = dict(re.findall(r'^(ProductName|ProductVersion|BuildVersion):\s*(\S+)\s*$', text, re.M))
        require(values == {'ProductName':'macOS','ProductVersion':'27.0.1','BuildVersion':'26A434'}, 'unsupported host')

    def compile(self, folder, name, mode, code=0):
        record, text = self.cmd(folder/(name+'.json'), code)
        require(record['argv'][0] == self.swiftc and record['executableSHA256'] == FRONTEND, 'client compiler identity')
        compile_flags(record['argv'], mode, self.sdk)
        return record, text

    def run_rows(self, folder, stem, expected, failures):
        record, text = self.cmd(folder/(stem+'.json'), int(failures > 0))
        require(len(record['argv']) == 1 and self.resolve(record['argv'][0]) == folder / 'consumer',
                'consumer run binding')
        return numerical(text, record['exit'], expected, failures)

    def consumers(self, mode, oracles):
        target = self.roots['migration']/'xcode-27.2'
        folder = target/('consumers-'+mode)
        summary = read_json(folder/'summary.json')
        self.mapping(summary['artifacts'], folder)
        inputs = read_json(folder/'inputs.json'); self.mapping(inputs)
        require(summary['libraryConfiguration'] == mode, 'producer configuration differs')
        build = summary['binPath']
        require(self.resolve(build) == target/'build/arm64-apple-macosx'/mode, 'producer bin path differs')
        binding = read_json(target/(mode+'-consumer-binding.json'))
        for key, name in [('summary',folder/'summary.json'), ('buildReceipt',target/(mode+'-full-suite.json'))]:
            require(self.resolve(binding[key]) == name, 'producer receipt path differs')
            self.hash(name, binding[key+'SHA256'])
        require(binding['toolchain'] == 'xcode-27.2' and binding['libraryMode'] == mode, 'producer binding tuple')
        record, text = self.cmd(target/(mode+'-full-suite.json'), int(mode=='release'))
        argv = record['argv']
        expected = [str(Path(self.frontend).with_name('swift')), 'test','--build-system','native','--scratch-path',
                    str(Path(self.profile['captureRoots']['migration'])/'xcode-27.2/build'), '--sdk',self.sdk,
                    '--triple',TARGET,'-c',mode,'-j','4']
        require(argv == expected and record['executableSHA256'] == FRONTEND, 'native producer command differs')
        env=record['environment']
        require(env['SWIFT_EXEC']==self.swiftc and env['SDKROOT']==self.sdk and
                env['VECTORCORE_STRICT_PERF']=='0' and env['VECTORCORE_TEST_EXTENDED']=='0', 'native build environment differs')
        native_suite(text,record['exit'],mode)
        inventory(summary['cases'],('clientMode','negativeControl'),{(m,n) for m in MODES for n in (False,True)})
        for client in MODES:
            for omitted in (False,True):
                cell=folder/(client+('-omitted-write' if omitted else '-by-value'))
                witness,_=self.compile(cell,'witness-compile',client)
                require(('OMIT_WRITES' in witness['argv']) == omitted, 'witness omitted-write flag differs')
                compiled,_=self.compile(cell,'consumer-compile',client)
                link_inputs(compiled['argv'],inputs,build,[str(Path(witness['argv'][witness['argv'].index('-o')+1]))])
                observed=oracles[int(omitted)]
                self.run_rows(cell,'consumer-run',observed,19 if omitted else 0)
                if not omitted:
                    self.compile(cell,'witness-sil',client)
                    sil=(cell/'witness.sil').read_text()
                    require(any('witness_method' in line and '#SIMDStorage.withUnsafeMutableElements' in line for line in sil.splitlines()),
                            'new generic witness absent')
            cell=folder/(client+'-legacy-exposure')
            compiled,_=self.compile(cell,'compile',client)
            link_inputs(compiled['argv'],inputs,build)
            self.run_rows(cell,'run',{'concrete-inout':'40000000' if client=='release' else '3f800000',
                                     'concrete-value':'3f800000'},int(client=='release'))
        contracts={'SIMDStorage-new-conformer':None,'SIMDStorage-legacy-call':(6,"value of type 'S' has no member 'withUnsafeMutableBufferPointer'"),
                   'SIMDStorage-legacy-conformer':(5,"type 'LegacyOnlyBuffer' does not conform to protocol 'SIMDStorage'")}
        inventory(summary['migrationContracts'],('clientMode','id'),{(m,k) for m in MODES for k in contracts})
        for client in MODES:
            for name,diagnostic in contracts.items():
                _,text=self.compile(folder,client+'-'+name,client,int(diagnostic is not None))
                errors=re.findall(r'^.*?\.swift:(\d+):\d+: error: (.*)$',text,re.M)
                require(errors == ([] if diagnostic is None else [(str(diagnostic[0]),diagnostic[1])]), 'migration diagnostics differ')
        return {'producer':mode,'positiveObservations':48,'omittedWriteControls':2,'compileContracts':6,
                'legacyOptimizedInout':'known-failing','swiftTestingReported':1775,'swiftTestingDisabled':18,
                'xctestReported':218,'xctestSkipped':28}

    def residual(self):
        root=self.roots['residual']; target=root/'xcode-27.2'
        summary=read_json(root/'summary.json')
        selected=[row for row in summary['toolchains'] if row['toolchain']=='xcode-27.2']
        require(len(selected)==1,'residual toolchain inventory')
        row=selected[0]
        require(row['frontendSHA256']==FRONTEND and row['sdk']==self.sdk and row['target']==TARGET
                and row['producerConfiguration']=='release','residual tuple differs')
        self.mapping(row['artifacts'],target)
        inputs=read_json(target/'inputs.json');self.mapping(inputs)
        self.mapping(read_json(root/'probe-inputs.json'))
        binding=read_json(target/'producer-binding.json')
        migration=self.roots['migration']/'xcode-27.2'
        for key, path in [('priorBinding',migration/'release-consumer-binding.json'),('producerReceipt',migration/'release-full-suite.json')]:
            require(self.resolve(binding[key])==path,'residual producer binding differs')
            self.hash(path,binding[key+'SHA256'])
        require((binding['frontendSHA256'],binding['sdkSettingsSHA256'],binding['compiledSourceManifestSHA256'])
                ==(FRONTEND,SDK_HASH,SOURCE_HASH),'residual producer identities differ')
        self.host(self.cmd(root/'host-os.json',0)[1])
        require(self.cmd(root/'host-architecture.json',0)[1]=='arm64\n','unsupported architecture')
        inventory(row['cases'],('clientMode','omittedWrite'),{(m,n) for m in MODES for n in (False,True)})
        build=read_json(migration/'consumers-release/summary.json')['binPath']
        for mode in MODES:
            for omitted in (False,True):
                cell=target/(mode+('-omitted-write' if omitted else '-positive'))
                compiled,_=self.compile(cell,'compile',mode)
                require(('OMIT_WRITES' in compiled['argv'])==omitted,'residual omitted-write flag differs')
                link_inputs(compiled['argv'],inputs,build)
                wrong=set(RESIDUAL_CASES) if omitted else RESIDUAL_FAILURES if mode=='release' else set()
                expected={name:'40000000' if name in wrong else '3f800000' for name in RESIDUAL_CASES}
                self.run_rows(cell,'run',expected,len(wrong))
        return {'positiveObservations':34,'correctPositiveObservations':31,'knownFailures':sorted(RESIDUAL_FAILURES),
                'omittedWriteControls':2,'producerConfiguration':'release'}


def check(migration_root,residual_root,source_root):
    report={'schemaVersion':1,'draft':True,'compilerProfile':'unverified','nativeEvidence':'rejected',
            'supportPolicyAdopted':False,'releaseAdmitted':False,'remainingGates':REMAINING,'problems':[]}
    try:
        evidence=Evidence(migration_root,residual_root,source_root)
        evidence.anchors();evidence.sources()
        report['compilerObservations']=evidence.compiler()
        report['compilerProfile']='known-failing'
        oracles = consumer_oracles(evidence.profile)
        report['migration']=[evidence.consumers(mode,oracles) for mode in MODES]
        report['residual']=evidence.residual()
        # Rehash every examined integrity-bound file after analysis to catch intervening edits.
        evidence.recheck()
        report['nativeEvidence']='accepted'
        report['verifiedFiles']=len(evidence.verified)
        report['compiledSourceManifestSHA256']=SOURCE_HASH
        report['scope']='Retained native macOS 27.0.1 (26A434) arm64 evidence for the pinned migrated candidate only; no arbitrary downstream composition guarantee.'
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError) as error:
        report['problems'].append(str(error))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('migration-root','residual-root','source-root','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    # Reserve before doing any reads: existing output, including a symlink, is never overwritten.
    try:
        with args.output.open('x') as out:
            result=check(args.migration_root,args.residual_root,args.source_root)
            json.dump(result,out,indent=2,sort_keys=True);out.write('\n')
    except OSError as error:
        parser.exit(2,str(error)+'\n')
    print('nativeEvidence='+result['nativeEvidence']+'; compilerProfile='+result['compilerProfile']+'; releaseAdmitted=false')
    return 0 if result['nativeEvidence']=='accepted' else 1


if __name__=='__main__':raise SystemExit(main())
