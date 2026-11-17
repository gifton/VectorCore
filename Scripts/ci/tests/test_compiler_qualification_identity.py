"""Exact compiler pairs, legacy boundaries and newly reachable collection routes."""
import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_compiler_qualification as capture
import compiler_qualification as checker
import hosted_qualification as hosted
import qualification_ios as ios
from qualification_fixture import CaptureFixture, sha, write_json
from qualification_review_fixture import add_c_matrix
from ios_qualification_fixture import IOSCaptureFixture

LOCAL = ('448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f',
         '8635f454b0437014e3e03ccbaefb78c312d8d0337302f8a2348c15e59044762d')
HOSTED = ('020a521df9efc127b6817a2ed2f1560155108f0a0ead1ee1fa98147f5944eb4f',
          '51e35b74f76fe8df9d2a203a9edddf4b10fbd2e94e3036137babacc390d9901e')


class CompilerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('compiler_identity'), 'exact compiler-pair gate is missing')
        self.policy = importlib.import_module('compiler_identity')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_only_the_two_adopted_pairs_pass(self):
        for frontend in (LOCAL[0], HOSTED[0], 'a'*64):
            for driver in (LOCAL[1], HOSTED[1], 'b'*64):
                with self.subTest(frontend=frontend, driver=driver):
                    if (frontend,driver) in (LOCAL,HOSTED):
                        self.policy.validate_pair(frontend,driver)
                    else:
                        with self.assertRaises(ValueError): self.policy.validate_pair(frontend,driver)

    def test_legacy_frontend_cannot_admit_hosted_or_unlisted_compilers(self):
        self.policy.validate_legacy_frontend(LOCAL[0])
        for frontend in (HOSTED[0], 'a'*64, None, True):
            with self.subTest(frontend=frontend), self.assertRaises(ValueError):
                self.policy.validate_legacy_frontend(frontend)

    def toolchain(self, suffix=''):
        source = self.root/('source'+suffix)
        source.mkdir()
        for name in ('Sources','Tests','Benchmarks','Scripts/ci'):
            (source/name).mkdir(parents=True)
        (source/'Package.swift').write_text('// fixture\n')
        developer = self.root/('Xcode'+suffix)/'Contents/Developer'
        binary = developer/'Toolchains/XcodeDefault.xctoolchain/usr/bin'
        binary.mkdir(parents=True)
        for name in ('swift','swiftc','swift-frontend'):
            (binary/name).write_text('frontend'+suffix)
        (binary/'swift-driver').write_text('driver'+suffix)
        sdk = developer/'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'
        sdk.mkdir(parents=True)
        write_json(sdk/'SDKSettings.json', {})
        return source, binary/'swiftc', sdk

    def test_entry_point_substitution_rejects_even_with_an_allowed_pair(self):
        _, swiftc, _ = self.toolchain()
        good = self.policy.observe_tools(swiftc)
        pair = (good['frontendSHA256'],good['swiftDriverSHA256'])
        with patch.object(self.policy,'REVIEWED_PAIRS',frozenset({pair})):
            self.policy.validate_tools(good)
            for role in ('swift','swiftc'):
                path = swiftc.with_name(role)
                original = path.read_bytes()
                path.write_bytes(b'substituted entry point')
                with self.subTest(role=role), self.assertRaises(ValueError):
                    self.policy.validate_tools(self.policy.observe_tools(swiftc))
                path.write_bytes(original)

    def test_frontend_cannot_be_passed_as_the_swiftc_entry_point(self):
        _,swiftc,_ = self.toolchain()
        with self.assertRaisesRegex(ValueError,'swiftc'):
            self.policy.observe_tools(swiftc.with_name('swift-frontend'))

    def test_native_identity_and_nested_drivers_use_the_observed_pair(self):
        source, swiftc, sdk = self.toolchain()
        pair = (sha(swiftc.with_name('swift-frontend')),sha(swiftc.with_name('swift-driver')))
        with patch.object(self.policy,'REVIEWED_PAIRS',frozenset({pair})), patch.object(capture,'ROOT',source):
            c = capture.NativeCapture(source,swiftc,sdk,self.root/'evidence',4)
            c.preflight()
            outputs = ['Xcode 27.2\nBuild version 27B52\n',
                       'Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)\n',
                       '27.0.1','26A434','a'*40,'']
            with patch.object(c,'command',side_effect=[({},s) for s in outputs]), \
                 patch.object(capture.platform,'machine',return_value='arm64'):
                identity = c.identity()
            self.assertEqual((identity['frontendSHA256'],identity['swiftDriverSHA256']),pair)
            self.assertEqual(c.capture['schemaVersion'],2)
            for method,args in ((c.original_probes,()),(c.consumers,('debug',self.root/'products'))):
                with patch.object(c,'command',side_effect=RuntimeError('stop after dispatch')) as call:
                    with self.assertRaisesRegex(RuntimeError,'stop after dispatch'): method(*args)
                argv = call.call_args.args[1]
                self.assertEqual(argv[argv.index('--frontend-sha256')+1],pair[0])
            c.finish_inventory()
            self.assertIn(str(swiftc.with_name('swift-driver')),c.capture['artifactFiles'])
            swiftc.with_name('swift-driver').write_text('changed after preflight')
            with self.assertRaisesRegex(ValueError,'compiler'):
                c.finish_inventory()

    def test_ios_unknown_pair_stops_before_output_or_subprocess(self):
        source,swiftc,_ = self.toolchain()
        with patch.object(capture.NativeCapture,'command') as command:
            with self.assertRaisesRegex(ValueError,'pair'):
                ios.collect_ios(source,swiftc.parents[4],
                    'platform=iOS Simulator,id=00000000-0000-0000-0000-000000000001',
                    self.root/'ios-evidence','debug','release')
        command.assert_not_called()
        self.assertFalse((self.root/'ios-evidence').exists())

    def test_paired_capture_verifies_and_rejects_driver_substitution_and_downgrades(self):
        f = CaptureFixture(self.root)
        f.as_paired_capture()
        with f.compiler_policy():
            good = checker.check(f.profile_path,f.evidence,f.source)
            self.assertEqual(good['vectorCoreQualification'],'accepted',good['problems'])
            for changed in ('driver-hash','driver-path','missing-driver','capture-schema'):
                profile, captured = copy.deepcopy(f.profile), copy.deepcopy(f.capture)
                if changed=='driver-hash':
                    profile['identity']['swiftDriverSHA256']=captured['identity']['swiftDriverSHA256']='c'*64
                elif changed=='driver-path':
                    foreign = f.evidence/'foreign-driver'
                    foreign.write_bytes(Path(profile['identityFiles']['swiftDriverSHA256']).read_bytes())
                    profile['identityFiles']['swiftDriverSHA256']=str(foreign)
                    profile['pins'][str(foreign)]=sha(foreign)
                elif changed=='missing-driver':
                    del profile['identityFiles']['swiftDriverSHA256']
                else:
                    captured['schemaVersion']=1
                write_json(f.capture_path,captured)
                profile['pins'][str(f.capture_path)]=sha(f.capture_path)
                write_json(f.profile_path,profile)
                with self.subTest(changed=changed):
                    result=checker.check(f.profile_path,f.evidence,f.source)
                    self.assertEqual(result['vectorCoreQualification'],'rejected',result)

    def test_paired_native_and_ios_roles_reject_repinned_entry_point_bytes(self):
        for fixture in (CaptureFixture, IOSCaptureFixture):
            for role in ('swift', 'swiftc'):
                with tempfile.TemporaryDirectory() as directory, self.subTest(fixture=fixture.__name__, role=role):
                    f = fixture(directory)
                    if fixture is CaptureFixture:
                        # Exercise producer, client, separate witness and all C build/link cells.
                        f.add_command('producer', [str(f.swift), 'build', '-c', 'release'], '', 0, executable=f.swift)
                        f.capture['producers']['release']['buildCommand'] = 'producer'
                        f.profile['producers'] = copy.deepcopy(f.capture['producers'])
                        f.capture['commands']['producer'] = f.commands['producer']['receipt']
                        obj, module = f.evidence/'witness.o', f.evidence/'Witness.swiftmodule'
                        obj.write_bytes(b'witness object')
                        module.write_bytes(b'witness module')
                        f.add_command('witness', [str(f.compiler), '-O', '-swift-version', '6', '-target',
                            f.identity['target'], '-sdk', str(f.sdk.parent), '-emit-object', '-emit-module',
                            '-emit-module-path', str(module), '-o', str(obj)], '', 0, executable=f.compiler)
                        f.capture['commands']['witness'] = f.commands['witness']['receipt']
                        for row in f.case_specs + f.capture_cases:
                            row['witness'] = dict(command='witness', object=str(obj), module=str(module))
                        f.commands['compile']['argv'].append(str(obj))
                        f.alter_receipt('compile', argv=f.commands['compile']['argv'])
                        add_c_matrix(f)
                    # The iOS fixture includes its producer, app and separate witness.
                    f.as_paired_capture()
                    with f.compiler_policy():
                        baseline = checker.check(f.profile_path, f.evidence, f.source)
                        self.assertEqual(baseline['vectorCoreQualification'], 'accepted', baseline['problems'])
                        entry = Path(f.profile['identityFiles']['frontendSHA256']).with_name(role)
                        entry.write_bytes(b'unreviewed replacement entry point')
                        for name, expected in f.commands.items():
                            receipt = json.loads(Path(expected['receipt']).read_text())
                            if Path(receipt['executable']).resolve() == entry.resolve():
                                f.alter_receipt(name, executableSHA256=sha(entry))
                        f.save(repin=True)
                        with self.assertRaisesRegex(ValueError, 'entry-point'):
                            self.policy.validate_tools(self.policy.observe_tools(entry.with_name('swiftc')))
                        result = checker.check(f.profile_path, f.evidence, f.source)
                        self.assertEqual(result['vectorCoreQualification'], 'rejected', result['problems'])
                        self.assertTrue(any('entry-point' in p for p in result['problems']), result['problems'])
                        self.assertIs(result['releaseAdmitted'], False)

    def test_hosted_unknown_pair_retains_preflight_without_running_product_commands(self):
        args = SimpleNamespace(source_root=self.root/'candidate',policy_root=self.root/'policy',
            output=self.root/'out',candidate_commit='a'*40,policy_commit='b'*40,repository='owner/repo',
            run_id='123',platform='native',simulator_runtime='com.apple.CoreSimulator.SimRuntime.iOS-27-0')
        observed = dict(frontendSHA256='a'*64,swiftDriverSHA256='b'*64,swiftSHA256='a'*64,swiftcSHA256='a'*64)
        with patch.object(hosted,'checkout_identity'), patch.object(self.policy,'observe_tools',return_value=observed), \
             patch.dict(os.environ,DEVELOPER_DIR=str(hosted.DEVELOPER)), patch.object(hosted,'run') as run:
            with self.assertRaisesRegex(ValueError,'pair'): hosted.collect(args)
        run.assert_not_called()
        self.assertEqual(json.loads((args.output/'compiler-preflight.json').read_text())['observed'],observed)
        self.assertTrue((args.output/'collection.json').is_file())
        self.assertFalse((args.output/'evidence/capture.json').exists())
        upload=self.root/'upload'
        hosted.bundle(SimpleNamespace(output=args.output,upload=upload))
        review=dict(hosted.metadata(args),sha256=hosted.file_sha(upload/'evidence.tar.gz'))
        with self.assertRaisesRegex(ValueError,'missing collection/capture'):
            hosted.extract_reviewed_bundle(upload/'evidence.tar.gz',self.root/'extracted',review,hosted.metadata(args))

    def collect_hosted(self, pair, platform, label, mutation=None):
        args = SimpleNamespace(source_root=self.root/'candidate',policy_root=self.root/'policy',
            output=self.root/label,candidate_commit='a'*40,policy_commit='b'*40,repository='owner/repo',
            run_id='123',platform=platform,simulator_runtime='com.apple.CoreSimulator.SimRuntime.iOS-27-0')
        observed = dict(frontendSHA256=pair[0],swiftDriverSHA256=pair[1],swiftSHA256=pair[0],swiftcSHA256=pair[0])
        commands = []
        def run(argv,*unused,**kwargs):
            argv=list(map(str,argv)); commands.append(argv)
            if '-version' in argv:
                return {},'Xcode 27.2\nBuild version 27B52\n'
            if 'runtimes' in argv:
                return {},json.dumps({'runtimes':[{'identifier':args.simulator_runtime,'isAvailable':True}]})
            if 'create' in argv:
                return {},'00000000-0000-0000-0000-000000000001\n'
            if '--source-root' in argv:
                target=Path(argv[argv.index('--output')+1]); target.mkdir()
                record=dict(schemaVersion=2,sourceCommit=args.candidate_commit,collectionComplete=True,
                            identity={'frontendSHA256':pair[0],'swiftDriverSHA256':pair[1]})
                if mutation=='legacy': record['schemaVersion']=1
                if mutation=='pair': record['identity']['swiftDriverSHA256']='e'*64
                if mutation=='source': record['sourceCommit']='c'*40
                write_json(target/'capture.json',record)
            return {},''
        with patch.object(hosted,'checkout_identity'), patch.object(self.policy,'observe_tools',return_value=observed), \
             patch.dict(os.environ,DEVELOPER_DIR=str(hosted.DEVELOPER)), \
             patch.object(hosted.platform,'machine',return_value='arm64'), patch.object(hosted,'run',side_effect=run):
            hosted.collect(args)
        return args,commands

    def test_hosted_collection_forwards_actual_frontend_for_each_pair_and_platform(self):
        for index,pair in enumerate((LOCAL,HOSTED)):
            for platform in ('native','ios-simulator'):
                with self.subTest(pair=index,platform=platform):
                    args,commands=self.collect_hosted(pair,platform,f'{platform}-{index}')
                    collect=next(argv for argv in commands if '--source-root' in argv)
                    if platform=='native':
                        self.assertEqual(collect[collect.index('--frontend-sha256')+1],pair[0])
                    result=json.loads((args.output/'compiler-preflight.json').read_text())
                    self.assertIsNone(result['problem'])
                    self.assertEqual(result['observed']['swiftDriverSHA256'],pair[1])

    def test_hosted_collection_rejects_capture_identity_or_schema_drift(self):
        for mutation in ('legacy','pair','source'):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.collect_hosted(HOSTED,'native',mutation,mutation)


if __name__ == '__main__':
    unittest.main()
