"""Actual producer/admission controls using a fixture driver, never native builds."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'Scripts/ci'))
from v03_process import run, sha, test_environment
from source_binding import produce, capture_source, admit_association

SWIFT = '''#!/usr/bin/env python3
import sys
from pathlib import Path
root=Path(__file__).parent
if (root/'mode').read_text() == 'fail': raise SystemExit(3)
scratch=Path(sys.argv[sys.argv.index('--scratch-path')+1])
binpath=scratch/'arm64-apple-macosx/release'
bundle=binpath/'VectorCorePackageTests.xctest/Contents/MacOS/VectorCorePackageTests'
bundle.parent.mkdir(parents=True)
bundle.write_bytes((root/'Tests/T.swift').read_bytes())
if (root/'mode').read_text() == 'change': (root/'Tests/T.swift').write_text('M.S/test body=changed-during-build')
print('FIXTURE_BUILD_ONLY -O -enable-testing')
'''
DRIVER = '''import argparse,json,sys,shutil
from pathlib import Path
from v03_process import run,sha,test_environment,write_json
p=argparse.ArgumentParser()
for arg in ['sanitizer','scratch-root','output','configuration','swift-build-system','jobs']:p.add_argument('--'+arg,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2];out=Path(a.output);out.mkdir();scratch=Path(a.scratch_root)/a.sanitizer
swift=root/'fixture-swift';(out/'find-swift.stdout').write_text(str(swift)+'\\n')
env=test_environment();env.update(ASAN_OPTIONS='detect_leaks=0:halt_on_error=1',TSAN_OPTIONS='halt_on_error=1')
flags=['--configuration',a.configuration,'--jobs',a.jobs,'--scratch-path',str(scratch),'--sanitize',a.sanitizer,'--build-system',a.swift_build_system,'-Xswiftc','-enable-testing']
run([swift,'build','--build-tests',*flags,'--verbose'],out,'build',cwd=root,environment=env,timeout=15)
binpath=scratch/'arm64-apple-macosx/release';bundle=binpath/'VectorCorePackageTests.xctest/Contents/MacOS/VectorCorePackageTests';copy=out/'artifacts/bundle';copy.parent.mkdir();shutil.copy2(bundle,copy)
write_json(out/'artifacts.json',[{'built':str(bundle),'retained':str(copy),'sha256':sha(copy),'bytes':copy.stat().st_size}])
write_json(out/'result.json',{'sanitizer':a.sanitizer,'configuration':a.configuration,'buildSystem':a.swift_build_system,'binPath':str(binpath),'loadedBundles':{'ComprehensiveTests':{'path':str(bundle),'sha256':sha(bundle)}},'manifestSHA256':sha(root/'Scripts/ci/sanitizer_bodies.json'),'completed':[{'id':'M.S/test'}]})
'''


class SourceBindingControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        for directory in ['Sources', 'Tests', 'Benchmarks', 'Scripts/ci']:
            (self.root/directory).mkdir(parents=True, exist_ok=True)
        (self.root/'Package.swift').write_text('fixture tools 6.0')
        (self.root/'Sources/Core.swift').write_text('fixture library')
        (self.root/'Tests/T.swift').write_text('M.S/test body=v1')
        (self.root/'Benchmarks/B.swift').write_text('fixture benchmark')
        (self.root/'Scripts/ci/check_sanitizers.py').write_text(DRIVER)
        (self.root/'Scripts/ci/sanitizer_bodies.json').write_text(json.dumps({'sanitizers':{'address':[{'id':'M.S/test'}],'thread':[{'id':'M.S/test'}]}}))
        shutil.copy2(ROOT/'Scripts/ci/v03_process.py',self.root/'Scripts/ci/v03_process.py')
        (self.root/'fixture-swift').write_text(SWIFT)
        (self.root/'fixture-swift').chmod(0o755)
        (self.root/'mode').write_text('normal')
        run(['/usr/bin/git','init','-q',self.root],Path(self.temp.name)/'setup','git-init',cwd=self.root,environment=test_environment(),timeout=15)

    def build(self, name='first', sanitizer='address'):
        return produce(self.root,Path(self.temp.name)/(name+'-output'),Path(self.temp.name)/(name+'-scratch'),sanitizer,test_environment())

    def current(self, name='admit'):
        return capture_source(self.root,Path(self.temp.name)/name,'current',test_environment())

    def test_ignored_source_is_captured_and_generated_bytecode_is_excluded(self):
        (self.root/'.gitignore').write_text('Tests/Ignored.swift\n')
        (self.root/'Tests/Ignored.swift').write_text('compiled despite gitignore')
        (self.root/'Scripts/ci/__pycache__').mkdir()
        (self.root/'Scripts/ci/__pycache__/tool.pyc').write_bytes(b'cache only')
        identity = self.current()
        self.assertEqual(identity['inputs']['Tests/Ignored.swift'],sha(self.root/'Tests/Ignored.swift'))
        self.assertFalse(any('__pycache__' in p for p in identity['driverInputs']))

    def test_fresh_producer_binds_source_command_result_and_retained_bundle(self):
        for sanitizer in ['address', 'thread']:
            with self.subTest(sanitizer=sanitizer):
                association=self.build(sanitizer,sanitizer)
                admitted=admit_association(association,self.root,self.current('admit-'+sanitizer))
                self.assertEqual(admitted['identity']['inputs']['Tests/T.swift'],sha(self.root/'Tests/T.swift'))
                self.assertEqual(admitted['bundle'].read_text(),'M.S/test body=v1')
                self.assertEqual(admitted['data']['configuration'],'release')
                self.assertEqual(admitted['data']['buildSystem'],'native')
                self.assertEqual(admitted['data']['sanitizer'],sanitizer)

    def test_current_source_cannot_admit_stale_otherwise_valid_bundle(self):
        old=self.build()
        (self.root/'Tests/T.swift').write_text('M.S/test body=v2')
        with self.assertRaisesRegex(ValueError,'[Ss]ource'):
            admit_association(old,self.root,self.current())
        new=self.build('second')
        admitted=admit_association(new,self.root,self.current('admit-second'))
        self.assertEqual(admitted['bundle'].read_text(),'M.S/test body=v2')

    def test_unrelated_fresh_snapshot_cannot_replace_bound_prebuild_record(self):
        association=self.build()
        (self.root/'Tests/T.swift').write_text('M.S/test body=v2')
        current=self.current()
        binding=json.loads(association.read_text())
        # The otherwise valid old result/bundle remains untouched.
        binding['records']['sourceBefore']['path']=str(Path(self.temp.name)/'admit/current-source.json')
        association.write_text(json.dumps(binding))
        with self.assertRaises(ValueError):admit_association(association,self.root,current)

    def test_replacing_both_snapshots_cannot_rebind_old_build_command(self):
        association=self.build()
        (self.root/'Tests/T.swift').write_text('M.S/test body=v2')
        current=self.current()
        binding=json.loads(association.read_text())
        for key in ['sourceBefore','sourceAfter']:
            path=Path(binding['records'][key]['path'])
            path.write_text(json.dumps(current,indent=2,sort_keys=True)+'\n')
            binding['records'][key]['sha256']=sha(path)
        association.write_text(json.dumps(binding))
        # The original successful build/driver records and old bundle still match.
        with self.assertRaisesRegex(ValueError,'source.*build|Source.*build'):
            admit_association(association,self.root,current)

    def test_changed_source_during_driver_never_mints_association(self):
        (self.root/'mode').write_text('change')
        with self.assertRaisesRegex(ValueError,'[Ss]ource'):self.build()
        self.assertFalse((Path(self.temp.name)/'first-output/association.json').exists())
        self.assertTrue((Path(self.temp.name)/'first-output/after-source.json').exists())

    def test_failed_driver_never_mints_association(self):
        (self.root/'mode').write_text('fail')
        with self.assertRaises((ValueError,RuntimeError)):self.build()
        self.assertFalse((Path(self.temp.name)/'first-output/association.json').exists())

    def test_overlapping_build_and_evidence_roots_are_refused(self):
        outer=Path(self.temp.name)/'overlap'
        for output,scratch in [(outer,outer),(outer,outer/'scratch'),(outer/'output',outer)]:
            with self.subTest(output=output,scratch=scratch):
                with self.assertRaisesRegex(ValueError,'separate'):
                    produce(self.root,output,scratch,'address',test_environment())
                self.assertFalse(outer.exists())

    def test_existing_scratch_or_output_is_rejected_before_driver(self):
        scratch=Path(self.temp.name)/'first-scratch';scratch.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):self.build()
        self.assertFalse((Path(self.temp.name)/'first-output').exists())
        scratch.rmdir();output=Path(self.temp.name)/'first-output';output.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):self.build()
        self.assertFalse(scratch.exists())

    def test_runner_rejects_old_independent_identity_interface_before_execution(self):
        output=Path(self.temp.name)/'must-not-execute'
        record,log=run([sys.executable,'-B',HERE/'run_full_suites.py','--driver-result','old-result.json',
                        '--source-identity','current-source.json','--scope','fixture','--output',output],
                       Path(self.temp.name)/'cli','old-interface',cwd=self.root,environment=test_environment(),timeout=15,expected=None)
        self.assertEqual(record['exit'],2)
        self.assertIn('--association',log)
        self.assertFalse(output.exists())

    def test_changed_result_or_retained_artifact_cannot_admit(self):
        for part in ['result','retained']:
            with self.subTest(part=part):
                association=self.build(part)
                binding=json.loads(association.read_text())
                if part=='result':target=Path(binding['records']['result']['path'])
                else:
                    artifacts=json.loads(Path(binding['records']['artifacts']['path']).read_text())
                    target=Path(artifacts[0]['retained'])
                target.write_text('changed independently')
                with self.assertRaises(ValueError):admit_association(association,self.root,self.current('admit-'+part))

if __name__=='__main__':unittest.main()
