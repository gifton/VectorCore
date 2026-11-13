"""Independent-review regressions: logical source drift and mixed Swift tools."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import compiler_qualification as checker
import qualification_records as records
from qualification_fixture import CaptureFixture, sha, write_json
from qualification_review_fixture import add_c_matrix


class QualificationReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f=CaptureFixture(self.tmp.name)

    def check(self):
        return checker.check(self.f.profile_path,self.f.evidence,self.f.source)

    def accepted(self):
        result=self.check()
        self.assertEqual(result['vectorCoreQualification'],'accepted',result)

    def rejected(self):
        result=self.check()
        self.assertEqual(result['vectorCoreQualification'],'rejected',result)
        self.assertIs(result['releaseAdmitted'],False)
        return result

    def mutate_during_read(self, change):
        original=records.read_json
        changed=[]
        receipt=Path(self.f.commands['compile']['receipt']).resolve()
        def reading(path):
            value=original(path)
            if Path(path).resolve()==receipt and not changed:
                change()
                changed.append(True)
            return value
        with patch.object(records,'read_json',side_effect=reading):
            self.rejected()
        self.assertTrue(changed)

    def test_added_compiled_files_during_verification_reject_in_every_source_root(self):
        for folder in ('Sources','Tests','Benchmarks'):
            with self.subTest(folder=folder):
                self.accepted()
                added=self.f.source/folder/'Added.swift'
                try: self.mutate_during_read(lambda:added.write_text('let added = 1\n'))
                finally: added.unlink(missing_ok=True)

    def test_source_symlink_retargeting_rejects_even_when_bytes_match(self):
        f=self.f
        alternate=f.source/'Sources/Alternate.swift'
        alias=f.source/'Sources/Alias.swift'
        for same_bytes in (False,True):
            with self.subTest(same_bytes=same_bytes):
                alternate.write_bytes((f.source/'Sources/Library.swift').read_bytes() if same_bytes else b'let alternate = 99\n')
                alias.unlink(missing_ok=True)
                alias.symlink_to('Library.swift')
                write_json(f.manifest,{str(p.relative_to(f.source)):sha(p) for p in f.source.rglob('*') if p.is_file()})
                f.profile['sourceManifestSHA256']=sha(f.manifest)
                f.save(repin=True)
                self.accepted()
                def retarget():
                    alias.unlink()
                    alias.symlink_to('Alternate.swift')
                self.mutate_during_read(retarget)

    def other_tool(self,name):
        tool=self.f.evidence/'other-toolchain'/name
        tool.parent.mkdir(exist_ok=True)
        tool.write_text('other Swift tool\n')
        (tool.parent/'swift-frontend').write_text('other frontend\n')
        return tool

    def replace_command_tool(self,command,tool):
        f=self.f
        argv=list(f.commands[command]['argv'])
        argv[0]=str(tool)
        f.commands[command]['argv']=argv
        f.alter_receipt(command,argv=argv,executable=str(tool),executableSHA256=sha(tool))
        f.save(repin=True)

    def test_native_consumer_cannot_use_another_pinned_toolchain(self):
        self.accepted()
        self.replace_command_tool('compile',self.other_tool('swiftc'))
        self.rejected()

    def test_ios_app_and_witness_cannot_substitute_another_tool_in_the_same_directory(self):
        from ios_qualification_fixture import IOSCaptureFixture
        for target in ('QualificationHost','BufferValueWitness'):
            with tempfile.TemporaryDirectory() as name, self.subTest(target=target):
                f=IOSCaptureFixture(name)
                check=lambda:checker.check(f.profile_path,f.evidence,f.source)
                self.assertEqual(check()['vectorCoreQualification'],'accepted')
                f.swift_commands[target][0]=f.tool['clang']
                f.alter_stdout('app-build',f.build_log())
                f.save(repin=True)
                result=check()
                self.assertEqual(result['vectorCoreQualification'],'rejected',result)

    def test_native_witness_cannot_use_another_pinned_toolchain(self):
        f=self.f
        obj,mod=f.evidence/'witness.o',f.evidence/'Witness.swiftmodule'
        obj.write_text('witness object')
        mod.write_text('witness module')
        f.add_command('witness',[str(f.compiler),'-O','-swift-version','6','-target',f.identity['target'],
            '-sdk',str(f.sdk.parent),'-emit-object','-emit-module','-emit-module-path',str(mod),'-o',str(obj)],'',0,executable=f.compiler)
        for row in f.case_specs+f.capture_cases:
            row['witness']=dict(command='witness',object=str(obj),module=str(mod))
        f.commands['compile']['argv'].append(str(obj))
        f.alter_receipt('compile',argv=f.commands['compile']['argv'])
        f.capture['commands']['witness']=f.commands['witness']['receipt']
        f.save(repin=True)
        self.accepted()
        self.replace_command_tool('witness',self.other_tool('swiftc'))
        self.rejected()

    def test_native_producer_and_swift_exec_override_bind_to_declared_frontend(self):
        f=self.f
        f.add_command('producer',[str(f.swift),'test','--sdk',str(f.sdk.parent),'--triple',f.identity['target'],'-c','release'],
            '',0,executable=f.swift)
        f.capture['producers']['release']['buildCommand']='producer'
        f.profile['producers']=copy.deepcopy(f.capture['producers'])
        f.capture['commands']['producer']=f.commands['producer']['receipt']
        f.save(repin=True)
        self.accepted()
        original=Path(f.commands['producer']['receipt']).read_text()
        expected=copy.deepcopy(f.commands['producer'])
        for kind in ('driver','SWIFT_EXEC'):
            write_json(f.commands['producer']['receipt'],json.loads(original))
            f.commands['producer']=copy.deepcopy(expected)
            f.profile['commands']=f.commands
            if kind=='driver': self.replace_command_tool('producer',self.other_tool('swift'))
            else:
                environment={'SWIFT_EXEC':str(self.other_tool('swiftc'))}
                f.commands['producer']['environment']=environment
                f.alter_receipt('producer',environment=environment)
                f.save(repin=True)
            with self.subTest(kind=kind): self.rejected()

    def test_every_c_build_and_link_role_rejects_a_mixed_toolchain(self):
        f=self.f
        root=add_c_matrix(f)
        self.accepted()
        for mode in ('debug','release'):
            for feature in ('off','on'):
                for command,tool_name in (('build','swift'),('link-public-routes','swiftc')):
                    path=root/(mode+'-c-'+feature)/(command+'.json')
                    original=path.read_bytes()
                    record=json.loads(original)
                    record['argv'][0]=str(self.other_tool(tool_name))
                    write_json(path,record)
                    f.save(repin=True)
                    try:
                        with self.subTest(mode=mode,feature=feature,command=command): self.rejected()
                    finally:
                        path.write_bytes(original)
                        f.save(repin=True)
