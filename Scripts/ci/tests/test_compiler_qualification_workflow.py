"""Two-stage hosted evidence contracts; no network or compiler execution."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Scripts/ci'))
import hosted_qualification as hosted
from qualification_fixture import CaptureFixture, write_json


class HostedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.expected = dict(runId='123', repository='owner/VectorCore',
            candidateCommit='a'*40, policyCommit='b'*40, platform='native', name='native-evidence')

    def archive(self, extra=None, metadata=None):
        archive = self.root / 'evidence.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            entries = [('collection.json', json.dumps(metadata or self.expected).encode()),
                       ('evidence/capture.json', b'{"collectionComplete":true}')]
            for name, data in entries:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            if extra:
                tar.addfile(*extra)
        return archive, dict(self.expected, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())

    def test_reviewed_archive_binds_run_candidate_platform_and_digest(self):
        archive, review = self.archive()
        evidence = hosted.extract_reviewed_bundle(archive, self.root/'out', review, self.expected)
        self.assertTrue((evidence/'capture.json').is_file())

    def test_each_provenance_field_and_archive_digest_must_match(self):
        archive, review = self.archive()
        for field in (*self.expected, 'sha256'):
            changed = dict(review, **{field:'wrong'})
            with self.subTest(field=field), self.assertRaises(ValueError):
                hosted.extract_reviewed_bundle(archive, self.root/field, changed, self.expected)
            self.assertFalse((self.root/field).exists())

    def test_archive_metadata_must_agree_with_review(self):
        archive, review = self.archive(metadata=dict(self.expected, candidateCommit='c'*40))
        with self.assertRaisesRegex(ValueError, 'collection metadata'):
            hosted.extract_reviewed_bundle(archive, self.root/'out', review, self.expected)

    def test_tar_traversal_links_special_files_and_duplicates_reject(self):
        for kind in ('traversal','absolute','symlink','hardlink','fifo','duplicate'):
            info = tarfile.TarInfo({'traversal':'../escape','absolute':'/escape',
                                   'duplicate':'evidence/capture.json'}.get(kind,'evidence/item'))
            if kind in ('symlink','hardlink'):
                info.type = tarfile.SYMTYPE if kind=='symlink' else tarfile.LNKTYPE
                info.linkname = '../../escape'
            elif kind=='fifo': info.type = tarfile.FIFOTYPE
            archive, review = self.archive(extra=(info, None))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                hosted.extract_reviewed_bundle(archive, self.root/kind, review, self.expected)
            self.assertFalse((self.root/'escape').exists())

    def test_evidence_helpers_are_never_executed(self):
        data = b'raise RuntimeError("never execute evidence helpers")'
        info = tarfile.TarInfo('evidence/helper.py')
        info.size = len(data)
        archive, review = self.archive(extra=(info, io.BytesIO(data)))
        hosted.extract_reviewed_bundle(archive, self.root/'out', review, self.expected)

    def test_fresh_destination_required(self):
        archive, review = self.archive()
        destination = self.root/'out'
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            hosted.extract_reviewed_bundle(archive, destination, review, self.expected)

    def test_only_explicit_scoped_acceptance_passes(self):
        result = dict(vectorCoreQualification='accepted',compilerProfile='known-failing',releaseAdmitted=False,problems=[])
        hosted.require_accepted(result)
        for key, values in {'vectorCoreQualification':['unassessed','rejected',None],
                            'compilerProfile':['unverified',None], 'releaseAdmitted':[True,None],
                            'problems':[['failure'],None]}.items():
            for value in values:
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    hosted.require_accepted(dict(result, **{key:value}))

    def test_candidate_requires_full_immutable_commit(self):
        for value in ('main','abc123','A'*40,'a'*39,'a'*41,'a'*40+'\n','$(anything)'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                hosted.full_commit(value)
        self.assertEqual(hosted.full_commit('a'*40), 'a'*40)

    def test_profile_must_be_tracked_unchanged_and_inside_policy_checkout(self):
        repo = self.root/'policy'
        profile = repo/'Scripts/ci/qualification_profiles/review.json'
        profile.parent.mkdir(parents=True)
        profile.write_text('{}\n')
        def git(*args):
            return subprocess.check_output(['git','-C',str(repo),*args], text=True, timeout=15).strip()
        git('init','-q')
        git('add','.')
        git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
        revision = git('rev-parse','HEAD')
        relative = str(profile.relative_to(repo))
        self.assertEqual(hosted.reviewed_profile(repo,relative,revision),profile.resolve())
        for path, commit in [('../review.json',revision),('/tmp/review.json',revision),
                              (relative,'a'*40),('Scripts/ci/qualification_profiles/missing.json',revision)]:
            with self.subTest(path=path,commit=commit), self.assertRaises(ValueError):
                hosted.reviewed_profile(repo,path,commit)
        profile.write_text('{"changed":true}')
        with self.assertRaises(ValueError): hosted.reviewed_profile(repo,relative,revision)
        profile.unlink()
        external = self.root/'external.json'
        external.write_text('{}\n')
        profile.symlink_to(external)
        with self.assertRaises(ValueError): hosted.reviewed_profile(repo,relative,revision)

    def test_verify_uses_the_reviewed_policy_revision_and_enforces_capture_and_numerical_results(self):
        for kind in ('accepted','wrong-candidate','supported-failure'):
            with self.subTest(kind=kind):
                base = self.root/kind
                base.mkdir()
                f = CaptureFixture(base)
                def commit(repo):
                    for args in [('init','-q'),('add','.'),('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')]:
                        subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,timeout=15)
                    return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=15).strip()
                candidate = commit(f.source)
                f.capture['sourceCommit'] = 'c'*40 if kind=='wrong-candidate' else candidate
                if kind=='supported-failure':
                    f.alter_stdout('supported','BUFFER_VALUE first observed=2\nBUFFER_VALUE_COMPLETE count=1 failures=1\n',code=1)
                    f.commands['supported']['exit'] = 1
                    f.case_specs[0]['observed'] = {'first':'2'}
                f.save(repin=True)
                meta = dict(self.expected,candidateCommit=candidate)
                write_json(base/'collection.json',meta)
                archive = base/'evidence.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    tar.add(f.evidence,arcname='evidence')
                    tar.add(base/'collection.json',arcname='collection.json')
                policy = base/'policy'
                profile = policy/'Scripts/ci/qualification_profiles/review.json'
                profile.parent.mkdir(parents=True)
                write_json(profile,dict(f.profile,reviewedArtifact=dict(meta,sha256=hosted.file_sha(archive))))
                policy_commit = commit(policy)
                args = SimpleNamespace(source_root=f.source,policy_root=policy,profile=str(profile.relative_to(policy)),
                    candidate_commit=candidate,policy_commit=policy_commit,run_id='123',repository=meta['repository'],
                    platform='native',archive=archive,output=base/'verified')
                if kind=='accepted':
                    hosted.verify(args)
                    report=json.loads((args.output/'verification.json').read_text())
                    self.assertEqual(report['hostedProvenance']['reviewedPolicyCommit'],policy_commit)
                    self.assertEqual(report['hostedProvenance']['policyCommit'],'b'*40)
                    self.assertIs(report['releaseAdmitted'],False)
                else:
                    with self.assertRaises(ValueError): hosted.verify(args)
                    if kind=='supported-failure':
                        report=json.loads((args.output/'verification.json').read_text())
                        self.assertEqual(report['vectorCoreQualification'],'rejected')

    def test_bundle_retains_raw_bytes_as_data_and_remains_unassessed(self):
        raw = self.root/'raw'
        raw.mkdir()
        (raw/'failure.stdout').write_text('wrong numeric value\n')
        (raw/'alias').symlink_to('failure.stdout')
        upload = self.root/'upload'
        hosted.bundle(SimpleNamespace(output=raw,upload=upload))
        metadata = json.loads((upload/'archive.json').read_text())
        self.assertEqual(metadata['vectorCoreQualification'],'unassessed')
        self.assertIs(metadata['releaseAdmitted'],False)
        with tarfile.open(upload/'evidence.tar.gz') as tar:
            self.assertTrue(all(m.isfile() for m in tar.getmembers()))
            self.assertEqual(tar.extractfile('alias').read(),b'wrong numeric value\n')


def contract(workflow):
    jobs = workflow['jobs']
    assert set(jobs) == {'native','ios-simulator','qualification-required'}
    assert workflow['permissions'] == {'contents':'read','actions':'read'}
    assert workflow['env']['DEVELOPER_DIR'] == '/Applications/Xcode_27.2.app/Contents/Developer'
    for platform in ('native','ios-simulator'):
        job = jobs[platform]
        assert job['runs-on'] == 'xcode-27' and 'if' not in job and 'continue-on-error' not in job
        steps = job['steps']
        assert all('continue-on-error' not in s for s in steps)
        checkouts = [s for s in steps if s.get('uses','').startswith('actions/checkout@')]
        assert [(s['with']['path'],s['with']['ref']) for s in checkouts] == [('policy','${{ github.sha }}'),('candidate','${{ inputs.candidate_commit }}')]
        assert all(s['with']['persist-credentials'] is False for s in checkouts)
        commands = {s['name']:s for s in steps if 'run' in s}
        assert commands['Validate dispatch mode']['run'].strip() == 'case "$QUALIFICATION_MODE" in\n  collect|verify) ;;\n  *) exit 1 ;;\nesac'
        collect = commands['Collect unqualified evidence']
        assert collect['if'] == "inputs.mode == 'collect'"
        assert 'hosted_qualification.py collect' in collect['run']
        assert '--platform '+platform in collect['run']
        verify = commands['Verify reviewed evidence']
        assert verify['if'] == "inputs.mode == 'verify'"
        assert 'hosted_qualification.py verify' in verify['run']
        assert '--profile "$REVIEWED_PROFILE"' in verify['run']
        assert '--run-id "$EVIDENCE_RUN_ID"' in verify['run']
        assert commands['Archive raw evidence']['if'] == "always() && inputs.mode == 'collect'"
        uploads = [s for s in steps if s.get('uses','').startswith('actions/upload-artifact@')]
        assert len(uploads)==2 and all(s['if'].startswith('always()') for s in uploads)
        assert commands['Collection is not qualification']['if'] == "always() && inputs.mode == 'collect'"
        assert commands['Collection is not qualification']['run'].strip().endswith('exit 1')
    aggregate = jobs['qualification-required']
    assert aggregate['if'] == 'always()' and aggregate['needs'] == ['native','ios-simulator']
    gate = aggregate['steps'][-1]
    assert 'if' not in gate and 'continue-on-error' not in gate
    assert gate['run'] == 'python3 policy/Scripts/ci/check_required.py native ios-simulator'
    assert gate['env']['NEEDS_JSON'] == '${{ toJSON(needs) }}'


class HostedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT/'.github/workflows/compiler-qualification.yml'
        cls.workflow = json.loads(subprocess.check_output(['ruby','-ryaml','-rjson','-e',
            'puts JSON.generate(YAML.safe_load(File.read(ARGV[0]), aliases: false))',str(path)],text=True,timeout=15))

    def test_two_stage_workflow_contract(self):
        contract(self.workflow)

    def test_bypass_mutations_are_rejected(self):
        for platform in ('native','ios-simulator'):
            for field, value in [('runs-on','macos-latest'),('continue-on-error',True),('if','false')]:
                mutated = copy.deepcopy(self.workflow)
                mutated['jobs'][platform][field] = value
                with self.subTest(platform=platform,field=field), self.assertRaises(AssertionError): contract(mutated)
            for name in ('Collect unqualified evidence','Verify reviewed evidence','Archive raw evidence','Collection is not qualification'):
                mutated = copy.deepcopy(self.workflow)
                step = next(s for s in mutated['jobs'][platform]['steps'] if s.get('name')==name)
                step['if'] = 'false'
                with self.subTest(platform=platform,step=name), self.assertRaises(AssertionError): contract(mutated)

    def test_aggregate_rejects_missing_failed_skipped_or_upload_only_jobs(self):
        success = {k:{'result':'success'} for k in ('native','ios-simulator')}
        payloads = [(success,0)]
        for job in success:
            for state in ('failure','skipped','cancelled',None):
                changed = copy.deepcopy(success)
                changed[job]['result'] = state
                payloads.append((changed,1))
            payloads.append(({k:v for k,v in success.items() if k!=job},1))
        payloads.append(({'artifact-upload':{'result':'success'}},1))
        for needs, code in payloads:
            result = subprocess.run([sys.executable,'-B',str(ROOT/'Scripts/ci/check_required.py'),'native','ios-simulator'],
                env=dict(os.environ,NEEDS_JSON=json.dumps(needs)),capture_output=True,timeout=15)
            self.assertEqual(result.returncode, code, result.stderr)
