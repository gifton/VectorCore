"""Content references preserve exact evidence; no historical code is executed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import archived_evidence
except ModuleNotFoundError:
    archived_evidence = None


class ArchivedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(archived_evidence, 'archive reference loader is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.path = self.repo / 'profiles/review.json'
        self.path.parent.mkdir()
        self.raw = b'{"observations":{"value":"3f800000"},"releaseAdmitted":false}\n'
        self.path.write_bytes(self.raw)
        self.commit()
        self.reference = dict(artifactReference=1, gitCommit=self.git('rev-parse', 'HEAD').strip(),
                              gitPath='profiles/review.json', bytes=len(self.raw),
                              sha256=hashlib.sha256(self.raw).hexdigest())
        self.save()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True,
                                       stderr=subprocess.PIPE, timeout=20)

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit', '-qm', 'fixture')

    def save(self, data=None):
        self.path.write_text(json.dumps(self.reference if data is None else data) + '\n')

    def test_reference_reads_original_bytes_and_does_not_use_current_path_contents(self):
        self.assertEqual(archived_evidence.read_bytes(self.path), self.raw)
        self.assertEqual(archived_evidence.read_json(self.path),
                         {'observations': {'value': '3f800000'}, 'releaseAdmitted': False})

    def test_inline_private_profiles_still_work_without_a_git_checkout(self):
        path = Path(self.tmp.name) / 'private.json'
        path.write_bytes(self.raw)
        self.assertEqual(archived_evidence.read_bytes(path), self.raw)

    def test_corrupt_digest_or_byte_count_rejects_before_data_is_used(self):
        for field, value in [('sha256', '0' * 64), ('bytes', len(self.raw) + 1)]:
            with self.subTest(field=field):
                self.save(dict(self.reference, **{field: value}))
                with self.assertRaisesRegex(ValueError, 'digest|size'):
                    archived_evidence.read_json(self.path)

    def test_mutable_revisions_ambiguous_paths_and_malformed_references_reject(self):
        mutations = [('artifactReference', True), ('artifactReference', 2),
                     ('gitCommit', 'main'), ('gitCommit', 'a' * 39), ('gitCommit', 'A' * 40),
                     ('gitCommit', self.reference['gitCommit'] + '\n'),
                     ('gitPath', '/outside.json'), ('gitPath', '../outside.json'),
                     ('gitPath', './profiles/review.json'), ('gitPath', 'profiles//review.json'),
                     ('gitPath', 'profiles\\review.json'), ('gitPath', 'profiles/review.py'),
                     ('bytes', True), ('bytes', -1), ('sha256', 'abc'), ('sha256', 'A' * 64)]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                self.save(dict(self.reference, **{key: value}))
                with self.assertRaises(ValueError):
                    archived_evidence.read_json(self.path)
        for key in self.reference:
            bad = dict(self.reference)
            del bad[key]
            # Without the marker it is ordinary JSON, rejected by the consuming verifier.
            if key == 'artifactReference':
                continue
            self.save(bad)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                archived_evidence.read_json(self.path)
        self.save(dict(self.reference, fallback={'releaseAdmitted': True}))
        with self.assertRaises(ValueError):
            archived_evidence.read_json(self.path)

    def test_missing_object_does_not_fall_back_to_unverified_current_data(self):
        self.save(dict(self.reference, gitCommit='0' * 40))
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            archived_evidence.read_json(self.path)

    def archive_payload(self, raw):
        self.path.write_bytes(raw)
        self.commit()
        self.reference.update(gitCommit=self.git('rev-parse', 'HEAD').strip(),
                              bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.save()

    def test_duplicate_json_keys_reject_in_reference_and_archived_payload(self):
        self.path.write_text(json.dumps(self.reference)[:-1] + ',"artifactReference":1}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            archived_evidence.read_json(self.path)
        self.archive_payload(b'{"releaseAdmitted":false,"releaseAdmitted":true}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            archived_evidence.read_json(self.path)

    def test_nested_references_reject_instead_of_chasing_a_second_trust_anchor(self):
        self.archive_payload(json.dumps(self.reference).encode())
        with self.assertRaisesRegex(ValueError, 'nested'):
            archived_evidence.read_json(self.path)

    def test_shallow_clone_is_offline_by_default_and_fetches_only_on_request(self):
        self.commit()
        clone = Path(self.tmp.name) / 'shallow'
        subprocess.run(['git', 'clone', '-q', '--depth=1', self.repo.as_uri(), str(clone)],
                       check=True, capture_output=True, timeout=20)
        path = clone / 'profiles/review.json'
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            archived_evidence.read_json(path)
        self.assertEqual(archived_evidence.read_bytes(path, fetch=True), self.raw)
        self.assertEqual(subprocess.check_output(['git', '-C', str(clone), 'status', '--porcelain'], timeout=20), b'')
        self.assertEqual(archived_evidence.read_bytes(path), self.raw)

    def test_partial_clone_cannot_lazy_fetch_without_explicit_permission(self):
        self.commit()
        self.git('config', 'uploadpack.allowFilter', 'true')
        self.git('config', 'uploadpack.allowAnySHA1InWant', 'true')
        clone = Path(self.tmp.name) / 'partial'
        subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout',
                        self.repo.as_uri(), str(clone)], check=True, capture_output=True, timeout=20)
        subprocess.run(['git', '-C', str(clone), 'checkout', '-q', '--detach', 'HEAD'],
                       check=True, capture_output=True, timeout=20)
        blob = self.git('rev-parse', self.reference['gitCommit'] + ':' + self.reference['gitPath']).strip()
        probe = ['git', '--no-lazy-fetch', '-C', str(clone), 'cat-file', '-e', blob]
        self.assertNotEqual(subprocess.run(probe, capture_output=True, timeout=20).returncode, 0)
        trace = Path(self.tmp.name) / 'git-trace.txt'
        with patch.dict(os.environ, GIT_TRACE=str(trace)):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                archived_evidence.read_bytes(clone / 'profiles/review.json')
        self.assertNotIn(' fetch ', trace.read_text())
        self.assertNotEqual(subprocess.run(probe, capture_output=True, timeout=20).returncode, 0)
        self.assertEqual(archived_evidence.read_bytes(clone / 'profiles/review.json', fetch=True), self.raw)

    def test_materialize_cli_checks_content_and_refuses_to_overwrite(self):
        output = Path(self.tmp.name) / 'retained.json'
        cmd = [sys.executable, '-B', str(Path(archived_evidence.__file__)), str(self.path),
               '--output', str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_bytes(), self.raw)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), self.raw)


if __name__ == '__main__':
    unittest.main()
