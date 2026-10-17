#!/usr/bin/env python3
"""Bounded ordinary Swift 6 persistence consumer with immutable linked artifacts."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'Scripts/ci'))
import swiftpm_products as products
# argv[1] is the SwiftPM bin path (swift build -c release --show-bin-path); native and swiftbuild layouts both work.
build=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snap=products.snapshot_consumer_inputs(build,out)
objects=snap['objects']
artifacts=[dst for _,dst in snap['records']]+[snap['modulemap']]
fixtures=[Path(__file__),root/'Scripts/ci/tests/Q02PublicConsumer.swift',root/'Scripts/ci/tests/Q02InitializerAdmission.swift']
for p in fixtures:(out/p.name).write_bytes(p.read_bytes())
provenance={'sourceHashes':{str(p.relative_to(root)):sha(p) for p in root.glob('Sources/**/*.swift')},'fixtures':{p.name:sha(p) for p in fixtures},'artifacts':{str(p.relative_to(out)):sha(p) for p in artifacts},'runs':[]}
def run(name,argv,env=None):
 log=out/(name+'.log')
 with log.open('wb') as stream:
  child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  timedout=False;cleanup=False
  try:status=child.wait(timeout=15)
  except subprocess.TimeoutExpired:
   timedout=True
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   try:child.wait(timeout=2)
   except subprocess.TimeoutExpired:cleanup=True
   status=None
 record={'name':name,'argv':argv,'pid':child.pid,'exitStatus':status,'timedOut':timedout,'cleanupTimedOut':cleanup,'deadlineSeconds':15,'cleanupSeconds':2,'logSHA256':sha(log)}
 provenance['runs'].append(record)
 if (out/'consumer').exists():provenance['executableSHA256']=sha(out/'consumer')
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print(name,status,timedout,flush=True)
 assert not timedout and status==0,record
 return log
includes=['-I',str(snap['modules']),'-Xcc','-fmodule-map-file='+str(snap['modulemap']),'-Xcc','-I'+str(root/'Sources/VectorCoreC/include')]
run('initializer-admission',['swiftc','-swift-version','6','-typecheck',*includes,str(out/'Q02InitializerAdmission.swift')])
run('compile',['swiftc','-O','-swift-version','6','-parse-as-library',*includes,str(out/'Q02PublicConsumer.swift'),*map(str,objects),'-o',str(out/'consumer')])
env=os.environ.copy()
for key in ['CI','GITHUB_ACTIONS','TRAVIS','XCTestConfigurationFilePath','VECTORCORE_DISABLE_TUNER_CACHE','__CFBundleIdentifier']:env.pop(key,None)
log=run('consumer',[str(out/'consumer'),str(out/'temporary-store')],env)
assert 'Q02_PUBLIC_BODY_BEGIN' in log.read_text() and 'Q02_PUBLIC_RESULT PASS' in log.read_text() and 'Q02_PUBLIC_BODY_END' in log.read_text()
provenance['storeArtifacts']={str(p.relative_to(out)):sha(p) for p in (out/'temporary-store').rglob('*') if p.is_file()}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Q02_PUBLIC_DRIVER_COMPLETE')
