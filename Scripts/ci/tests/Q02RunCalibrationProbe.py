#!/usr/bin/env python3
"""Snapshot and bounded public-client baseline/final selection observations."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'Scripts/ci'))
import swiftpm_products as products
# argv[1] is the SwiftPM bin path (swift build -c release --show-bin-path); native and swiftbuild layouts both work.
build=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
baseline=sys.argv[3]=='baseline';coverage=len(sys.argv)>4 and sys.argv[4]=='coverage'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snap=products.snapshot_consumer_inputs(build,out)
objects=snap['objects']
artifacts=[dst for _,dst in snap['records']]+[snap['modulemap']]
fixtures=[Path(__file__),root/'Scripts/ci/tests/Q02CalibrationProbe.swift',root/'Scripts/ci/tests/Q02Allocation.c']
for p in fixtures:(out/p.name).write_bytes(p.read_bytes())
provenance={'baseline':baseline,'coverage':coverage,'sourceHashes':{str(p.relative_to(root)):sha(p) for p in root.glob('Sources/**/*.swift')},'fixtureHashes':{p.name:sha(p) for p in fixtures},'artifacts':{str(p.relative_to(out)):sha(p) for p in artifacts},'runs':[]}
def run(name,argv,expected=0,env=None):
 log=out/(name+'.log')
 with log.open('wb') as file:
  p=subprocess.Popen(argv,stdout=file,stderr=subprocess.STDOUT,env=env,start_new_session=True);timeout=False;cleanup=False
  try:status=p.wait(timeout=15)
  except subprocess.TimeoutExpired:
   timeout=True
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   try:p.wait(timeout=2)
   except subprocess.TimeoutExpired:cleanup=True
   status=None
 record={'name':name,'argv':argv,'exitStatus':status,'expectedStatus':expected,'timedOut':timeout,'cleanupTimedOut':cleanup,'executionDeadlineSeconds':15,'cleanupDeadlineSeconds':2,'pid':p.pid,'log':str(log),'logSHA256':sha(log),'environmentOverrides':{k:v for k,v in (env or {}).items() if k.startswith(('LLVM_','DYLD_'))}}
 provenance['runs'].append(record)
 if (out/'probe').is_file():provenance['executableSHA256']=sha(out/'probe')
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(name,status,timeout,flush=True)
 assert not timeout and status==expected,record
 return log
run('compile-c',['clang','-O2','-fno-builtin-malloc','-fno-builtin-free','-c',str(out/'Q02Allocation.c'),'-o',str(out/'allocation.o')])
flags=(['-D','Q02_BASELINE'] if baseline else [])+(['-profile-generate','-profile-coverage-mapping'] if coverage else [])
run('compile-swift',['swiftc','-O','-swift-version','6','-parse-as-library',*flags,'-I',str(snap['modules']),'-Xcc','-fmodule-map-file='+str(snap['modulemap']),'-Xcc','-I'+str(root/'Sources/VectorCoreC/include'),str(out/'Q02CalibrationProbe.swift'),str(out/'allocation.o'),*map(str,objects),'-o',str(out/'probe')])
env=os.environ.copy()
for key in ['CI','GITHUB_ACTIONS','TRAVIS','XCTestConfigurationFilePath','VECTORCORE_DISABLE_TUNER_CACHE','__CFBundleIdentifier']:env.pop(key,None)

for width in [512,768,1536]:
 for count in ([1] if coverage else [50,500]):
  for mode in (['work'] if coverage else ['measure','work']):
   childenv=env.copy()
   if coverage:childenv['LLVM_PROFILE_FILE']=str(out/f'{mode}-{width}-{count}.profraw')
   log=run(f'{mode}-{width}-{count}',[str(out/'probe'),mode,str(width),str(count)],env=childenv)
   assert 'Q02_BODY_BEGIN' in log.read_text() and 'Q02_BODY_END' in log.read_text()
print('Q02_PROBE_COMPLETE',flush=True)
