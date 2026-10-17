#!/usr/bin/env python3
"""Immutable Release artifact snapshot and bounded actual-library allocation probe."""
import hashlib,json,os,signal,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'Scripts/ci'))
import swiftpm_products as products
# argv[1] is the SwiftPM bin path (swift build -c release --show-bin-path); native and swiftbuild layouts both work.
build=Path(sys.argv[1])
output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snap=products.snapshot_consumer_inputs(build,output)
assert snap['objects']
records=[{'path':str(src),'sha256':sha(dst)} for src,dst in snap['records']]+[{'path':'VectorCoreC.modulemap','sha256':sha(snap['modulemap'])}]
fixtures=[Path(__file__),root/'Scripts/ci/tests/Q04Allocation.c',root/'Scripts/ci/tests/Q04AllocationProbe.swift']
for p in fixtures:(output/p.name).write_bytes(p.read_bytes())
sources={str(p.relative_to(root)):sha(p) for p in root.glob('Sources/**/*.swift')}
commands=[['clang','-O2','-fno-builtin-malloc','-fno-builtin-free','-c',str(output/'Q04Allocation.c'),'-o',str(output/'allocation.o')],['swiftc','-O','-swift-version','6','-I',str(snap['modules']),'-Xcc','-fmodule-map-file='+str(snap['modulemap']),'-Xcc','-I'+str(root/'Sources/VectorCoreC/include'),str(output/'Q04AllocationProbe.swift'),str(output/'allocation.o'),*map(str,snap['objects']),'-o',str(output/'probe')],[str(output/'probe')]]
provenance={'build':str(build),'snapshot':str(output),'sourceHashes':sources,'fixtureHashes':{p.name:sha(p) for p in fixtures},'artifacts':records,'runs':[]}
for index,command in enumerate(commands):
 path=output/f'phase-{index}.log'
 with path.open('wb') as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  timeout=False;cleanup=False
  try:status=process.wait(timeout=15)
  except subprocess.TimeoutExpired:
   timeout=True
   try:os.killpg(process.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   try:status=process.wait(timeout=2)
   except subprocess.TimeoutExpired:status=None;cleanup=True
 provenance['runs'].append({'command':command,'exit':status,'timedOut':timeout,'cleanupTimedOut':cleanup,'pid':process.pid,'executionDeadlineSeconds':15,'cleanupDeadlineSeconds':2,'log':str(path),'logSHA256':sha(path)})
 if (output/'probe').exists():provenance['executableSHA256']=sha(output/'probe')
 (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print(f'phase={index} exit={status} timeout={timeout} cleanupTimeout={cleanup} log={path}',flush=True)
 if timeout or status!=0:raise SystemExit(124 if timeout else status)
assert 'Q04_PROBE_COMPLETE' in (output/'phase-2.log').read_text()
