# Compatible quiet-host measurement procedure

Run after the combined candidate source and compiler have settled, all builds/tests on the host have stopped, and the host has cooled. The commands below establish a new baseline for the repaired 21 cases. They cannot establish an old-to-new regression verdict: the old dot/normalization validation boundaries and copy fixture differ. Never add `--allow-incompatible` or `--allow-missing`, remove recorded failures, or substitute a different seed/statistic to obtain acceptance.

Use a fresh output directory outside the checkout. From the clean integrated checkout, set `VC_PHASE2_OUT` to that absolute directory and `VC_PHASE2_SWIFT_BIN` to the already verified Swift.org 6.4.2 `usr/bin` directory. Keep the recorded SDK unchanged. The following preparation uses the existing bounded capture and product-discovery helpers; it does not time a candidate while compiling.

```python
import os, sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'Scripts/ci'))
from v03_process import run, sha, test_environment, write_json
from swiftpm_products import resolve_bin_path, find_executable
out = Path(os.environ['VC_PHASE2_OUT']).resolve()
out.mkdir(parents=True, exist_ok=False)
bin_dir = Path(os.environ['VC_PHASE2_SWIFT_BIN']).resolve()
env = test_environment()
env['PATH'] = str(bin_dir) + ':' + env['PATH']
env['SDKROOT'] = '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk'
frontend = bin_dir / 'swift-frontend'
if sha(frontend) != '956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301':
    raise ValueError('Compiler identity changed: requalify the procedure')
sdk_settings = Path(env['SDKROOT']) / 'SDKSettings.json'
if sha(sdk_settings) != '82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6':
    raise ValueError('SDK identity changed: requalify the procedure')
_, checkout = run(['git', 'rev-parse', 'HEAD'], out, 'checkout', cwd=root, environment=env)
_, status = run(['git', 'status', '--porcelain'], out, 'status', cwd=root, environment=env)
if status.strip(): raise ValueError('Candidate checkout must be clean')
flags = ['-c', 'release', '--jobs', '2', '--scratch-path', str(out / 'build')]
run([bin_dir / 'swift', 'build', *flags, '--product', 'vectorcore-bench'], out, 'build',
    cwd=root, environment=env, timeout=900)
products = resolve_bin_path(str(bin_dir / 'swift'), flags, out, 'bin-path', cwd=root, environment=env)
binary = find_executable(products, 'vectorcore-bench')
comparator = out / 'compare-benchmarks'
run([bin_dir / 'swiftc', '-module-cache-path', out / 'comparator-cache',
     root / 'Scripts/compare_benchmarks.swift', '-o', comparator], out, 'compile-comparator',
    cwd=root, environment=env, timeout=120)
inputs = [root / 'Package.swift', root / 'Scripts/compare_benchmarks.swift']
inputs += [p for directory in ['Sources', 'Benchmarks'] for p in (root / directory).rglob('*') if p.is_file()]
write_json(out / 'prepared.json', {
    'root': str(root), 'gitSHA': checkout.strip(), 'binary': str(binary), 'binarySHA256': sha(binary),
    'comparator': str(comparator), 'comparatorSHA256': sha(comparator),
    'frontendSHA256': sha(frontend), 'sdkSHA256': sha(sdk_settings),
    'sources': {str(p.relative_to(root)): sha(p) for p in sorted(inputs)}})
```

After preparation finishes, confirm there are no competing compiler/test jobs and record the host model, power mode, load and available thermal observations. Allow cooling before launching the next block. Keep the same root, output directory, toolchain path and environment. This phase deliberately uses seed 0, adaptive measurement time of 0.25 seconds per sample, 15 samples, three full repeats and the median per declared work unit. The benchmark performs its normal warm-up; preserve its raw samples and warm-up counts.

```python
import json, os, sys, time
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'Scripts/ci'))
from v03_process import run, sha, test_environment, write_json
out = Path(os.environ['VC_PHASE2_OUT']).resolve()
p = json.loads((out / 'prepared.json').read_text())
env = test_environment()
env['PATH'] = os.environ['VC_PHASE2_SWIFT_BIN'] + ':' + env['PATH']
env['SDKROOT'] = '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk'
if str(root) != p['root']: raise ValueError('Checkout changed')
_, checkout = run(['git', 'rev-parse', 'HEAD'], out, 'checkout-before', cwd=root, environment=env)
if checkout.strip() != p['gitSHA']: raise ValueError('Candidate commit changed')
if sha(Path(os.environ['VC_PHASE2_SWIFT_BIN']) / 'swift-frontend') != p['frontendSHA256']:
    raise ValueError('Compiler changed')
if sha(Path(env['SDKROOT']) / 'SDKSettings.json') != p['sdkSHA256']:
    raise ValueError('SDK changed')
for relative, expected in p['sources'].items():
    if sha(root / relative) != expected: raise ValueError('Source changed: ' + relative)
for key in ['binary', 'comparator']:
    if sha(p[key]) != p[key + 'SHA256']: raise ValueError('Product changed: ' + key)
run(['sysctl', 'hw.model', 'hw.ncpu', 'hw.memsize'], out, 'hardware', cwd=root, environment=env)
run(['pmset', '-g'], out, 'power', cwd=root, environment=env)
base = [p['binary'], '--suites', 'dot,normalize,memory', '--dims', '512,768,1536',
        '--filter', 'dot.*,normalize.*,mem.copy.*', '--no-ab', '--run-seed', '0',
        '--samples', '15', '--min-time', '0.25', '--strict-correctness', '--format', 'json']
comparisons = []
for index in range(3):
    name = 'repeat-' + str(index)
    run(['ps', '-axo', 'pid,comm,pcpu,pmem'], out, name + '-processes', cwd=root, environment=env)
    run(['uptime'], out, name + '-load', cwd=root, environment=env)
    run(['pmset', '-g', 'therm'], out, name + '-thermal-before', cwd=root, environment=env, expected=None)
    report = out / (name + '-report.json')
    run([*base, '--out', report], out, name, cwd=root, environment=env, timeout=900)
    result = json.loads(report.read_text())
    if result['failures'] or len(result['results']) != 21: raise ValueError('Invalid candidate report')
    run(['pmset', '-g', 'therm'], out, name + '-thermal-after', cwd=root, environment=env, expected=None)
    if index:
        record, _ = run([p['comparator'], '--baseline', out / 'repeat-0-report.json', '--current', report,
                         '--metric', 'median', '--per-unit', '--threshold', '5'], out, name + '-compare',
                        cwd=root, environment=env, expected=(0, 1))
        comparisons.append({'repeat': index, 'exit': record['exit']})
    if index < 2: time.sleep(30)
for relative, expected in p['sources'].items():
    if sha(root / relative) != expected: raise ValueError('Source changed during measurement: ' + relative)
_, checkout = run(['git', 'rev-parse', 'HEAD'], out, 'checkout-after', cwd=root, environment=env)
_, status = run(['git', 'status', '--porcelain'], out, 'status-after', cwd=root, environment=env)
if checkout.strip() != p['gitSHA'] or status.strip(): raise ValueError('Checkout changed during measurement')
write_json(out / 'comparison-observations.json', comparisons)
```

Review the host records, thermal availability, per-case raw samples/RSD, all strict correctness records and both comparison exits before deciding acceptance. An exit 1 is a retained timing/missing-case failure; do not relabel it as success. An exit 2 aborts the procedure as an invalid comparison. `pmset` may provide no current thermal-pressure reading; that absence is a limitation, not proof of no throttling. Process/load snapshots also do not prove the host was idle between snapshots. If host control or compatibility is insufficient, retain the run as an observation and leave acceptance pending. Keep tested binaries, source/compiler/SDK/product hashes and command logs, excluding bulk build/module caches from the evidence archive. Broader benchmark families need their own consistently selected compatible reports.
