"""Complete synthetic C receipts for compiler-binding regressions; never run."""
import copy
import json
from pathlib import Path

from qualification_fixture import sha, write_json
from native_qualification import route_references


def add_c_matrix(f):
    f.identity['sdk'] = str(f.sdk.parent)
    f.capture['identity'] = copy.deepcopy(f.identity)
    f.profile['identity'] = copy.deepcopy(f.identity)
    root = f.evidence/'c-modes'
    root.mkdir()
    env = {'PATH':str(f.compiler.parent), 'SDKROOT':str(f.sdk.parent), 'SWIFT_EXEC':str(f.compiler)}
    def receipt(directory,name,argv,stdout='',stderr='',code=0):
        streams={}
        for label,text in [('stdout',stdout),('stderr',stderr)]:
            p=directory/(name+'.'+label)
            p.write_text(text)
            streams[label]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
        write_json(directory/(name+'.json'),dict(status='completed',timeout=False,exit=code,
            argv=list(map(str,argv)),cwd=str(f.source),environment=env,**streams))
    def artifact(path): return {'path':str(path),'sha256':sha(path)}
    cells=[]
    for mode in ('debug','release'):
        for enabled in (False,True):
            name=mode+('-c-on' if enabled else '-c-off')
            directory=root/name
            directory.mkdir()
            argv=['swift','build','-c',mode]+(['-Xswiftc','-DVC_USE_C_KERNELS'] if enabled else [])
            receipt(directory,'build',argv)
            write_json(directory/'artifacts.json',[dict(buildPath=str(f.object),retainedPath=str(f.object),sha256=sha(f.object))])
            provenance={key:artifact(f.source/'Sources/Library.swift') for key in
                ('originalDispatcher','originalSource','derivativeSource','consumerSource','shimSource')}
            provenance.update(consumer=artifact(f.binary),linkedObjects=[artifact(f.object)])
            write_json(directory/'link-provenance.json',provenance)
            receipt(directory,'link-public-routes',['swiftc','-O' if mode=='release' else '-Onone',f.object,'-o',f.binary])
            rows=[]
            for (dimension, shape, metric), reference in route_references().items():
                calls=3 if enabled and (shape=='equal' or (shape=='unequal' and metric!='euclidean')) else 0
                rows.append(dict(dimension=dimension,mode=shape,metric=metric,actual=reference,
                                 reference=reference,calls=calls,expectedCalls=calls))
            output='\n'.join(['V02_ROUTE_ENTER',*['V02_ROUTE '+json.dumps(r) for r in rows],
                *[f'V02_C_TAIL {n} calls=1' for n in (0,1,3,4,15,16,17,31,32,33)],
                'V02_ROUTE_COMPLETE widths=3 routes=36 tails=10'])+'\n'
            receipt(directory,'public-routes',[f.binary,'1' if enabled else '0'],stdout=output)
            for request in ('absent','matching'):
                receipt(directory,'benchmark-'+request,[f.binary])
                write_json(directory/(request+'-report.json'),dict(metadata=dict(coreConfiguration=mode,
                    benchmarkConfiguration=mode,flags=dict(useCKernels=enabled,requestedCKernels=enabled if request=='matching' else None),
                    selectedCaseIDs=['dist.euclidean.512.optimized']),failures=[],results=[dict(iterations=2)]))
            receipt(directory,'benchmark-mismatch',[f.binary],stderr='does not match compiled',code=1)
            cells.append(dict(name=name,configuration=mode,cKernels=enabled,routeRows=36,directCTails=10,
                artifactManifest=str(directory/'artifacts.json'),linkProvenance=str(directory/'link-provenance.json'),consumerSHA256=sha(f.binary)))
    write_json(root/'result.json',dict(schemaVersion=1,marker='V02_BUILD_MODES_COMPLETE',cells=cells))
    f.add_command('c-driver',[str(f.binary)],'',0)
    f.capture['commands']['c-driver']=f.commands['c-driver']['receipt']
    f.capture['checks']=[dict(id='c-modes',format='c-build-modes',command='c-driver',result=str(root/'result.json'))]
    f.profile['checks']=copy.deepcopy(f.capture['checks'])
    f.save(repin=True)
    return root
