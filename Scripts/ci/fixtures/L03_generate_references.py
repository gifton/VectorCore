"""Independent derivative-free curve minima and Decimal smooth references.

No production code or LM/Jacobian is imported. Inputs are explicit Binary32 words.
Nelder-Mead is run from three starts; Decimal controls reproduce at 110/160 digits.
"""
import argparse
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import struct


def f32(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def objective(point, ratio):
    alpha, beta = math.exp(point[0]), math.exp(point[1])
    return math.fsum((1 / (1 + alpha * (3*i/299)**(2*beta)) -
                      (1 if 3*i/299 < ratio else math.exp(-(3*i/299-ratio))))**2
                     for i in range(1, 300))


def minimize(ratio, start):
    simplex = [list(start), [start[0]+0.25, start[1]], [start[0], start[1]+0.25]]
    for iteration in range(5000):
        simplex.sort(key=lambda p: objective(p, ratio))
        scores = [objective(p, ratio) for p in simplex]
        diameter = max(abs(simplex[i][j]-simplex[0][j]) for i in [1,2] for j in [0,1])
        if diameter < 2e-9 and scores[-1]-scores[0] < 1e-14:
            return simplex[0], scores[0], iteration, diameter
        center = [(simplex[0][j]+simplex[1][j])/2 for j in [0,1]]
        reflection = [2*center[j]-simplex[2][j] for j in [0,1]]
        fr = objective(reflection, ratio)
        if fr < scores[0]:
            expansion = [center[j]+2*(reflection[j]-center[j]) for j in [0,1]]
            simplex[2] = expansion if objective(expansion, ratio) < fr else reflection
        elif fr < scores[1]:
            simplex[2] = reflection
        else:
            outer = fr < scores[2]
            target = reflection if outer else simplex[2]
            contraction = [(center[j]+target[j])/2 for j in [0,1]]
            if objective(contraction, ratio) < (fr if outer else scores[2]):
                simplex[2] = contraction
            else:
                simplex = [simplex[0]] + [[(simplex[0][j]+p[j])/2 for j in [0,1]] for p in simplex[1:]]
    raise RuntimeError('Nelder-Mead failed stopping criterion')


def smooth(words, precision):
    with localcontext() as c:
        c.prec = precision
        values = [Decimal.from_float(f32(word)) for word in words]
        rho = min(x for x in values if x>0)
        deltas = [max(Decimal(0), x-rho) for x in values]
        target = Decimal(len(values)+1).ln()/Decimal(2).ln()
        # Independent physical-bandwidth bisection, not log-normalized production search.
        low, high = Decimal(0), max(deltas)*2
        for _ in range(1200):
            middle = (low+high)/2
            total = sum((-delta/middle).exp() if delta else Decimal(1) for delta in deltas)
            if total < target: low = middle
            else: high = middle
            if abs(total-target) < Decimal('1e-90'): break
        else: raise RuntimeError('Decimal reference did not converge')
        return {'sigma':float(middle), 'rho':float(rho), 'target':float(target),
                'weights':[float((-d/middle).exp()) if d else 1.0 for d in deltas],
                'iterations':_+1}


def generate():
    curves=[]
    for word in [0,0x3dcccccd,0x3e000000,0x3e800000,0x3f000000,0x3f400000,0x3f800000]:
        ratio=f32(word)
        results=[minimize(ratio,start) for start in [(0,0),(-0.5,-0.5),(1,0.5)]]
        best=min(results,key=lambda x:x[1]);a,b=map(math.exp,best[0])
        assert max(x[1] for x in results)-best[1]<1e-12
        curves.append({'ratioWord':word,'ratio':ratio,'alpha':a,'beta':b,'sse':best[1],
                       'starts':[{'point':x[0],'sse':x[1],'iterations':x[2],'diameter':x[3]} for x in results]})
    regression=[]
    for min_word, spread_word in [(0x3f000000,0x3fc00000),(0x3e800000,0x3f333333)]:
        ratio=f32(min_word)/f32(spread_word)
        results=[minimize(ratio,start) for start in [(0,0),(-0.5,-0.5),(1,0.5)]]
        best=min(results,key=lambda x:x[1]);a,b=map(math.exp,best[0])
        assert max(x[1] for x in results)-best[1]<1e-12
        regression.append({'minDistWord':min_word,'spreadWord':spread_word,'ratio':ratio,
                           'alpha':a,'beta':b,'sse':best[1],
                           'starts':[{'point':x[0],'sse':x[1],'iterations':x[2],'diameter':x[3]} for x in results]})
    rows=[]
    for words in [[0x3f800000,0x40000000],[0x3f800000,0x40000000,0x40400000],
                  [1,2],[0x00800000,0x01000000],[0x7e800000,0x7f000000],
                  [1,2,0x7f7fffff],[0x3f800000,0x3f800001,0x40000000]]:
        a=smooth(words,110);b=smooth(words,160)
        assert {k:v for k,v in a.items() if k!='iterations'} == {k:v for k,v in b.items() if k!='iterations'}
        rows.append({'words':words,**a})
    return {'solver':'multi-start derivative-free Nelder-Mead; objective summed with math.fsum',
            'curveStop':'simplex coordinate diameter <2e-9 and SSE spread <1e-14, max5000',
            'smoothPrecision':[110,160],'smoothStop':'absolute target residual <1e-90, max1200 physical bisections',
            'curves':curves,'regressionCurves':regression,'smooth':rows, 'curveCases':[{'reference':i,'spreadWord':word,'minDistWord':struct.unpack('<I',struct.pack('<f',f32(word)*row['ratio']))[0]} for i,row in enumerate(curves) for word in [0x2f800000,0x37800000,0x3f800000,0x47800000,0x4f800000]]}


def swift(data):
    lines=['// Generated by L03_generate_references.py; independent of production solvers.',
           'enum L03Reference {', '    static let curves: [(word: UInt32, alpha: Double, beta: Double, sse: Double)] = [']
    for row in data['curves']:
        lines.append(f"        (0x{row['ratioWord']:08x}, {row['alpha']!r}, {row['beta']!r}, {row['sse']!r}),")
    lines += ['    ]', '    static let curveCases: [(reference: Int, spreadWord: UInt32, minDistWord: UInt32)] = [']
    for row in data['curveCases']:
        lines.append(f"        ({row['reference']}, 0x{row['spreadWord']:08x}, 0x{row['minDistWord']:08x}),")
    lines += ['    ]','    static let regressionCurves: [(minDistWord: UInt32, spreadWord: UInt32, alpha: Double, beta: Double, sse: Double)] = [']
    for row in data['regressionCurves']:
        lines.append(f"        (0x{row['minDistWord']:08x}, 0x{row['spreadWord']:08x}, {row['alpha']!r}, {row['beta']!r}, {row['sse']!r}),")
    lines += ['    ]','    static let smooth: [(words: [UInt32], sigma: Double, rho: Double, target: Double, weights: [Double])] = [']
    for row in data['smooth']:
        words=', '.join(f'0x{x:08x}' for x in row['words'])
        lines.append(f"        ([{words}], {row['sigma']!r}, {row['rho']!r}, {row['target']!r}, {row['weights']!r}),")
    lines += ['    ]','}','']
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--record',type=Path);a=p.parse_args()
    data=generate();text=swift(data);path=Path(__file__).resolve().parents[3]/'Tests/ComprehensiveTests/L03Reference.swift'
    if a.check:
        assert path.read_text()==text,'Reference fixture differs'
    else:path.write_text(text)
    if a.record:a.record.write_text(json.dumps(data,indent=2)+'\n')
    print('L03_REFERENCE_PASS curves=7 regressionCurves=2 starts=3 smooth=7 precision=110/160')
