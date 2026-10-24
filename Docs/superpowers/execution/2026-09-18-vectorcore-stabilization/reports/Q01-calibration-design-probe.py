"""Root design experiment only; independent from production, not a regression closure."""
import json, math, random, struct

def f32(x):
    return struct.unpack('>f', struct.pack('>f', x))[0]
def bits(x):
    return struct.unpack('>I', struct.pack('>f', x))[0]
def from_bits(x):
    return struct.unpack('>f', struct.pack('>I', x))[0]
def away(x):
    return math.floor(x + .5) if x >= 0 else math.ceil(x - .5)
MAX = from_bits(0x7f7fffff)
MIN = from_bits(1)
def params(lo, hi, symmetric):
    if lo == 0 and hi == 0: return 1., 0
    if symmetric:
        ideal = max(abs(lo), abs(hi)) / 127
        z = 0
    else:
        low, high = min(lo, 0), max(hi, 0)
        ideal = (high-low) / 255
        z = min(127, max(-128, away(-128-low/ideal)))
    scale = max(MIN, f32(ideal))
    # Bound every legal code's linear decode before final Float projection.
    cap = MAX/max(abs(-128-z), abs(127-z))
    rounded_cap = f32(cap)
    if rounded_cap > cap: rounded_cap = from_bits(bits(rounded_cap)-1)
    return min(scale, rounded_cap), z

def quantize(x, scale, z):
    return min(127, max(-128, away(x/scale+z)))

def run():
    rng = random.Random(20260919)
    edges = [0., MIN, -MIN, from_bits(0x007fffff), from_bits(0x00800000), 1., -1., MAX, -MAX]
    pairs = [(min(a,b), max(a,b)) for a in edges for b in edges]
    for _ in range(100000):
        a = from_bits(rng.randrange(0x7f800000)) * rng.choice((-1,1))
        b = from_bits(rng.randrange(0x7f800000)) * rng.choice((-1,1))
        pairs.append((min(a,b),max(a,b)))
    worst = None
    count = 0
    for lo, hi in pairs:
        for symmetric in (False,True):
            scale,z = params(lo,hi,symmetric)
            assert math.isfinite(scale) and scale > 0 and -128 <= z <= 127
            for q in (-128,127): assert abs((q-z)*scale) <= MAX
            for x in (lo,hi):
                q=quantize(x,scale,z)
                decoded=f32((q-z)*scale)
                assert math.isfinite(decoded)
                ratio=abs(decoded-x)/scale
                if worst is None or ratio>worst['error_in_steps']:
                    worst=dict(lo=lo,hi=hi,symmetric=symmetric,scale=scale,z=z,x=x,q=q,decoded=decoded,error_in_steps=ratio)
                # Deliberately generous proposed regression bound, not universal proof.
                assert ratio <= 2
                count += 1
    return dict(status='design experiment only, no production validation',seed=20260919,intervals=len(pairs),endpoint_checks=count,worst=worst)
if __name__=='__main__': print(json.dumps(run(),indent=2))
