"""Offline binary64 references; no production helper or runtime test dependency.

Run from any directory. All inputs are exact IEEE-754 bit patterns. Decimal's
2500/3000 digit calculations must agree after rounding to binary64. The fixture
block is replaced idempotently. Evidence is written beside the caller's temp logs.
"""
from decimal import Decimal, localcontext
from pathlib import Path
import json
import struct


def bits(value):
    return struct.unpack('>Q', struct.pack('>d', value))[0]


def value(pattern):
    return struct.unpack('>d', struct.pack('>Q', pattern))[0]


normalization_inputs = [
    [0x7fefffffffffffff, 1, 0xffdfffffffffffff, 0],
    [1, 2, 3, 0x8000000000000007],
    [0x0010000000000000, 0x000fffffffffffff, 0x8010000000000001],
    [0x7fd123456789abcd, 0xffe0123456789abc, 0x5feabcdef0123456, 0x12345],
]
division_inputs = [
    ([0x0010000000000001, 0x800fffffffffffff, 1, 0x8000000000000000], 3),
    ([0x7fefffffffffffff, 0xffefffffffffffff, 0x0010000000000000, 1], 0x7feffffffffffffe),
    ([0x3ff123456789abcd, 0xbfd3456789abcdef, 0x000fffffffffffff], 0x3fe23456789abcde),
    ([1, 2, 0x8000000000000003, 0x0010000000000001], 0x0010000000000000),
]
precisions = [2500, 3000]
normalization_outputs = []
residuals = []
for row in normalization_inputs:
    rounded_runs = []
    for precision in precisions:
        with localcontext() as context:
            context.prec = precision
            exact = [Decimal.from_float(value(pattern)) for pattern in row]
            norm = sum(x*x for x in exact).sqrt()
            rounded = [bits(float(x/norm)) for x in exact]
            rounded_runs.append(rounded)
            if precision == 3000:
                rounded_exact = [Decimal.from_float(value(pattern)) for pattern in rounded]
                residual = abs(sum(x*x for x in rounded_exact).sqrt()-1)
                assert residual <= Decimal(32) * Decimal.from_float(2**-52)
                residuals.append(str(residual))
    assert rounded_runs[0] == rounded_runs[1]
    normalization_outputs.append(rounded_runs[0])

division_outputs = []
for row, divisor in division_inputs:
    rounded_runs = []
    for precision in precisions:
        with localcontext() as context:
            context.prec = precision
            denominator = Decimal.from_float(value(divisor))
            rounded_runs.append([bits(float(Decimal.from_float(value(pattern))/denominator)) for pattern in row])
    assert rounded_runs[0] == rounded_runs[1]
    division_outputs.append(rounded_runs[0])


def swift_array(row):
    return '[' + ', '.join(hex(pattern) for pattern in row) + ']'




def wrap_swift_lines(text):
    lines = []
    for line in text.splitlines():
        indent = ' ' * (len(line) - len(line.lstrip()) + 4)
        while len(line) > 200:
            split = line.rfind(',', 0, 175) + 1
            assert split > 0
            lines.append(line[:split])
            line = indent + line[split:].lstrip()
        lines.append(line)
    return '\n'.join(lines) + '\n'

text = '\n// Generated independently by N01-reference-generator.py at 2500 and 3000 decimal digits.\n'
text += 'let n01DoubleFixtures: [([UInt64], [UInt64])] = [\n'
for inputs, outputs in zip(normalization_inputs, normalization_outputs):
    text += f'    ({swift_array(inputs)}, {swift_array(outputs)}),\n'
text += ']\n'
text += 'let n01DoubleDivisionFixtures: [([UInt64], UInt64, [UInt64])] = [\n'
for (inputs, divisor), outputs in zip(division_inputs, division_outputs):
    text += f'    ({swift_array(inputs)}, {hex(divisor)}, {swift_array(outputs)}),\n'
text += ']\n// End independently generated N01 references.\n'
path = Path(__file__).resolve().parents[1] / 'Tests/ComprehensiveTests/NumericalReferenceFixtures.swift'
source = path.read_text()
start = source.index('\n// Generated independently by N01-reference-generator.py')
marker = '// End independently generated N01 references.\n'
end = source.index(marker, start)+len(marker) if marker in source else source.index('\n]\n',start)+3
path.write_text(source[:start]+wrap_swift_lines(text)+source[end:])
Path('/private/tmp/N01-reference-stability.json').write_text(json.dumps({
    'precisions': precisions, 'identical_rounded_outputs': True,
    'normalization_inputs': normalization_inputs, 'normalization_outputs': normalization_outputs,
    'rounded_oracle_norm_residuals_at_3000_digits': residuals,
    'division_inputs': division_inputs, 'division_outputs': division_outputs,
},indent=2)+'\n')
