"""Bounded lexical guard for the two serialization suites, not semantic proof."""
import re
from pathlib import Path

def mask_comments_and_strings(source):
    out=list(source);i=0
    def blank(a,b):
        for k in range(a,b):
            if out[k]!='\n':out[k]=' '
    while i<len(source):
        if source.startswith('//',i):
            end=source.find('\n',i);end=len(source) if end<0 else end;blank(i,end);i=end
        elif source.startswith('/*',i):
            start=i;i+=2;depth=1
            while i<len(source) and depth:
                if source.startswith('/*',i):depth+=1;i+=2
                elif source.startswith('*/',i):depth-=1;i+=2
                else:i+=1
            if depth:raise ValueError('unterminated block comment')
            blank(start,i)
        else:
            match=re.match(r'(#+)?("""|")',source[i:])
            if not match:i+=1;continue
            start=i;hashes=match.group(1) or '';quote=match.group(2);i+=len(match.group(0));end=quote+hashes
            while i<len(source):
                if not hashes and source[i]=='\\':i+=2;continue
                if source.startswith(end,i):i+=len(end);break
                i+=1
            else:raise ValueError('unterminated string')
            blank(start,i)
    return ''.join(out)

def registered_bodies(source):
    masked=mask_comments_and_strings(source);result=[]
    for m in re.finditer(r'@Test\b[^@]*?\bfunc\s+(\w+)[^{]*\{',masked):
        start=m.end();i=start;depth=1
        while i<len(masked) and depth:
            if masked[i]=='{':depth+=1
            elif masked[i]=='}':depth-=1
            i+=1
        if depth:raise ValueError('unterminated test body')
        result.append((m.group(1),start,i-1,masked[start:i-1]))
    return result

def empty_tests(source):
    return [name for name,_,_,body in registered_bodies(source) if not body.strip()]

def golden_errors(source):
    errors=[]
    for name in ['emptyFrame','ordinaryFrame','specialFrame']:
        match=re.search(r'static\s+let\s+'+name+r'\s*:\s*\[UInt8\]\s*=\s*\[([^\]]*)\]',source)
        if not match or len([x for x in match.group(1).split(',') if x.strip()])<16:errors.append(name)
    return errors

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();files=['SerializationEdgeCaseTests.swift','VectorSerializationTests.swift']
    report={name:empty_tests((a.root/'Tests/ComprehensiveTests'/name).read_text()) for name in files}
    report['goldens']=golden_errors((a.root/'Tests/ComprehensiveTests/B01Golden.swift').read_text())
    print(json.dumps(report,indent=2))
    raise SystemExit(1 if any(report.values()) else 0)
