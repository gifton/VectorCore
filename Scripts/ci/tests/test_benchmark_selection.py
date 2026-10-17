#!/usr/bin/env python3
"""Bounded real CLI selection/gating matrix. Each child retains command/status/files."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("v02_capture", ROOT / "Scripts/ci/check_benchmark_build_modes.py")
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();binary=args.binary.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[]
    def run(name,arguments,expected_ids=None,exit_code=0,strict=False,scopes=("reused-input",),unavailable=(),repeats=1):
        folder=out/name;folder.mkdir();report=folder/"report.json"
        command=[binary,*arguments,"--min-time","0","--repeats",str(repeats),"--samples","2","--format","json","--out",report]
        if strict:command.append("--strict-correctness")
        helper.run(command,folder,"run",expected=exit_code)
        status=json.loads((folder/"run.json").read_text())["exit"]
        row={"name":name,"exit":status,"executableSHA256":helper.sha(binary)}
        if expected_ids is not None:
            value=json.loads(report.read_text())
            actual=[case["name"] for case in value["results"]]
            assert sorted(actual)==sorted(expected_ids),(name,actual,expected_ids)
            assert sorted(value["metadata"]["selectedCaseIDs"])==sorted(expected_ids)
            assert all(case["iterations"]==2*repeats and case["samples"]==2 for case in value["results"])
            # V04: schema 3, run-level boundary version, and a catalog-supplied boundary on every result.
            assert value["metadata"]["schemaVersion"]==3 and value["metadata"]["measurementBoundaryVersion"]=="V04-boundaries-1"
            assert value["metadata"]["statisticUnit"]=="per-sample-average-ns-per-operation" and value["metadata"]["invocationStatisticUnit"]=="per-sample-average-ns-per-invocation" and sorted(value["metadata"]["unavailableCaseIDs"])==sorted(unavailable)
            for case in value["results"]:
                boundary=case["boundary"]
                assert boundary["version"]=="V04-boundary-1" and boundary["timingScope"] in scopes,(name,case["name"])
                assert boundary["unitsPerInvocation"]==case["unitCount"],(name,case["name"])
                assert boundary["consumer"] and boundary["route"] and boundary["fixture"]["version"]=="V02-inputs-1",(name,case["name"])
                assert len(case["sampleNsPerOperation"])==2 and case["medianNsPerOperation"]>0 and case["stddevNsPerOperation"]>=0,(name,case["name"])
                assert abs(case["sampleNsPerOperation"][0]*boundary["unitsPerInvocation"]-case["sampleNsPerInvocation"][0])<=1e-6*case["sampleNsPerInvocation"][0],(name,case["name"])
                assert case["sampleIterations"]==[repeats,repeats] and len(case["sampleNsPerInvocation"])==2 and case["warmupIterations"]>=1,(name,case["name"])
            if strict:assert status==int(bool(value["failures"])),(name,status,value["failures"])
            row.update(actualCases=len(actual),failures=value["failures"],metadata=value["metadata"])
            rows.append(row);return value
        assert not report.exists(),(name,"invalid configuration produced a run")
        rows.append(row);return None
    for d in [512,768,1536]:
        run(f"dot-optimized-{d}",["--suites","dot","--dims",str(d),"--filter","*.optimized"],[f"dot.{d}.optimized"])
        run(f"dot-exclude-{d}",["--suites","dot","--dims",str(d),"--exclude","*.generic"],[f"dot.{d}.optimized"])
        run(f"distance-mixed-{d}",["--suites","distance","--dims",str(d),"--filter","dist.euclidean.*,dist.cosine.*","--exclude","*.generic"],[f"dist.euclidean.{d}.optimized",f"dist.cosine.{d}.optimized"])
        run(f"normalize-exact-{d}",["--suites","normalize","--dims",str(d),"--filter",f"normalize.success.{d}.optimized"],[f"normalize.success.{d}.optimized"])
        family=[f"batch.cosine.{d}.N2.optimized-{strategy}.{provider}" for provider in ["sequential","parallel","automatic"] for strategy in ["fused","preNorm"]]
        value=run(f"cosine-pairs-{d}",["--suites","batch","--dims",str(d),"--batch-ns","2","--filter","batch.cosine.*"],family)
        assert len(value["abComparisons"])==3
        value=run(f"no-ab-cosines-{d}",["--suites","batch","--dims",str(d),"--batch-ns","2","--filter","batch.cosine.*","--no-ab"],family)
        assert not value.get("abComparisons")
        pair=[f"batch.{metric}.{d}.N2.generic.sequential" for metric in ["euclidean2","euclidean"]]
        value=run(f"ab-only-complete-{d}",["--suites","batch","--dims",str(d),"--batch-ns","2","--ab-only","--filter",f"batch.euclidean*.{d}.N2.generic.sequential"],pair)
        assert len(value["abComparisons"])==1
        run(f"ab-only-incomplete-{d}",["--suites","batch","--dims",str(d),"--batch-ns","2","--ab-only","--filter",pair[0]],exit_code=1)
        for strict in [False,True]:
            value=run(f"matrix-{d}-strict-{int(strict)}",["--suites","matrix","--dims",str(d),"--batch-ns","3"],[f"matrix.euclidean2.{d}.N3.optimized"],exit_code=(0,1) if strict else 0,strict=strict)
            correctness=value["results"][0]["correctness"]
            assert correctness["status"]=="valid" and correctness["samples"]==9
            assert bool(value["failures"])==(correctness["maxRelError"]>1e-6)
        run(f"soa-{d}",["--suites","batch","--dims",str(d),"--batch-ns","100","--prefer-soa","--filter","*.optimized-soa"],[f"batch.euclidean2.{d}.N100.optimized-soa"])
        count=(12*1024*1024)//(d*4)+1
        for metric in ["euclidean2","cosine"]:
            run(f"fp16-{d}-{metric}",["--suites","batch","--dims",str(d),"--batch-ns",str(count),"--use-mixed-precision","--filter",f"batch.{metric}.*.optimized-fp16"],[f"batch.{metric}.{d}.N{count}.optimized-fp16"])
        run(f"fp16-ineligible-{d}",["--suites","batch","--dims",str(d),"--batch-ns","2","--use-mixed-precision","--filter","*.optimized-fp16"],exit_code=1)
    # Pair counts under --ab-only, one child per (width, provider). The per-case cost is dominated by the harness's fixed 200 ms
    # warmup, not by N, so the work is divided across children (6 pairs and 12 cases each) instead of shrinking N. Pairs are
    # provider-local, so each child is complete. Across the nine children the declared families keep their V02 counts under
    # boundary-declared pairing: 36 euclidean2-vs-euclidean and 18 cosine-prenorm-vs-fused.
    family_counts={"euclidean2_vs_euclidean":0,"cosine_prenorm_vs_fused":0}
    for d in [512,768,1536]:
        for provider in ["sequential","parallel","automatic"]:
            ids=[f"batch.{metric}.{d}.N{n}.{variant}.{provider}" for n in [2,3] for metric in ["euclidean2","euclidean"] for variant in ["generic","optimized"]]
            ids+=[f"batch.cosine.{d}.N{n}.optimized-{strategy}.{provider}" for n in [2,3] for strategy in ["fused","preNorm"]]
            value=run(f"ab-only-{d}-{provider}",["--suites","batch","--dims",str(d),"--batch-ns","2,3","--ab-only","--filter","*."+provider],ids)
            assert not value["abMissingSides"]
            for comparison in value["abComparisons"]:
                family_counts[comparison["family"]]+=1
                assert comparison["leftBoundary"]["timingScope"]==comparison["rightBoundary"]["timingScope"]=="reused-input"
                assert comparison["purpose"] and comparison["allowedDifferences"]
    assert family_counts=={"euclidean2_vs_euclidean":36,"cosine_prenorm_vs_fused":18},family_counts
    # Opt-in boundary suite: exact IDs per width, unavailable SoA-FP16 entries with their reason, no A/B rows, correctness on
    # every executable companion, FP16 error statistics, and a selection of only unavailable entries exits 1.
    for d in [512,768,1536]:
        executable=[f"batch.euclidean2.{d}.N{n}.{v}" for n in [2,3] for v in ["aos-fp32-reused","aos-fp32-setup","soa-fp32-reused","soa-fp32-setup","aos-fp16-reused","aos-fp16-setup","consumer-only"]]
        executable+=[f"matrix.euclidean2.{d}.N{n}.consumer-only" for n in [2,3]]
        missing=[f"batch.euclidean2.{d}.N{n}.soa-fp16-{v}" for n in [2,3] for v in ["reused","setup"]]
        value=run(f"boundary-{d}",["--suites","boundary","--dims",str(d),"--batch-ns","2,3"],executable,
                  scopes=("reused-input","setup-inclusive","consumer-only"),unavailable=missing,repeats=256)
        assert not value.get("abComparisons") and not value["abMissingSides"]
        assert sorted(entry["name"] for entry in value["unavailable"])==sorted(missing)
        assert all("sqrt" in entry["reason"] and entry["boundary"]["route"].startswith("unavailable: ") for entry in value["unavailable"])
        # Nothing is constructed or prepared for an entry that is never invoked (checked on the real CLI output).
        assert all(entry["boundary"]["inclusions"]["inputConstruction"]=="notApplicable" and entry["boundary"]["inclusions"]["inputPreparation"]=="notApplicable"
                   and any("unavailable; never invoked" in note for note in entry["boundary"]["notes"]) for entry in value["unavailable"])
        assert value["metadata"]["harnessCalibration"].startswith("not used (fixed --repeats 256")
        for case in value["results"]:
            assert case["correctness"]["status"]=="valid" and case["correctness"]["samples"]>0,case["name"]
            assert case["boundary"]["consumer"]=="u64-bitfold-sink-v1" and case["boundary"].get("comparison") is None,case["name"]
            if "fp16" in case["name"]: assert case["correctness"]["maxAbsError"]>0,case["name"]
    # Auto-scaling (no --repeats) copes with sub-tick consumer-only bodies: the run that failed nondeterministically before the
    # geometric calibration. Run three times; each must succeed and record the calibration rule. --min-time 0.005 keeps each child inside the 15 s bound
    # (12 cases x 9 samples at the default 0.2 s would not); the unmodified-default run is recorded separately in the report evidence.
    for attempt in range(3):
        folder=out/f"boundary-consumer-auto-{attempt}";folder.mkdir();report=folder/"report.json"
        helper.run([binary,"--suites","boundary","--dims","512,768,1536","--batch-ns","3,17","--filter","*consumer-only","--samples","9","--min-time","0.005","--format","json","--out",report],folder,"run",expected=0)
        auto=json.loads(report.read_text());assert len(auto["results"])==12,len(auto["results"])
        assert auto["metadata"]["harnessCalibration"].startswith("V04-calibration-2") and "clock tick" in auto["metadata"]["harnessCalibration"]
        assert all(len(c["sampleNsPerInvocation"])==9 and min(c["sampleIterations"])>1 for c in auto["results"])
        assert not any("Unusable benchmark clock interval" in f["reason"] for f in auto["failures"])
        rows.append({"name":f"boundary-consumer-auto-{attempt}","exit":0,"executableSHA256":helper.sha(binary)})
    run("boundary-only-unavailable",["--suites","boundary","--dims","512","--batch-ns","2","--filter","*soa-fp16*"],exit_code=1)
    run("boundary-excluded-never-run",["--suites","boundary","--dims","512","--batch-ns","2","--filter","*.aos-fp32-reused"],["batch.euclidean2.512.N2.aos-fp32-reused"],repeats=256)
    run("memory-copy-only",["--suites","memory","--dims","1","--filter","mem.copy.1"],["mem.copy.1"])
    run("memory-exclude",["--suites","memory","--dims","1","--exclude","mem.copy.*,mem.fill.*"],["mem.alloc.aligned.1","mem.pool.acquire.1"])
    run("noop-regex",["--suites","noop","--dims",str(sys.maxsize),"--filter-mode","regex","--filter","scalar"],["noop.scalar-add"])
    run("empty",["--suites","noop","--filter","no-match"],exit_code=1)
    run("bad-regex",["--suites","noop","--filter-mode","regex","--filter","["],exit_code=1)
    for option in ["--use-underscored","--no-use-underscored","--release-size","--no-release-size"]:
        run("unsupported-"+option[2:],["--suites","noop",option],exit_code=1)
    helper.write_json(out/"result.json",{"marker":"V02_SELECTION_COMPLETE","cases":rows})
    print("V02_SELECTION_COMPLETE",len(rows))


if __name__=="__main__":main()
