#!/usr/bin/env python3
"""Independent comparator JSON tables and Python CSV parsing. Import-safe."""
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("v02_build_modes", ROOT / "Scripts/ci/check_benchmark_build_modes.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def metadata():
    return {"schemaVersion": 3, "fixtureVersion": "V02-inputs-1", "measurementBoundaryVersion": "V04-boundaries-1",
            "arch": "arm64", "deviceModel": "fixture-model", "deviceTag": "fixture-device", "os": "fixture-os",
            "runtimeToolchain": {"value": "fixture-swift"}, "coreConfiguration": "release", "benchmarkConfiguration": "release",
            "flags": {"useCKernels": False, "preferSoA": False, "useMixedPrecision": False}, "runSeed": 0,
            "samples": 2, "measurementMode": "fixed-repeats", "repeats": 1, "minTimeSeconds": 0}


def boundary():
    """A structurally complete V04 boundary; the comparator treats it as an opaque identity object."""
    return {"version": "V04-boundary-1", "timingScope": "reused-input", "operation": "fixture.operation",
            "route": "fixture route", "provider": "sequential", "workUnit": "candidate", "unitsPerInvocation": 1,
            "inputRepresentation": "aos-fp32", "outputRepresentation": "scores",
            "fixture": {"id": "fixture", "version": "V02-inputs-1", "dimension": 512, "count": 1, "seeds": {"query": "1"},
                        "values": "lcg", "preparation": "none"},
            "inclusions": {"inputConstruction": "excluded", "inputPreparation": "notApplicable", "routeInternalPacking": "notApplicable",
                           "callerOutputAllocation": "notApplicable", "routeInternalAllocation": "included", "providerScheduling": "included",
                           "providerObservation": "included", "kernelWork": "included", "outputConsumption": "included",
                           "correctnessValidation": "excluded"},
            "consumer": "scalar-sink-v1", "notes": ["fixture"]}


def case(name="fixture", median=25, units=1, with_boundary=True):
    value = {"name": name, "iterations": 2, "totalNS": 50, "nsPerOp": 25, "nsPerUnit": 25/units,
             "unitCount": units, "medianNsPerOp": median, "meanNsPerOp": median+1, "p90NsPerOp": median+2}
    if with_boundary:
        value["boundary"] = boundary()
        value["boundary"]["unitsPerInvocation"] = units
    return value


def run_report(rows=None):
    return {"metadata": metadata(), "results": rows if rows is not None else [case()], "failures": []}


def correctness_cases():
    valid = {"status": "valid", "samples": 2, "maxAbsError": 0, "meanAbsError": 0,
             "maxRelError": 0, "meanRelError": 0}
    rows = [("absent", run_report(), 0)]
    for label, value in [("null", None), ("valid", valid), ("string", "valid"),
                         ("array", []), ("bool", True), ("number", 1)]:
        report = run_report(); report["results"][0]["correctness"] = copy.deepcopy(value)
        rows.append((label, report, 0 if label in ["null", "valid"] else 2))
    for field, values in [("status", [None, True, 1, [], "invalid", "VALID"]),
                          ("samples", [None, True, False, 0, -1, 1.5, "2"]),
                          ("iterations", [None, True, False, 0, -1, 1.5, "2"])]:
        for index, value in enumerate(values):
            report = run_report(); report["results"][0]["correctness"] = copy.deepcopy(valid)
            target = report["results"][0] if field == "iterations" else report["results"][0]["correctness"]
            if value is None: del target[field]
            else: target[field] = value
            rows.append((field + "-" + str(index), report, 2))
    for field in ["maxAbsError", "meanAbsError", "maxRelError", "meanRelError"]:
        for index, value in enumerate([None, True, -1, "0"]):
            report = run_report(); report["results"][0]["correctness"] = copy.deepcopy(valid)
            if value is None: del report["results"][0]["correctness"][field]
            else: report["results"][0]["correctness"][field] = value
            rows.append((field + "-" + str(index), report, 2))
    return rows


def check_product_discovery(output):
    root = output / "product-discovery"
    root.mkdir()
    def rejects(path):
        try: helper.find_executable(path, "debug")
        except AssertionError: return
        raise AssertionError("missing or ambiguous product accepted")
    rejects(root)
    product = root / "arch/debug/vectorcore-bench"
    product.parent.mkdir(parents=True); product.write_bytes(b"product fixture")
    dwarf = root / "arch/debug/vectorcore-bench.dSYM/Contents/Resources/DWARF/vectorcore-bench"
    dwarf.parent.mkdir(parents=True); dwarf.write_bytes(b"debug symbols fixture")
    assert helper.find_executable(root, "debug") == product
    # The previous name-only discovery would incorrectly find two regular files.
    assert len([p for p in root.rglob("vectorcore-bench") if p.is_file()]) == 2
    duplicate = root / "other/debug/vectorcore-bench"
    duplicate.parent.mkdir(parents=True); duplicate.write_bytes(b"ambiguous product fixture")
    rejects(root)


def mutate(value, path):
    """Return a copy of a boundary with the leaf at the dotted path changed."""
    result = copy.deepcopy(value); target = result; parts = path.split(".")
    for part in parts[:-1]: target = target[part]
    old = target[parts[-1]]
    target[parts[-1]] = (not old) if isinstance(old, bool) else (old + 1) if isinstance(old, int) else (old + ["mutated"]) if isinstance(old, list) else str(old) + "~mutated"
    return result


def leaf_paths(value, prefix=""):
    for key, item in value.items():
        if isinstance(item, dict): yield from leaf_paths(item, prefix + key + ".")
        else: yield prefix + key


def boundary_table(check):
    """Per-case boundary identity: match, every per-field mismatch, legacy/missing, malformed."""
    check("boundary-match", run_report(), run_report(), 0)
    check("boundary-scope-mismatch", run_report(), scope_mismatch(), 2, (), ("Boundary mismatch", "timingScope"))
    check("boundary-scope-mismatch-not-waivable", run_report(), scope_mismatch(), 2, ("--allow-incompatible",), ("Boundary mismatch", "timingScope"))
    paths = list(leaf_paths(boundary()))
    assert len(paths) >= 25, paths
    for path in paths:
        after = run_report(); after["results"][0]["boundary"] = mutate(after["results"][0]["boundary"], path)
        # A present-but-different boundary is never waivable, with or without the explicit override.
        for flags in [(), ("--allow-incompatible",)]:
            check("boundary-field-" + path + ("-override" if flags else ""), run_report(), after, 2, flags, ("Boundary mismatch", path))
    # Legacy (schema 2) baseline without boundaries: metadata and boundary identity are waivable only together, explicitly.
    legacy = run_report([case(with_boundary=False)]); legacy["metadata"]["schemaVersion"] = 2; legacy["metadata"]["measurementBoundaryVersion"] = "V02-boundaries-1"
    check("boundary-legacy-missing", legacy, run_report(), 2)
    check("boundary-legacy-override", legacy, run_report(), 0, ("--allow-incompatible",),
          ("Boundary identity missing", "OVERRIDE --allow-incompatible: missing legacy boundary identity waived for 1 entries"))
    # Legacy means the key is absent or an integer below 3. A future, non-integer, sentinel-string, null or boolean schema is an
    # integrity error: no override waives a missing boundary there.
    for index, schema in enumerate([4, 99, "x", 3.5, "", "unknown", "Unavailable", "UNKNOWN", None, True, "3x"]):
        future = run_report([case(with_boundary=False)]); future["metadata"]["schemaVersion"] = schema
        for flags in [(), ("--allow-incompatible",)]:
            check("boundary-schema-%d" % index + ("-override" if flags else ""), future, run_report(), 2, flags)
            check("boundary-schema-current-%d" % index + ("-override" if flags else ""), run_report(), future, 2, flags)
        with_boundary = run_report(); with_boundary["metadata"]["schemaVersion"] = schema
        check("boundary-schema-present-%d" % index, with_boundary, run_report(), 2, ("--allow-incompatible",), ("unsupported schemaVersion",))
    # Nested structure and value types are validated, not only the top-level keys.
    nested = [("fixture-missing-key", lambda b: b["fixture"].pop("seeds")), ("fixture-count-string", lambda b: b["fixture"].__setitem__("count", "1")),
              ("fixture-seed-number", lambda b: b["fixture"]["seeds"].__setitem__("query", 1)), ("fixture-not-object", lambda b: b.__setitem__("fixture", [])),
              ("inclusion-missing", lambda b: b["inclusions"].pop("kernelWork")), ("inclusion-unknown-state", lambda b: b["inclusions"].__setitem__("kernelWork", "maybe")),
              ("inclusion-number", lambda b: b["inclusions"].__setitem__("kernelWork", 1)), ("inclusions-not-object", lambda b: b.__setitem__("inclusions", "x")),
              ("scope-unknown", lambda b: b.__setitem__("timingScope", "lunch")), ("units-zero", lambda b: b.__setitem__("unitsPerInvocation", 0)),
              ("units-bool", lambda b: b.__setitem__("unitsPerInvocation", True)), ("units-fraction", lambda b: b.__setitem__("unitsPerInvocation", 1.5)),
              ("consumer-empty", lambda b: b.__setitem__("consumer", "")), ("route-number", lambda b: b.__setitem__("route", 3))]
    for label, mutate_boundary in nested:
        bad = run_report(); mutate_boundary(bad["results"][0]["boundary"])
        for flags in [(), ("--allow-incompatible",)]:
            check("boundary-malformed-" + label + ("-override" if flags else ""), run_report(), bad, 2, flags, ("malformed",) if flags else ())
    check("boundary-unverified-accepted", run_report(), unverified_report(), 2, ("--allow-incompatible",), ("Boundary mismatch", "inclusions.kernelWork"))
    both = unverified_report()
    check("boundary-unverified-equal", both, unverified_report(), 0)
    # A schema-3 report must carry a valid boundary; the override does not apply.
    for label, mutate_result in [("absent", lambda r: r.pop("boundary")), ("null", lambda r: r.__setitem__("boundary", None)),
                                 ("string", lambda r: r.__setitem__("boundary", "x")), ("array", lambda r: r.__setitem__("boundary", [])),
                                 ("missing-keys", lambda r: r["boundary"].pop("fixture")), ("missing-consumer", lambda r: r["boundary"].pop("consumer"))]:
        bad = run_report(); mutate_result(bad["results"][0])
        for flags in [(), ("--allow-incompatible",)]:
            check("boundary-schema3-" + label + ("-override" if flags else ""), run_report(), bad, 2, flags)
            check("boundary-schema3-baseline-" + label + ("-override" if flags else ""), bad, run_report(), 2, flags)


def unverified_report():
    value = run_report(); value["results"][0]["boundary"]["inclusions"]["kernelWork"] = "unverified"
    return value


def scope_mismatch():
    after = run_report(); after["results"][0]["boundary"]["timingScope"] = "setup-inclusive"
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    binary = output / "compare-benchmarks"
    helper.run(["swiftc", "-module-cache-path", output / "module-cache", ROOT / "Scripts/compare_benchmarks.swift", "-o", binary], output, "compile", timeout=120)
    check_product_discovery(output)
    rows = [{"name": "product-discovery", "missingRejected": True, "dSYMExcluded": True, "ambiguousRejected": True}]
    def check(name, before, after, expected, flags=(), fragments=()):
        directory = output/name; directory.mkdir()
        a, b = directory/"before.json", directory/"after.json"
        helper.write_json(a,before); helper.write_json(b,after)
        text = helper.run([binary,"--baseline",a,"--current",b,*flags],directory,"compare",expected=expected)
        for fragment in fragments: assert fragment in text, (name,fragment,text)
        rows.append({"name":name,"expectedExit":expected,"actualExit":expected})
    check("median-denominator",run_report([case(median=20)]),run_report([case(median=25)]),1,fragments=("delta=25.0%",))
    check("improvement-denominator",run_report([case(median=25)]),run_report([case(median=20)]),0,("--show-improvements",),("delta=-20.0%",))
    check("per-unit",run_report([case(median=50,units=2)]),run_report([case(median=40,units=2)]),0,("--per-unit","--show-improvements"),("baseline=25.0 current=20.0",))
    for metric,field in [("median","medianNsPerOp"),("mean","meanNsPerOp"),("p90","p90NsPerOp")]:
        bad=run_report();del bad["results"][0][field]
        check("missing-"+metric,run_report(),bad,2,("--metric",metric,"--allow-incompatible"))
    bad=run_report();bad["metadata"]["arch"]="x86_64"
    check("incompatible-arch",run_report(),bad,2)
    check("incompatible-override",run_report(),bad,0,("--allow-incompatible",),("Metadata mismatch: arch", "OVERRIDE --allow-incompatible"))
    old=run_report();old["metadata"]={}
    check("legacy",old,old,2)
    check("legacy-explicit",old,old,0,("--allow-incompatible",),("schemaVersion", "runtimeToolchain.value"))
    boundary_table(check)
    for field in ["schemaVersion","fixtureVersion","measurementBoundaryVersion","arch","deviceModel","deviceTag","os","runtimeToolchain","coreConfiguration","benchmarkConfiguration","flags","runSeed","samples","measurementMode","repeats"]:
        bad=run_report();del bad["metadata"][field]
        check("identity-missing-"+field,run_report(),bad,2)
    for key in ["preferSoA","useMixedPrecision","useCKernels"]:
        bad=run_report();bad["metadata"]["flags"][key]=True
        check("identity-flag-"+key,run_report(),bad,2)
    before=run_report();after=run_report()
    before["metadata"].update(measurementMode="estimated-duration",minTimeSeconds=.1)
    after["metadata"].update(measurementMode="estimated-duration",minTimeSeconds=.2)
    check("estimated-time",before,after,2)
    after=run_report();after["metadata"].update(date="different",gitSHA="different",runLabel="different")
    check("runtime-label-observations",run_report(),after,0)
    check("duplicate",run_report(),run_report([case(),case()]),2,("--allow-incompatible",))
    check("boundary-units-differ",run_report(),run_report([case(units=2)]),2,("--allow-incompatible",),("Boundary mismatch","unitsPerInvocation"))
    # Same boundary, different reported unit count: the V02 unit-consistency check still applies independently.
    bad=run_report();bad["results"][0]["unitCount"]=2;bad["results"][0]["nsPerUnit"]=12.5
    check("units",run_report(),bad,2,("--allow-incompatible",))
    for field,value in [("medianNsPerOp",0),("medianNsPerOp",-1),("unitCount",0),("unitCount",1.5),("nsPerOp",float("nan")),("medianNsPerOp",float("inf"))]:
        bad=run_report();bad["results"][0][field]=value
        check("numeric-"+field+"-"+str(value),run_report(),bad,2,("--allow-incompatible",))
    bad=run_report();bad["failures"]=[{"name":"fixture","reason":"failure"}]
    check("failed-run",run_report(),bad,2,("--allow-incompatible",))
    bad=run_report();bad["results"][0]["correctness"]={"status":"invalid","samples":1}
    check("failed-correctness",run_report(),bad,2,("--allow-incompatible",))
    check("missing-strict",run_report([case(),case("extra")]),run_report(),1)
    check("missing-permitted",run_report([case(),case("extra")]),run_report(),0,("--allow-missing",),("extra", "OVERRIDE --allow-missing"))
    check("no-matches",run_report([case("left")]),run_report([case("right")]),2,("--allow-missing","--allow-incompatible"))
    check("invalid-threshold",run_report(),run_report(),2,("--threshold","nan"))
    for label, report, expected in correctness_cases():
        for override in [False, True]:
            check("correctness-" + label + ("-override" if override else ""), run_report(), report,
                  expected, ("--allow-incompatible",) if override else ())
    if args.csv:
        expected='comma,quote"\nline\rreturn'
        with args.csv.open(newline="") as stream: table=list(csv.reader(stream))
        header=next(i for i,row in enumerate(table) if row and row[0]=="name")
        meta={row[0]:row[1] for row in table[:header] if len(row)==2}
        for key in ["runLabel","deviceTag","deviceModel","swiftVersion","runtimeToolchain.value"]: assert meta[key]==expected,(key,meta)
        data=dict(zip(table[header],table[header+1]));assert data["name"]==expected
        assert float(data["ns_per_op"])==25 and float(data["ns_per_unit"])==5
        columns=table[header]
        v02_prefix=["name","kind","metric","dim","variant","provider","N","iterations","total_ns","ns_per_op","unit_count","ns_per_unit","throughput_per_s",
                    "gflops","samples","mean_ns_per_op","median_ns_per_op","p90_ns_per_op","stddev_ns_per_op","rsd_percent","suspicious"]
        assert len(v02_prefix)==21 and columns[:21]==v02_prefix,("V02 CSV prefix changed",columns[:21])
        for column in ["sample_ns_per_operation","median_ns_per_operation","mean_ns_per_operation","p90_ns_per_operation","stddev_ns_per_operation","rsd_percent_per_operation"]:
            assert column in columns,column
        assert data["statistic_unit"]=="per-sample-average-ns-per-operation"
        # fixture: samples 24 and 26 ns per invocation over 5 units => 4.8 and 5.2 ns per operation
        per_operation=[float(x) for x in data["sample_ns_per_operation"].split(";")]
        assert abs(per_operation[0]-4.8)<1e-9 and abs(per_operation[1]-5.2)<1e-9,per_operation
        assert abs(float(data["median_ns_per_operation"])-5.0)<1e-9 and abs(float(data["mean_ns_per_operation"])-5.0)<1e-9
        assert abs(float(data["p90_ns_per_operation"])-5.2)<1e-9 and abs(float(data["stddev_ns_per_operation"])-0.2*2**0.5)<1e-9
        for column in ["boundary_version","timing_scope","operation","route","provider_mode","work_unit","units_per_invocation","input_representation",
                       "output_representation","fixture_id","fixture_preparation","incl_input_construction","incl_input_preparation","incl_route_internal_packing",
                       "incl_caller_output_allocation","incl_route_internal_allocation","incl_provider_scheduling","incl_provider_observation","incl_kernel_work",
                       "incl_output_consumption","incl_correctness_validation","consumer","comparison_family","comparison_side","statistic_unit",
                       "sample_ns_per_invocation","sample_iterations","warmup_iterations","boundary_json"]:
            assert column in columns,(column,columns)
        assert columns.index("suspicious")<columns.index("boundary_version"),"boundary columns are appended after the V02 columns"
        assert data["timing_scope"]=="fixture-injected" and data["route"]==expected and data["operation"]=="op,"+expected
        assert data["units_per_invocation"]=="5"
        assert data["sample_ns_per_invocation"]=="24.0;26.0" and data["sample_iterations"]=="1;1" and data["warmup_iterations"]=="3"
        parsed=json.loads(data["boundary_json"])
        assert parsed["route"]==expected and parsed["unitsPerInvocation"]==5 and parsed["timingScope"]=="fixture-injected"
        unavailable=next(i for i,row in enumerate(table) if row[:2]==["unavailable_name","unavailable_reason"])
        assert table[unavailable+1][:4]==[expected,expected,"fixture-injected",expected]
        assert json.loads(table[unavailable+1][4])["route"]==expected
        failure=next(i for i,row in enumerate(table) if row==["failure_name","failure_reason","failure_boundary_json"])
        assert table[failure+1][:2]==[expected,expected] and table[failure+1][2]==""
        # A failure that carries a boundary writes it as canonical JSON in the third column.
        assert table[failure+2][:2]==[expected,expected] and json.loads(table[failure+2][2])["route"]==expected
        # Sections present: metadata key/values, measurements, unavailable, failures.
        rows.append({"name":"independent-python-csv","sections":4,"passed":True})
    helper.write_json(output/"result.json",{"marker":"V02_REPORTING_COMPLETE","cases":rows,"comparatorSHA256":helper.sha(binary)})
    print("V02_REPORTING_COMPLETE",len(rows))


if __name__=="__main__": main()
