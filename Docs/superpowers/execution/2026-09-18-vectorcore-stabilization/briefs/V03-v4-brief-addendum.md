# V03 v4 brief addendum — root ruling, round 2 (binding)

Close the unbounded-subprocess class in CI Python. Scripts/ci/tests/test_child_process_bounds.py only inspected .wait()/.communicate(), so the legacy unbounded subprocess.run in test_check_required.py passed it.

Additional owned files: Scripts/ci/tests/test_child_process_bounds.py, Scripts/ci/consumer_smoke.py, Scripts/ci/tests/test_consumer_smoke.py, Scripts/ci/tests/test_hooks.py.

1. Factor a detector and add a test flagging any subprocess.run / call / check_call / check_output (attribute form on the `subprocess` module) without a timeout= keyword or with timeout=None, across every Scripts/ci/**/*.py. Add a negative-control test on synthetic sources: missing timeout and timeout=None must be flagged; a constant and a variable timeout must be accepted. Keep the existing wait/communicate rule unchanged.
2. Bound the six current violations, adding only the keyword: consumer_smoke.py:75 swift build release timeout=1800; :76 --show-bin-path check_output timeout=600; test_consumer_smoke.py:51 timeout=60; test_hooks.py:32/:37/:105 timeout=60.
3. Re-run full discovery and the bounds module, and show a mutation fails the new rule, with byte-identical restoration.
