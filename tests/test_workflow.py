"""The aurora workflow is a self-restarting loop; these pin the parts that keep
the feed fresh (a schedule alone fired once in 12 hours, 2026-11-18)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(HERE, "..", ".github", "workflows")


def _read(name):
    with open(os.path.join(WF, name)) as f:
        return f.read()


def test_the_loop_rebuilds_every_ten_minutes_for_five_hours():
    y = _read("aurora.yml")
    assert "end=$(( $(date +%s) + 5 * 3600 ))" in y
    assert "left=$(( 600 - ($(date +%s) - t0) ))" in y
    assert "python run_aurora.py --out site" in y


def test_the_job_outlasts_the_loop():
    y = _read("aurora.yml")
    minutes = int(re.search(r"timeout-minutes: (\d+)", y).group(1))
    assert 5 * 60 + 10 <= minutes <= 360


def test_each_run_starts_the_next_even_when_it_failed():
    y = _read("aurora.yml")
    tail = y[y.index("- name: start the next run"):]
    assert "if: always()" in tail
    assert "gh workflow run aurora.yml --ref main" in tail
    assert "actions: write" in y


def test_a_run_that_died_in_setup_does_not_restart_at_once():
    tail = _read("aurora.yml")
    tail = tail[tail.index("- name: start the next run"):]
    assert tail.index("sleep") < tail.index("gh workflow run")


def test_only_one_run_at_a_time_and_none_is_cancelled():
    y = _read("aurora.yml")
    assert re.search(r"concurrency:\n  group: aurora\n  cancel-in-progress: false", y)


def test_a_run_that_published_nothing_ends_red():
    y = _read("aurora.yml")
    assert 'if [ "$published" = "0" ]; then echo "::error::nothing published in this run"; exit 1; fi' in y


def test_the_watchdog_dispatches_only_when_no_run_is_live():
    y = _read("aurora-watchdog.yml")
    assert 'select(.status != "completed")' in y
    assert 'if [ "$live" = "0" ]; then' in y
    assert "gh workflow run aurora.yml --ref main" in y
