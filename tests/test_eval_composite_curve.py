import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from eval_composite_curve import discover_steps, load_scored_rows, write_curve


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


def test_discover_steps_intersection(tmp_path):
    d = str(tmp_path)
    for p, steps in [(0, [50, 100, 150]), (1, [50, 100]), (2, [50, 100, 150])]:
        for k in steps:
            _touch(os.path.join(d, f"phase_{p}", f"step_{k}.pt"))
    # common to all 3 phases: 50, 100 (phase 1 lacks 150)
    assert discover_steps(d, 3) == [50, 100]


def test_discover_steps_empty_when_a_phase_has_none(tmp_path):
    d = str(tmp_path)
    _touch(os.path.join(d, "phase_0", "step_50.pt"))
    # phase_1 has no checkpoints -> no common steps
    assert discover_steps(d, 2) == []


def _write_curve(p, rows):
    json.dump({"student_dir": "x", "curve": rows}, open(p, "w"))


def test_load_scored_rows_missing_file(tmp_path):
    # No prior curve.json -> nothing to resume from.
    assert load_scored_rows(str(tmp_path / "curve.json")) == ([], set(), None)


def test_load_scored_rows_returns_fully_scored_steps(tmp_path):
    f = str(tmp_path / "curve.json")
    _write_curve(f, [{"step": 0, "FID": 12.3}, {"step": 2000, "FID": 9.1}])
    rows, done, cfg = load_scored_rows(f)
    assert done == {0, 2000}
    assert [r["step"] for r in rows] == [0, 2000]
    assert cfg is None  # legacy curve.json (no eval_config key) -> None, not an error


def test_load_scored_rows_drops_partial_rows_without_fid(tmp_path):
    # A row written before its FID was parsed must NOT count as done (recompute it).
    f = str(tmp_path / "curve.json")
    _write_curve(f, [{"step": 0, "FID": 12.3}, {"step": 2000}])
    rows, done, cfg = load_scored_rows(f)
    assert done == {0}
    assert [r["step"] for r in rows] == [0]


def test_load_scored_rows_corrupt_json_starts_clean(tmp_path):
    # A truncated write (killed mid-json.dump) must not crash resume.
    f = str(tmp_path / "curve.json")
    open(f, "w").write('{"curve": [{"step": 0, "FID":')
    assert load_scored_rows(f) == ([], set(), None)


def test_load_scored_rows_missing_file_writes_no_parse_error_marker(tmp_path):
    # Hardening item 3: a file that legitimately doesn't exist yet is normal
    # "no data yet" progress state, NOT corruption -- must not alert.
    f = str(tmp_path / "curve.json")
    load_scored_rows(f)
    assert not os.path.exists(f + ".PARSE_ERROR")


def test_load_scored_rows_corrupt_json_writes_parse_error_marker(tmp_path):
    # An EXISTING but corrupt curve.json must be distinguishable from "no rows
    # yet": a watcher that treats both the same way would silently report an
    # empty curve.
    f = str(tmp_path / "curve.json")
    open(f, "w").write('{"curve": [{"step": 0, "FID":')
    load_scored_rows(f)
    marker = f + ".PARSE_ERROR"
    assert os.path.exists(marker)
    assert "curve.json" in open(marker).read()


def test_load_scored_rows_wrong_shape_writes_parse_error_marker(tmp_path):
    # The exact historical bug shape: a dict WITHOUT a 'curve' key (e.g. the
    # old watch_ditb_bedroom_ext.sh bug assumed {'rows': [...]}).
    f = str(tmp_path / "curve.json")
    json.dump({"rows": [{"step": 0, "fid": 12.3}]}, open(f, "w"))
    rows, done, cfg = load_scored_rows(f)
    assert (rows, done, cfg) == ([], set(), None)
    assert os.path.exists(f + ".PARSE_ERROR")


def test_load_scored_rows_clears_stale_parse_error_marker_on_recovery(tmp_path):
    # Once a corrupt file is replaced by a healthy one, the marker must not
    # linger and falsely alert forever.
    f = str(tmp_path / "curve.json")
    open(f, "w").write('{"curve": [{"step": 0, "FID":')
    load_scored_rows(f)
    assert os.path.exists(f + ".PARSE_ERROR")
    _write_curve(f, [{"step": 0, "FID": 12.3}])
    load_scored_rows(f)
    assert not os.path.exists(f + ".PARSE_ERROR")


def test_write_curve_sorts_and_round_trips_through_resume(tmp_path):
    # write_curve persists sorted rows; load_scored_rows reads them back -> the
    # resume contract (write a point, requeue, skip it next run) holds end to end.
    f = str(tmp_path / "curve.json")
    write_curve(f, "sd", [{"step": 4000, "FID": 8.0}, {"step": 0, "FID": 12.0}])
    data = json.load(open(f))
    assert data["student_dir"] == "sd"
    assert [r["step"] for r in data["curve"]] == [0, 4000]  # sorted
    rows, done, cfg = load_scored_rows(f)
    assert done == {0, 4000}


def test_write_curve_leaves_no_tmp_file(tmp_path):
    # Atomic rename must not leave the .tmp staging file behind.
    f = str(tmp_path / "curve.json")
    write_curve(f, "sd", [{"step": 0, "FID": 1.0}])
    assert not os.path.exists(f + ".tmp")
