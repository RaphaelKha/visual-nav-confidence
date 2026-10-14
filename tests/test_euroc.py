"""EuRoC loader tests on a tiny synthetic ASL-format sequence (no dataset needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from euroc import interp_groundtruth, load_euroc  # noqa: E402

T0 = 1_403_636_579_758_555_392  # realistic epoch in ns


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(",".join(str(x) for x in r) for r in rows) + "\n")


def make_sequence(root):
    mav = Path(root) / "mav0"
    imu_ns = [T0 + i * 5_000_000 for i in range(201)]            # 200 Hz, 1 s
    _write(mav / "imu0" / "data.csv", "#imu", [[n, 0.1, 0.2, 0.3, 0, 0, 9.81] for n in imu_ns])
    # ground truth from 0.1 s to 0.9 s: p = (t, 2t, 0), v = (1, 2, 0), yaw rotating at 1 rad/s about z
    gt_rows = []
    for i in range(20, 181):
        t = i * 0.005
        h = t / 2
        gt_rows.append([T0 + i * 5_000_000, t, 2 * t, 0, np.cos(h), 0, 0, np.sin(h), 1, 2, 0,
                        0, 0, 0, 0.1, 0.2, 0.3])
    _write(mav / "state_groundtruth_estimate0" / "data.csv", "#gt", gt_rows)
    _write(mav / "cam0" / "data.csv", "#cam", [[T0 + i * 50_000_000, f"{i}.png"] for i in range(21)])
    return root


def test_load_shapes_and_time_origin(tmp_path):
    d = load_euroc(make_sequence(tmp_path))
    assert d.imu.t[0] == 0.0 and np.isclose(d.imu.t[-1], 1.0, atol=1e-12)
    assert d.imu.accel.shape == (201, 3) and d.imu.gyro.shape == (201, 3)
    assert np.isclose(d.imu.accel[0, 2], 9.81)
    assert np.isclose(d.gt.t[0], 0.1) and np.isclose(d.gt.t[-1], 0.9)
    assert d.gt.p.shape == (161, 3) and d.gt.q.shape == (161, 4)
    assert np.allclose(d.gt.b_a[0], [0.1, 0.2, 0.3])
    assert len(d.cam_t) == 21 and np.isclose(np.diff(d.cam_t).mean(), 0.05)


def test_interpolation_and_validity(tmp_path):
    d = load_euroc(make_sequence(tmp_path))
    t = np.array([0.05, 0.1234, 0.5, 0.9, 0.95])
    out, valid = interp_groundtruth(d.gt, t)
    assert valid.tolist() == [False, True, True, True, False]
    assert np.allclose(out["p"][valid], np.stack([t[valid], 2 * t[valid], 0 * t[valid]], axis=1), atol=1e-12)
    assert np.allclose(out["v"][valid], [1, 2, 0])
    yaw_half = t[valid] / 2  # slerp between samples 5 ms apart: exact up to the chord error
    expected = np.stack([np.cos(yaw_half), 0 * yaw_half, 0 * yaw_half, np.sin(yaw_half)], axis=1)
    assert np.allclose(out["q"][valid], expected, atol=1e-6)
    assert np.allclose(np.linalg.norm(out["q"], axis=1), 1.0)
