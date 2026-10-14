"""Relocalisation sanity check on real images: a small map from part of MH_01 must localise the other MH_01
frames of that segment to within a few cm (easy case: same sequence, close keyframes)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from euroc import interp_groundtruth, load_groundtruth  # noqa: E402

DATA = ROOT / "data" / "MH_01_easy"
ZIP = Path.home() / "Downloads" / "machine_hall" / "MH_01_easy" / "MH_01_easy.zip"
cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(not (ZIP.exists() and (DATA / "mav0" / "cam1" / "sensor.yaml").exists()),
                                reason="EuRoC MH_01_easy zip or calibration not available")


def test_camera_calibration_parsed():
    from reloc import load_camera
    cam = load_camera(DATA / "mav0" / "cam0" / "sensor.yaml")
    assert np.allclose(cam.K[0, 0], 458.654) and cam.size == (752, 480)
    assert np.allclose(cam.T_BC[3], [0, 0, 0, 1]) and np.allclose(cam.T_BC[:3, :3] @ cam.T_BC[:3, :3].T, np.eye(3), atol=1e-6)


def test_relocalisation_on_same_sequence():
    from reloc import ImageSource, Relocalizer, build_map, load_camera
    cam0 = load_camera(DATA / "mav0" / "cam0" / "sensor.yaml")
    cam1 = load_camera(DATA / "mav0" / "cam1" / "sensor.yaml")
    gt, cam_ns, cam_t = load_groundtruth(DATA, 0)
    sub = slice(1400, 1800)
    m = build_map([(ZIP, gt, cam_ns[sub], cam_t[sub])], cam0, cam1, log=lambda *_: None)
    rl = Relocalizer(m, cam0)
    g, _ = interp_groundtruth(gt, cam_t)
    src = ImageSource(ZIP)
    errs = []
    for i in range(1405, 1800, 41):
        r = rl.locate(src.read("cam0", cam_ns[i]))
        assert r["ok"]
        errs.append(np.linalg.norm(r["p"] - g["p"][i]))
    assert np.max(errs) < 0.03, errs
