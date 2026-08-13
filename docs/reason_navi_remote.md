# ReasonNavi deployment notes for `pn0:/efs/anbangwang`

The canonical two-stage pipeline can be developed locally, but a real episode
should run on a Linux GPU host. The local macOS environment does not currently
have Habitat-Sim, NumPy, Torch, or OpenCV. A read-only probe of `pn0` found:

- eight NVIDIA H200 GPUs;
- shared storage mounted at `/efs/anbangwang`;
- Habitat-Sim 0.3.3 in the existing `hssd_interaction` environment;
- no existing environment that also contains the full preprocessing, Torch,
  detector, and Ark SDK dependency set;
- no HM3D ObjectNav dataset or detector checkpoint discovered under the
  probed `/efs/anbangwang` paths.

Accordingly, the recommended layout is:

```text
/efs/anbangwang/reason_navi/
├── repo/                 # this checkout
├── data/                 # HM3D datasets/scenes and robot URDF
├── models/               # detector/SAM checkpoints
├── env/                  # dedicated Python 3.10 environment
└── runs/                 # generated preprocessing/navigation/evaluation artifacts
```

Set runtime paths through the environment, not checked-in JSON:

```bash
export HABITAT_DATA_ROOT=/efs/anbangwang/reason_navi/data
export ARK_API_KEY="..."
export GROUNDING_DINO_CONFIG=/efs/anbangwang/reason_navi/repo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export GROUNDING_DINO_WEIGHTS=/efs/anbangwang/reason_navi/models/groundingdino_swint_ogc.pth
export YOLOV7_ROOT=/efs/anbangwang/reason_navi/repo/yolov7
export YOLOV7_WEIGHTS=/efs/anbangwang/reason_navi/models/yolov7-e6e.pt
export MOBILE_SAM_CHECKPOINT=/efs/anbangwang/reason_navi/models/mobile_sam.pt
```

Detector services must remain loopback-only. Do not bind them to `0.0.0.0`,
and do not use a public tunnel. If a local client must inspect a remote service,
use an SSH local forward bound to `127.0.0.1`; normal evaluation is simpler when
the evaluator and detector services all run on `pn0`.

Before a full episode, run these gates on `pn0`:

```bash
python3 -m unittest discover -s test -p 'test_reason_navi_*.py'
python3 -m unittest reason_navi.vlm.tests.test_runtime_config

python3 -c 'import habitat_sim, cv2, numpy, scipy, sklearn, torch'
test -f "$HABITAT_DATA_ROOT/versioned_data/hm3d-0.2/hm3d/val/.../scene.basis.glb"
test -f "$HABITAT_DATA_ROOT/robots/hab_fetch/robots/hab_fetch.urdf"
```

Only after these checks pass should the three detector services and one
single-episode evaluation be launched:

```bash
cd /efs/anbangwang/reason_navi/repo
bash reason_navi/start_vlm_services.sh

python3 -m reason_navi.evaluation.episode \
  reason_navi/evaluation/episode.example.json
```

Run `reason_navi.evaluation.batch` only after the single-episode gate passes.
Keep generated artifacts under `/efs/anbangwang/reason_navi/runs`; use an
absolute `output_dir` in the copied evaluation config rather than editing
source paths.
