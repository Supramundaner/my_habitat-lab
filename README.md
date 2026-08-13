# ReasonNavi

ReasonNavi 是一套基于 Habitat-Sim 的零训练 two-stage ObjectNav 管线。它先在全局俯视图上用 LLM 完成“选房间→选导航点”，再使用 RGB-D 占据地图、A*、VFH* 和目标检测在仿真环境中闭环导航，最后计算 SR/SPL。

> 当前仓库只保留原始 two-stage ObjectNav 主线。One-stage、image/text goal、restart、model comparison、UniGoal 等实验快照已移除。Habitat-Lab 与 Habitat-Baselines 作为上游底座保留。

## 管线

```text
ObjectNav episode + HM3D scene
              │
              ▼
  preprocessing: top-down map → room segmentation
                 → LLM room selection → navigation graph
                 → LLM node selection
              │
              │ action.json + wall_mask.png + metadata.json
              ▼
  navigation: RGB-D mapping → A* global planning → VFH* avoidance
              → detector-guided target approach
              │
              ▼
  evaluation: execution report + video + SR/SPL
```

这是 Sense–Plan–Act 研究管线，不包含训练阶段。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| [`reason_navi/preprocessing`](reason_navi/preprocessing) | 离线俯视图、房间分割、LLM 选点与导航请求生成 |
| [`reason_navi/navigation`](reason_navi/navigation) | 仿真、RGB-D 建图、全局/局部规划、感知与控制 |
| [`reason_navi/evaluation`](reason_navi/evaluation) | 单 episode 与 batch 评测、SR/SPL 统计 |
| [`reason_navi/vlm`](reason_navi/vlm) | 仅绑定 `127.0.0.1` 的 GroundingDINO/YOLOv7/MobileSAM 服务 |
| [`test`](test) | 不依赖 Habitat-Sim/GPU/网络的轻量回归测试 |
| [`habitat-lab`](habitat-lab) | 上游 Habitat-Lab 0.3.3 |
| [`habitat-baselines`](habitat-baselines) | 上游训练 baseline，不在 ReasonNavi 运行链上 |

详细边界和 Python 入口见 [`reason_navi/README.md`](reason_navi/README.md)。

## 环境准备

建议在 Linux + NVIDIA GPU 上运行完整 episode。轻量测试可以在本地直接运行。

```bash
git submodule update --init --recursive

# 项目 Python 依赖（当前是环境清单，不是 lock file）
python3 -m pip install -r requirements-reason-navi.txt

# 需要 Habitat-Lab API 时
python3 -m pip install -e habitat-lab
```

Habitat-Sim 0.3.3 需单独安装，并与机器的 CUDA 环境匹配。Torch 也应选择相容的 CUDA build。不同机器的路径通过环境变量传入：

```bash
export HABITAT_DATA_ROOT=/absolute/path/to/habitat-data
export ARK_API_KEY='...'
```

`HABITAT_DATA_ROOT` 下应包含 `datasets/`、`versioned_data/` 和 `robots/`。不要把 API key 写入 JSON 或生成物。

## 快速开始

### 1. 生成 two-stage 预处理产物

编辑 [`reason_navi/preprocessing/config.example.json`](reason_navi/preprocessing/config.example.json) 后运行：

```bash
python3 -m reason_navi.preprocessing \
  reason_navi/preprocessing/config.example.json
```

默认示例会在 `reason_navi/preprocessing/output/example/` 生成 `action.json`、`wall_mask.png` 和 `metadata.json`。详细的阶段与配置见 [preprocessing README](reason_navi/preprocessing/README.md)。

### 2. 启动本地 VLM 服务

配置模型仓库与 checkpoint 的绝对路径后：

```bash
export CONDA_ENV_NAME=reason-navi
export GROUNDING_DINO_CONFIG=/absolute/path/to/GroundingDINO/config.py
export GROUNDING_DINO_WEIGHTS=/absolute/path/to/groundingdino.pth
export YOLOV7_ROOT=/absolute/path/to/yolov7
export YOLOV7_WEIGHTS=/absolute/path/to/yolov7.pt
export MOBILE_SAM_CHECKPOINT=/absolute/path/to/mobile_sam.pt

bash reason_navi/start_vlm_services.sh
```

服务只监听 loopback；不要使用公网 tunnel 暴露本地端口。

### 3. 运行在线导航

```bash
python3 -m reason_navi \
  --config reason_navi/configs/default_config.json \
  --request reason_navi/preprocessing/output/example/action.json
```

### 4. 评测

```bash
# 单 episode
python3 -m reason_navi.evaluation.episode \
  reason_navi/evaluation/episode.example.json

# batch
python3 -m reason_navi.evaluation.batch \
  reason_navi/evaluation/batch.example.json
```

评测配置、输出结构和数据采样工具见 [evaluation README](reason_navi/evaluation/README.md)。

## 测试

```bash
python3 -m unittest discover -s test -p 'test_reason_navi_*.py'
python3 -m unittest reason_navi.vlm.tests.test_runtime_config
```

这些测试不调用 Habitat-Sim、GPU、LLM 或 detector 服务。完整 E2E 需要 HM3D ObjectNav 数据、Habitat-Sim、robot URDF 和本地 VLM 服务。

## 远程部署

`pn0:/efs/anbangwang` 的建议目录、环境变量和运行顺序见 [`docs/reason_navi_remote.md`](docs/reason_navi_remote.md)。所有服务应与 evaluator 在同一台 GPU 主机上运行，或仅通过绑定 `127.0.0.1` 的 SSH local forwarding 访问。

## 安全提醒

当前配置只保存 `api_key_env` 环境变量名。如果历史 commit 中曾出现过真实凭据，仅修改当前文件不足以保证安全：相关凭据仍应立即吊销/轮换，Git 历史清理需作为单独、明确授权的操作进行。

## 上游与许可

本仓库基于 [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) 开发。上游 API 和安装方式请参考其官方文档，本仓库许可证见 [`LICENSE`](LICENSE)。场景数据与派生数据仍受各自的数据许可协议约束。
