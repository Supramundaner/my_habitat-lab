# ReasonNavi package

`reason_navi` 是仓库内唯一受支持的自研导航包，对应原始 two-stage ObjectNav 主线。命名按责任划分，不再使用 `src`、`eval` 或数字 step 作为模块名。

## 包边界

```text
reason_navi/
├── preprocessing/   # episode → action.json + map artifacts
├── navigation/      # action.json → closed-loop navigation
├── evaluation/      # episode/batch orchestration + SR/SPL
├── vlm/             # loopback-only detector services and clients
├── configs/         # online navigation config
└── __main__.py      # python3 -m reason_navi
```

- `preprocessing.pipeline.PreprocessingPipeline` 是离线阶段的唯一编排入口。
- `navigation.runner.run_navigation` 是在线导航的唯一 application service；CLI 和 evaluator 都调用它。
- `navigation.controller.NavigationController` 负责导航状态与控制流程。
- `evaluation.episode.EpisodeEvaluator` 和 `evaluation.batch.BatchEvaluator` 负责评测编排，不复制 runtime 装配逻辑。

## 稳定入口

从仓库根目录运行：

```bash
# 离线 two-stage 目标选择
python3 -m reason_navi.preprocessing \
  reason_navi/preprocessing/config.example.json

# 仅运行在线导航
python3 -m reason_navi \
  --config reason_navi/configs/default_config.json \
  --request /absolute/path/to/action.json

# 单 episode / batch
python3 -m reason_navi.evaluation.episode \
  reason_navi/evaluation/episode.example.json
python3 -m reason_navi.evaluation.batch \
  reason_navi/evaluation/batch.example.json
```

在 Python 中嵌入 runtime：

```python
from pathlib import Path

from reason_navi.navigation.config import (
    load_json_object,
    load_navigation_config,
)
from reason_navi.navigation.runner import run_navigation

config = load_navigation_config("reason_navi/configs/default_config.json")
request = load_json_object("/absolute/path/to/action.json")
result = run_navigation(
    config,
    request,
    request_base_dir=Path("/absolute/path/to/artifact-directory"),
)
```

`run_navigation` 接受可注入的 `NavigationDependencies`，因此可以在不加载 Habitat-Sim/GPU 的测试中替换 simulator、occupancy map、video sink 和 controller。

## 导航请求契约

`action.json` 这个文件名为已有产物契约，暂不随源码重命名：

```json
{
  "agent_state": {
    "position": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0]
  },
  "target_info": {
    "coordinate": [1.0, 2.0],
    "name": "chair"
  },
  "wall_mask": "wall_mask.png",
  "map_metadata": "metadata.json"
}
```

- 坐标使用 Habitat world `[x, z]`。
- 四元数使用 `[x, y, z, w]`，并必须可归一化。
- `wall_mask` 和 `map_metadata` 默认相对 `action.json` 解析，整个产物目录可以在本地与 `/efs/anbangwang` 之间复制。
- runtime 会比对离线/在线投影的 scene、floor、bounds 和 spacing，不匹配时直接失败。
- 旧 `action[]`/`sequence` 格式不再声称兼容，因为原 controller 并未正确执行它们。

## 配置约定

- JSON 中的相对路径默认相对该 JSON 所在目录解析。
- `${HABITAT_DATA_ROOT}` 等占位符在加载时展开；缺失的环境变量是配置错误。
- LLM 凭据通过 `llm_config.api_key_env` 指向环境变量。
- `llm_config.fallback_on_failure` 默认为 `false`；不会在 LLM 失败后静默选择第一个房间或节点。
- `graph_generation.seed` 控制导航节点采样的可重复性。

更详细的运行说明见仓库根目录 [`README.md`](../README.md)。
