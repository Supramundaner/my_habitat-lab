# Two-stage preprocessing

`reason_navi.preprocessing` 是 ReasonNavi 的离线目标选择部分。输入一个 Habitat ObjectNav episode 的场景、起始姿态和目标类别，输出可直接交给 `reason_navi.navigation` 的 `action.json` 及其地图产物。

## 处理阶段

`PreprocessingPipeline` 按以下顺序执行：

| 阶段 | 模块 | 主要产物 |
| --- | --- | --- |
| 全局投影 | `generate_topdown` | `topdown_view.png`, `metadata.json` |
| 墙体提取 | `generate_wall_mask` | `wall_mask.png` |
| 房间分割 | `segment_rooms` | `room_annotation.png`, room bounding boxes |
| LLM 选房间 | `select_room` | `room_selection_log.json` |
| 导航图采样 | `generate_navigation_graph` | `navigation_nodes.json`, graph image |
| LLM 选节点 | `select_navigation_node` | `node_selection_log.json` |
| 生成请求 | `build_navigation_request` | `action.json` |

两次语义决策分别是“目标最可能在哪个房间”和“该房间内哪个可导航节点最合适”。余下步骤是几何/图像处理。

## 配置

复制并修改 [`config.example.json`](config.example.json)。核心字段：

```json
{
  "path_resolution": {"relative_to": "config"},
  "resolution": 2048,
  "scene_config": {
    "scene_path": "${HABITAT_DATA_ROOT}/versioned_data/.../scene.basis.glb",
    "target_floor": null,
    "target_coordinate": [1.0, 0.0, 2.0],
    "goal_object": "chair",
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "target_coverage": 0.9
  },
  "graph_generation": {
    "pds_radius": 0.5,
    "max_attempts": 30,
    "seed": 0
  },
  "llm_config": {
    "api_key_env": "ARK_API_KEY",
    "max_retries": 3,
    "fallback_on_failure": false
  },
  "prompts": {
    "choose_room_prompt": "prompts/select_room.txt",
    "choose_node_prompt": "prompts/select_navigation_node.txt"
  },
  "output": {"output_dir": "output/example"}
}
```

- `relative_to: "config"` 使相对路径以配置文件所在目录为基准；`"repo"` 则以仓库根目录为基准。
- 未解析的 `${ENV_VAR}` 会立即报错，不会被当成普通路径。
- `target_floor: null` 表示根据 episode 起始高度选择楼层。
- `max_attempts` 是 Poisson-disk sampler 的尝试次数，不再由代码硬编码。
- `seed` 使导航图采样可复现。
- API key 只从 `api_key_env` 指定的环境变量读取。

```bash
export HABITAT_DATA_ROOT=/absolute/path/to/habitat-data
export ARK_API_KEY='...'
```

## 运行

所有命令从仓库根目录执行：

```bash
python3 -m reason_navi.preprocessing \
  reason_navi/preprocessing/config.example.json
```

批量从某个 ObjectNav dataset shard 采样：

```bash
python3 -m reason_navi.preprocessing.sample_artifacts \
  /absolute/path/to/content/SCENE.json \
  reason_navi/preprocessing/config.example.json \
  reason_navi/preprocessing/sample_output \
  10 \
  --scene-root /absolute/path/to/hm3d/split
```

如需检查某个产物边界，可以单独运行对应模块。例如：

```bash
python3 -m reason_navi.preprocessing.generate_topdown \
  reason_navi/preprocessing/config.example.json

python3 -m reason_navi.preprocessing.generate_wall_mask \
  reason_navi/preprocessing/output/example/topdown_view.png \
  reason_navi/preprocessing/output/example
```

## 产物契约

成功运行后的关键文件：

```text
output.json                 # 阶段状态、错误与产物索引
topdown_view.png            # 未标注的正射投影
metadata.json               # scene/floor/bounds/spacing 投影契约
wall_mask.png               # 离线墙体 mask
room_annotation.png         # 供 LLM 选房间的标注图
navigation_nodes.json       # 可导航候选节点
node_selection_log.json     # 最终节点与推理记录
action.json                 # 在线 runtime 输入
```

`action.json` 中的 `wall_mask` 和 `map_metadata` 使用相对路径，因此必须把整个产物目录一起移动。在线 runtime 会对比投影 metadata，防止把错误 scene/floor 的 wall mask 静默用于导航。

`action.json` 保留这个历史文件名是为了产物兼容；在源码中它被视为 navigation request。

## 依赖与测试

```bash
python3 -m pip install -r requirements-reason-navi.txt
python3 -m unittest discover -s test -p 'test_reason_navi_*.py'
```

Habitat-Sim 0.3.3 需单独安装。完整预处理还需 HM3D scene 和 BytePlus Ark SDK/凭据；轻量测试不使用场景、GPU 或网络。
