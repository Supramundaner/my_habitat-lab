# ObjectNav evaluation

`reason_navi.evaluation` 把预处理、在线导航和指标计算组装成单 episode 或 batch 评测：

```text
episode JSON + scene
        → two-stage preprocessing
        → shared navigation runner
        → final geodesic distance
        → SR / SPL / execution artifacts
```

Evaluator 调用共享的 `reason_navi.navigation.runner.run_navigation`，不包含第二份 runtime 装配逻辑。

## 环境

```bash
export HABITAT_DATA_ROOT=/absolute/path/to/habitat-data
export ARK_API_KEY='...'
```

`HABITAT_DATA_ROOT` 下应包含 `datasets/`、`versioned_data/` 和 `robots/`。配置中的 `${HABITAT_DATA_ROOT}` 会在加载时展开；环境变量缺失时会在评测开始前报错。

## 单 episode

复制 [`episode.example.json`](episode.example.json)，至少检查：

- `episode.episode_json_path`
- `episode.episode_id`
- `scene.scene_file`
- `scene.robot_urdf`
- `preprocess.llm_config`
- `video_generation.object_detection`
- `output_dir`

运行：

```bash
python3 -m reason_navi.evaluation.episode \
  reason_navi/evaluation/episode.example.json
```

加 `--verbose` 可显示 setup failure 的 traceback。

## Batch

复制 [`batch.example.json`](batch.example.json) 后运行：

```bash
python3 -m reason_navi.evaluation.batch \
  reason_navi/evaluation/batch.example.json
```

`evaluation_tasks` 必须是非空列表，每项包含 dataset path、scene path 和非空 `episode_ids` 列表。Episode ID 不得包含路径分隔符或 `..`，避免输出逃逸 batch 目录。单条失败会写入 partial summary，不会丢掉之前已完成的结果。

## 生成 batch 配置

三个 sampler 都支持 `--data-root`、`--scene-root`、`--robot-urdf`、`--output` 和 `--seed`：

```bash
python3 -m reason_navi.evaluation.sample_episodes \
  --seed 42 \
  --output reason_navi/evaluation/batch.generated.json

python3 -m reason_navi.evaluation.sample_challenge_episodes \
  --seed 42 \
  --output reason_navi/evaluation/challenge-batch.generated.json

python3 -m reason_navi.evaluation.sample_challenge_scenes \
  --seed 42 \
  --output reason_navi/evaluation/challenge-scenes.generated.json
```

使用 `HABITAT_DATA_ROOT` 的默认路径时，生成的 JSON 保留 `${HABITAT_DATA_ROOT}` 占位符，便于迁移。显式 CLI 相对路径会被按输出 JSON 的位置正确重写，不依赖后续运行时的 cwd。

## 配置分区

| 字段 | 用途 |
| --- | --- |
| `episode` | dataset JSON 和 episode ID（单 episode） |
| `evaluation_tasks` | dataset/scene/episode ID 组（batch） |
| `scene` | scene asset 与 robot URDF |
| `preprocess` | top-down、房间分割、导航图和 LLM 设置 |
| `video_generation` | simulator、agent、occupancy map、VFH* 和 detector 设置 |
| `evaluation` | success distance threshold |
| `output_dir` | 评测产物根目录 |

JSON 的相对路径相对该 JSON 所在目录解析。Evaluator 会把默认 prompt 注入生成的 preprocessing config，所以 batch 配置无需复制 prompt 路径。

`preprocess` 和 `video_generation` 是已发布 JSON 配置的兼容字段；源码中的对应概念统一称为 `preprocessing` 和 `navigation`。后续如需替换 schema，应通过显式版本化 adapter 迁移，不应静默改变现有 JSON。

## 输出

单 episode 的输出结构：

```text
output.json
preprocess_config.json
video_config.json
preprocess/
├── action.json
├── wall_mask.png
└── metadata.json
video/
├── execution_report.json
└── output.mp4
```

`output.json` 包含阶段状态、错误、最终姿态、SR 和 SPL。不可达的 geodesic distance 写为 JSON `null` 并标记 `reachable: false`，不写非标准 `Infinity`。Batch 另外写入 `batch_output.json` 作为汇总。

`preprocess/` 是已存在的产物目录名，为了下游兼容保留；对应的 Python 包已统一命名为 `reason_navi.preprocessing`。

## 排查顺序

1. 检查 dataset episode 的 `scene_id` 与 `scene.scene_file` 是否一致。
2. 检查 scene 与 robot URDF 是否存在。
3. 检查 `ARK_API_KEY` 和 preprocessing `output.json`。
4. 检查 `action.json` 是否同时引用 wall mask 和 metadata。
5. 检查 detector 端口与 GPU device。
6. 先运行单 episode，成功后再运行 batch。

依赖、VLM 启动和测试命令见仓库根目录 [`README.md`](../../README.md)。
