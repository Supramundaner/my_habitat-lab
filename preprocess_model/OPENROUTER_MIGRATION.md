# OpenRouter API 迁移说明

本次修改将`prerprocess_model`中的火山引擎API更换为OpenRouter API，使用Qwen2.5-VL 7B Instruct模型。

## 主要修改内容

### 1. 配置文件修改 (`input_config.json`)

```json
{
  "llm_config": {
    "api_key": "<OPENROUTER_API_KEY>",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "qwen/qwen-2.5-72b-instruct",
    "max_tokens": 35000,
    "site_url": "<YOUR_SITE_URL>",
    "site_name": "<YOUR_SITE_NAME>"
  }
}
```

**需要配置的参数：**
- `api_key`: 你的OpenRouter API密钥
- `site_url`: (可选) 你的网站URL，用于OpenRouter统计排名
- `site_name`: (可选) 你的网站名称，用于OpenRouter统计排名

### 2. 代码修改

#### Step 3 - LLM房间选择 (`step_3_llm_room_selection.py`)
- 将`byteplussdkarkruntime.Ark`替换为`openai.OpenAI`
- 修改API调用方式以使用OpenRouter
- 更新响应解析逻辑（OpenRouter使用标准OpenAI格式，没有单独的reasoning_content）

#### Step 5 - LLM节点选择 (`step_5_node_selection.py`)
- 同样替换API客户端
- 更新多图片API调用方式
- 修改响应解析逻辑

### 3. 依赖包更新

需要安装OpenAI包：
```bash
pip install openai
```

可以卸载旧的火山引擎包：
```bash
pip uninstall volcenginesdkarkruntime
```

## 使用方法

1. 获取OpenRouter API密钥：
   - 访问 https://openrouter.ai/
   - 注册账户并获取API密钥

2. 更新配置文件：
   - 在`input_config.json`中填入你的API密钥
   - 可选：填入网站信息用于统计

3. 确保安装了依赖：
   ```bash
   pip install openai
   ```

4. 运行工作流：
   ```bash
   python main_workflow.py
   ```

## 模型选择

当前配置使用的是`qwen/qwen-2.5-72b-instruct`模型。你也可以选择其他兼容的视觉模型：

- `qwen/qwen-2-vl-7b-instruct` (更便宜的选择)
- `openai/gpt-4o` (OpenAI的多模态模型)
- `anthropic/claude-3.5-sonnet` (Anthropic的视觉模型)

只需在配置文件中修改`model`字段即可。

## 注意事项

1. **成本考虑**: OpenRouter按使用量计费，请注意API调用成本
2. **速率限制**: 不同模型可能有不同的速率限制
3. **图片大小**: 确保上传的图片大小符合模型要求
4. **错误处理**: 代码包含重试逻辑，但仍需监控API调用状态

## 故障排除

### 常见错误

1. **API密钥错误**
   ```
   Error: Invalid API key
   ```
   解决方案：检查`input_config.json`中的API密钥是否正确

2. **模型不可用**
   ```
   Error: Model not found
   ```
   解决方案：检查模型名称是否正确，或更换为可用模型

3. **网络连接问题**
   ```
   Error: Connection timeout
   ```
   解决方案：检查网络连接，可能需要使用代理

### 调试模式

可以在代码中添加更详细的日志来调试问题：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

这将显示详细的API请求和响应信息。
