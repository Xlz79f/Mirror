# 镜 · 情感对话助手

情感对话助手。导入聊天记录 → AI 生成画像 → 画像辅助对话。

## 使用方法

### 需要 Python 3.10+

双击 `启动.bat`，首次运行会自动安装依赖（3-5 分钟），之后秒开。

### 或者手动

```bash
pip install -r requirements.txt
python jing_demo.py
```

## 功能

- 导入 WeFlow Excel 或 TXT 聊天记录
- AI 自动分析生成关系画像（戈特曼 + EFT 框架）
- 对话时语义搜索相关历史
- 多服务商：DeepSeek / 硅基流动 / Ollama 本地
- API Key 各预设独立保存

## 服务商

| 服务商 | 默认模型 | 费用 |
|--------|---------|------|
| DeepSeek | deepseek-v4-flash | ¥1/百万 token |
| 硅基流动 | Qwen/Qwen3-8B | 新用户送额度 |
| Ollama | qwen3:8b | 免费（需本地 GPU） |

## 隐私

所有数据本地存储，但为了更准确更有效地了解和帮助用户，在回答问题时会将用户画像和部分辅助数据一起发送给LLM，这些数据可能会被供应商看见。
