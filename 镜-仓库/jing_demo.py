"""
镜 · 情感对话助手
================
"""

import sys, os

# ── 必须在所有 import 之前，修复 PyInstaller 编码问题 ──
if sys.stdout is None:
    import io
    sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
if sys.stderr is None:
    import io
    sys.stderr = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
# 阻止 httpx/urllib 在 Windows 上走 ascii 编码路径
import codecs
if sys.getdefaultencoding() != 'utf-8':
    sys._base_executable = sys.executable  # 防 PyInstaller 内部重置

import sqlite3, json, datetime
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NO_PROXY"] = "hf-mirror.com,huggingface.co"

import httpx
# ── 修复 httpx 在嵌入式 Python 中的 ASCII 头编码问题 ──
import httpx._models
_orig = httpx._models._normalize_header_value
def _patched(value, encoding=None):
    return _orig(value, encoding or "utf-8")
httpx._models._normalize_header_value = _patched

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QScrollArea, QFrame, QListWidget,
    QListWidgetItem, QGraphicsDropShadowEffect, QMessageBox, QDialog,
    QFormLayout, QLineEdit, QFileDialog, QProgressDialog, QComboBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent, QSize
from PyQt6.QtGui import QFont, QColor

# ═════════════════════════════════════════════════════════
# 路径
# ═════════════════════════════════════════════════════════

APP_DIR = Path(__file__).parent
SETTINGS_PATH = APP_DIR / "settings.json"
DB_PATH = APP_DIR / "jing_data.db"
MEMORY_INDEX_PATH = APP_DIR / "memory_index"

# ═════════════════════════════════════════════════════════
# 配色
# ═════════════════════════════════════════════════════════

BG = "#EFEAE3"
CARD = "#FDF9F4"
BORDER = "#DDD0C0"
ACCENT = "#A89880"
TEXT = "#3D3226"
GRAY = "#9B8E80"
INPUT_BG = "#F5F1EC"
WHITE = "#FAF7F2"
SIDEBAR_BG = "#F8F4EE"
SIDEBAR_HOVER = "#F0EBE2"
SIDEBAR_SELECTED = "#EBE4D8"
DANGER = "#c0392b"


# ═════════════════════════════════════════════════════════
# 设置系统
# ═════════════════════════════════════════════════════════

DEFAULT_SETTINGS = {
    "api_key": "",
    "api_base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-v4-flash",
    "user_name": "我",
    "partner_name": "对方",
    "proxy": "",
    "preset_configs": {},  # {"deepseek": {url, model, key}, "siliconflow": {...}, ...}
}


def _detect_proxy() -> str:
    """自动检测系统代理环境变量。"""
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return ""


def load_settings() -> dict:
    """加载设置，文件不存在时返回默认值。"""
    if SETTINGS_PATH.exists():
        try:
            s = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            # 兼容旧版没有 proxy 字段的设置
            for k, v in DEFAULT_SETTINGS.items():
                if k not in s:
                    s[k] = v
            return s
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict):
    """保存设置到文件。"""
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    # 验证写入成功
    try:
        saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if saved.get("api_base_url") != settings.get("api_base_url"):
            print(f"[警告] 设置写入验证失败")
    except Exception:
        pass


def _get_httpx_kwargs(settings: dict) -> dict:
    """根据设置生成 httpx 客户端参数（含代理）。
    localhost 地址自动跳过代理。"""
    kwargs = {"timeout": 120, "trust_env": False}
    api_url = settings.get("api_base_url", "")
    # localhost/Ollama 不走代理
    if "localhost" in api_url or "127.0.0.1" in api_url or "0.0.0.0" in api_url:
        return kwargs
    proxy = (settings.get("proxy") or "").strip()
    if proxy:
        kwargs["proxy"] = proxy
    else:
        env_proxy = _detect_proxy()
        if env_proxy:
            kwargs["proxy"] = env_proxy
    return kwargs


# ═════════════════════════════════════════════════════════
# 本地向量索引（零 API 成本）
# ═════════════════════════════════════════════════════════

class MemoryIndex:
    """聊天记录语义搜索索引。嵌入和检索都在本地运行，不消耗 API token。"""

    def __init__(self, db_path: Path, chroma_path: Path):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self._model = None
        self._collection = None

    @property
    def is_ready(self) -> bool:
        return self.chroma_path.exists() and any(self.chroma_path.iterdir())

    def unload_model(self):
        """释放嵌入模型，回收 ~500MB 内存。搜索时会自动重新加载。"""
        self._model = None
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
        return self._model

    @property
    def collection(self):
        if self._collection is None and self.is_ready:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = client.get_or_create_collection(
                name="imported_messages",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def build(self, progress_callback=None) -> int:
        """从 imported_records 构建向量索引。返回索引条数。"""
        import chromadb

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, msg_time, sender, content FROM imported_records WHERE msg_type='文本消息' AND content != '' ORDER BY id"
        ).fetchall()
        conn.close()

        if not rows:
            return 0

        total = len(rows)
        batch_size = 256

        # 初始化 ChromaDB
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        collection = client.get_or_create_collection(
            name="imported_messages",
            metadata={"hnsw:space": "cosine"},
        )

        # 检查已索引的
        try:
            existing = set()
            result = collection.get(include=["metadatas"])
            if result["ids"]:
                for meta in result["metadatas"]:
                    if meta and "db_id" in meta:
                        existing.add(meta["db_id"])
        except Exception:
            existing = set()

        new_rows = [r for r in rows if r["id"] not in existing]
        if not new_rows:
            return len(rows)

        for i in range(0, len(new_rows), batch_size):
            batch = new_rows[i : i + batch_size]
            texts = [f"{r['sender']}: {r['content']}" for r in batch]
            embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=batch_size).tolist()
            ids = [f"msg_{r['id']}" for r in batch]
            metadatas = [
                {"db_id": r["id"], "time": str(r["msg_time"]), "sender": r["sender"], "content": r["content"]}
                for r in batch
            ]
            collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            if progress_callback:
                progress_callback(f"索引中… {min(i+batch_size, len(new_rows))}/{len(new_rows)}")

        self._collection = collection
        return len(rows)

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """语义搜索相关历史消息。返回 [{time, sender, content, score}, ...]"""
        if not self.is_ready:
            return []

        try:
            col = self.collection
            if col is None or col.count() == 0:
                return []

            query_emb = self.model.encode(query, show_progress_bar=False).tolist()
            results = col.query(query_embeddings=[query_emb], n_results=top_k, include=["metadatas", "distances"])

            hits = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    similarity = round(1 - min(distance, 1), 4)
                    hits.append({
                        "time": meta.get("time", ""),
                        "sender": meta.get("sender", ""),
                        "content": meta.get("content", ""),
                        "score": similarity,
                    })
            return hits
        except Exception:
            return []

    def format_for_llm(self, results: list[dict], max_chars: int = 2500) -> str:
        """格式化搜索结果，注入 system prompt。"""
        if not results:
            return ""

        lines = ["\n## 相关历史对话（从聊天记录中检索到的最相关片段）\n"]
        total = 0
        for i, r in enumerate(results[:12], 1):
            date = r["time"][:10] if r["time"] else ""
            line = f"[{date}] {r['sender']}: {r['content']}\n"
            total += len(line)
            if total > max_chars:
                lines.append(f"\n（共检索到 {len(results)} 条相关消息，已截断）")
                break
            lines.append(line)
        return "".join(lines)


# ═════════════════════════════════════════════════════════
# 数据库
# ═════════════════════════════════════════════════════════

def init_db():
    """初始化所有表：对话列表、对话消息（Database B）、导入记录（Database A）、画像版本。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT DEFAULT '新对话',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS imported_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_time TIMESTAMP,
            sender TEXT,
            content TEXT,
            msg_type TEXT DEFAULT '文本消息'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_imported_time ON imported_records(msg_time)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT,
            record_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# ── Database A 操作 ───────────────────────────────────────

def count_imported() -> int:
    conn = sqlite3.connect(str(DB_PATH))
    count = conn.execute("SELECT COUNT(*) FROM imported_records").fetchone()[0]
    conn.close()
    return count


def sample_imported_chunks(chunk_size: int = 400) -> list[list[dict]]:
    """将全部导入记录按时间顺序分块，每块 chunk_size 条。返回 [[{time,sender,content},...],...]"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM imported_records").fetchone()[0]
    if total == 0:
        conn.close()
        return []

    chunks = []
    offset = 0
    while offset < total:
        rows = conn.execute(
            "SELECT msg_time, sender, content FROM imported_records ORDER BY msg_time ASC LIMIT ? OFFSET ?",
            (chunk_size, offset),
        ).fetchall()
        if not rows:
            break
        chunks.append([
            {"time": r["msg_time"], "sender": r["sender"], "content": r["content"]}
            for r in rows
        ])
        offset += chunk_size

    conn.close()
    return chunks


def sample_imported(limit: int = 200) -> list[dict]:
    """保留兼容旧版调用。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM imported_records").fetchone()[0]
    if total == 0:
        conn.close()
        return []
    step = max(1, total // limit) if total > limit else 1
    rows = conn.execute(
        "SELECT msg_time, sender, content FROM imported_records WHERE rowid % ? = 0 ORDER BY msg_time ASC LIMIT ?",
        (step, limit),
    ).fetchall()
    conn.close()
    return [{"time": r["msg_time"], "sender": r["sender"], "content": r["content"]} for r in rows]


def clear_imported():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM imported_records")
    conn.commit()
    conn.close()


def batch_insert_imported(records: list[dict]):
    conn = sqlite3.connect(str(DB_PATH))
    conn.executemany(
        "INSERT INTO imported_records(msg_time, sender, content, msg_type) VALUES(?,?,?,?)",
        [(r["time"], r["sender"], r["content"], r.get("msg_type", "文本消息")) for r in records],
    )
    conn.commit()
    conn.close()


# ── Database B 操作 ───────────────────────────────────────

def get_convs() -> list:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT id, title FROM conversations ORDER BY id DESC").fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1]} for r in rows]


def create_conv(title="新对话") -> int:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute("INSERT INTO conversations(title) VALUES(?)", (title,))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def load_history(conv_id: int) -> list:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conv_id=? ORDER BY id ASC", (conv_id,)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def save_msg(conv_id: int, role: str, content: str):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT INTO messages(conv_id, role, content) VALUES(?,?,?)", (conv_id, role, content))
    if role == "user":
        title = content[:20].replace("\n", " ")
        title = (title + "…") if len(content) > 20 else title
        conn.execute("UPDATE conversations SET title=? WHERE id=? AND title='新对话'", (title, conv_id))
    conn.commit()
    conn.close()


def clear_conv_msgs(conv_id: int):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
    conn.commit()
    conn.close()


def delete_conv(conv_id: int):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.commit()
    conn.close()


def clear_all_db():
    """清空所有数据（对话 + 导入记录 + 画像 + 索引）。测试用。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM messages")
    conn.execute("DELETE FROM conversations")
    conn.execute("DELETE FROM imported_records")
    conn.execute("DELETE FROM profiles")
    conn.commit()
    conn.close()
    # 清空向量索引
    if MEMORY_INDEX_PATH.exists():
        import shutil
        shutil.rmtree(str(MEMORY_INDEX_PATH), ignore_errors=True)


# ── 画像操作 ──────────────────────────────────────────────

def save_profile(source: str, content: str, record_count: int = 0):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT INTO profiles(source, content, record_count, created_at) VALUES(?,?,?,?)",
        (source, content, record_count, datetime.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_all_profiles() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM profiles ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"source": r["source"], "content": r["content"], "created_at": r["created_at"]} for r in rows]


def get_latest_profile() -> str:
    profiles = get_all_profiles()
    return profiles[0]["content"] if profiles else ""


def build_weighted_profile_context() -> str:
    profiles = get_all_profiles()
    if not profiles:
        return ""
    parts = []
    for i, p in enumerate(profiles[:5]):
        if i == 0:
            label = "【最新画像 · 权重最高】"
        elif i == 1:
            label = "【上轮画像】"
        else:
            date_str = p["created_at"][:10] if p["created_at"] else ""
            label = f"【历史画像 · {date_str}】"
        src_label = {"import": "基于导入的聊天记录", "chat": "基于与镜的对话", "init": "初始画像"}.get(p["source"], p["source"])
        parts.append(f"{label}（{src_label}）\n{p['content']}")
    return "\n\n---\n\n".join(parts)


# ═════════════════════════════════════════════════════════
# 系统提示词
# ═════════════════════════════════════════════════════════

def build_system_prompt(settings: dict) -> str:
    user = settings["user_name"]
    partner = settings["partner_name"]
    profile_context = build_weighted_profile_context()

    base = f"""你是"镜"，一位情感对话分析师。你正在和 {user} 对话。{user} 的恋爱对象叫 {partner}。

## 你的角色
倾听 {user} 的诉说，从关系心理学的角度帮助 ta 理解自己的处境和情绪。说话温暖、敏锐、不站队。

## 你说话的方式
- 先接住情绪，再说看到了什么
- 用日常语言，不用学术术语
- 偶尔用一个巧妙的比喻
- 温暖但不谄媚

## 心理框架（戈特曼 × 苏·约翰逊融合视角）

### 核心理念
大多数冲突的底层不是谁对谁错，是两个依恋系统在互相发警报。你不是法官，你的工作是识别模式、翻译情感、提供修复路径。先共情，再分析。对双方都有同理心。

### 情绪翻译器（Sue Johnson）
每一句攻击/防御/冷漠的话，都是被加密的依恋求援信号：
  愤怒指责 → "我害怕被忽视"
  冷漠沉默 → "我怕表达也没用"
  嘲讽贬低 → "我感到不被尊重"
  控制追问 → "我感受不到我们的联结了"

### 四骑士识别（Gottman）
批评："你总是""你怎么又" → 解毒：I feel + I need 软启动
蔑视：阴阳怪气、"呵呵""行行行你说的都对" → 解毒：暂停，建安全感（最强离婚预测因子）
防御："不是我的问题""要不是你先…" → 解毒：承担哪怕5%的责任
筑墙：已读不回、只回表情包、离开对话 → 解毒：约定暂停信号+回归时间

### 魔鬼对话（Sue Johnson）
找坏人（攻击-攻击）：互相翻旧账、比谁更委屈 → 把模式本身当敌人
抗议波尔卡（追-逃）：一个狂发消息一个已读不回 → 追方说恐惧不说愤怒，逃方说需要不说防御
冻结逃离（逃-逃）：冷暴力、默契沉默 → 任一方表达脆弱"我想修复但不知道怎么开口"

### 情感银行账户（Gottman）
5:1黄金比例：1次负面互动需要5次正面互动来维持平衡。最大的取款不是吵架，是对联结邀约的反复拒绝。

### 生理阈值
心率>100bpm时前额叶下线。信号：消息频率暴涨、全是短句、用平时不用的激烈词汇。此时不要讲道理，建议暂停20分钟。

### 依恋恐慌解读
攻击不是攻击，是喊救命；沉默不是拒绝，是恐惧；控制不是霸道，是害怕失去联结。

### 修复优先于解决
69%的关系问题无法解决。关系质量不取决于问题消失，取决于修复尝试是否被接受。

### 中文网络用语映射
破防→原始痛点触发+防御 | 摆烂→筑墙/冻结 | CPU→感知操纵+防御 | 上头→生理唤起 | emo→原发情绪上浮 | 已读不回→筑墙/逃避 | 阴阳怪气→蔑视(最高警报) | u1s1→防御性前置 | 小丑→自嘲中的受伤感 | 急了→防御

### 分析输出格式
1. 冲突模式识别（1-2句话）
2. 关键对话分析（原文 → 骑士/魔鬼对话信号 → 表面情绪 → 底层需求）
3. 情绪翻译（2-3句关键发言的深层翻译）
4. 修复路径（具体的、可操作的建议）
5. 双向反馈（分别给两个人一段话，让各自感到被看见）

### 价值观
爱是能力不是运气。冲突是信号不是敌人。情感安全比"正确"重要。模式比事件重要。脆弱是勇气的最高形式。

### 不做的事
不站队、不帮一方论证"就是你的错"、不在情绪高涨时讲道理、不建议分手。

### 边界
- 你可以引用画像中的关系背景，但不能编造画像中没有的具体事件、日期或对话
- 不确定就说"我不确定"
- 分析时使用框架思维但用日常语言表达洞察"""

    if profile_context:
        base += f"""

## 关系画像
以下是基于聊天记录生成的关系画像，按时间加权排列（最新画像权重最高）。

{profile_context}

【画像结束。以上是你能引用的全部关系背景。不要编造任何画像中没有的具体事件或细节。】"""
    else:
        base += """

## 画像状态
暂无关系画像。你可以做一般性情感倾听。
（提示：点击左侧「导入聊天记录」按钮上传聊天记录，AI 将自动生成画像。）"""

    return base


# ═════════════════════════════════════════════════════════
# 导入工作线程（解析文件 + 写入数据库，不阻塞 UI）
# ═════════════════════════════════════════════════════════

class ImportWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(int)   # 导入条数
    error = pyqtSignal(str)

    def __init__(self, filepath: str, user_name: str, partner_name: str):
        super().__init__()
        self.filepath = filepath
        self.user_name = user_name
        self.partner_name = partner_name

    def run(self):
        try:
            filepath = Path(self.filepath)
            if filepath.suffix.lower() in (".xlsx", ".xls"):
                records = self._parse_excel(filepath)
            elif filepath.suffix.lower() in (".txt", ".csv"):
                records = self._parse_txt(filepath)
            else:
                self.error.emit(f"不支持的文件格式: {filepath.suffix}")
                return

            if not records:
                self.error.emit("未能从文件中读取到有效聊天记录。")
                return

            self.progress.emit(f"正在写入数据库（{len(records)} 条）…")
            batch_insert_imported(records)
            self.progress.emit(f"已写入 {len(records)} 条记录")
            self.finished.emit(len(records))
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _parse_excel(self, filepath: Path) -> list[dict]:
        import pandas as pd
        self.progress.emit("正在读取 Excel 文件…")
        df = pd.read_excel(filepath, header=None, dtype=str)
        data_start = 5
        df = df.iloc[data_start:].copy()
        df.columns = ["seq", "time", "nickname", "wxid", "remark", "identity", "msg_type", "content"]
        df = df.iloc[1:].copy()
        df = df[df["seq"].notna() & (df["seq"].str.strip() != "")]
        df = df[df["seq"] != "序号"].copy()

        records = []
        total_rows = len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 5000 == 0:
                self.progress.emit(f"解析中… {i}/{total_rows}")
            sender = self.user_name if str(row["identity"]).strip() == "我" else self.partner_name
            content = str(row["content"]).strip() if pd.notna(row["content"]) else ""
            if not content:
                continue
            records.append({
                "time": str(row["time"]) if pd.notna(row["time"]) else "",
                "sender": sender,
                "content": content,
                "msg_type": str(row["msg_type"]) if pd.notna(row["msg_type"]) else "文本消息",
            })

        records.sort(key=lambda r: r["time"])
        return records

    def _parse_txt(self, filepath: Path) -> list[dict]:
        self.progress.emit("正在读取文本文件…")
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 2)
                if len(parts) >= 3 and ":" in parts[2]:
                    time_str = f"{parts[0]} {parts[1]}"
                    rest = parts[2]
                    if ":" in rest:
                        sender, content = rest.split(":", 1)
                        records.append({
                            "time": time_str,
                            "sender": sender.strip(),
                            "content": content.strip(),
                            "msg_type": "文本消息",
                        })
        return records


# 画像生成工作线程（demo 方案：取全部文本记录 → LLM 一次生成画像）
# ═════════════════════════════════════════════════════════

class ProfileGenWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            user = self.settings["user_name"]
            partner = self.settings["partner_name"]
            kwargs = _get_httpx_kwargs(self.settings)
            kwargs["timeout"] = 300

            # 取全部文本记录
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT msg_time, sender, content FROM imported_records
                WHERE msg_type='文本消息' AND content != ''
                ORDER BY msg_time ASC
            """).fetchall()
            conn.close()

            total = len(rows)
            if total == 0:
                self.error.emit("没有可分析的聊天记录")
                return

            self.progress.emit(f"正在读取 {total:,} 条聊天记录…")

            # 组装对话文本
            lines = [f"以下是 {user} 和 {partner} 的聊天记录（共 {total:,} 条消息）：\n"]
            for r in rows:
                lines.append(f"[{r['msg_time']}] {r['sender']}: {r['content']}")

            records_text = "\n".join(lines)

            # 如果太长，均匀采样
            max_chars = 80000
            if len(records_text) > max_chars:
                step = len(rows) // (max_chars // 50)
                sampled_lines = [lines[0], ""]
                for i in range(0, len(rows), max(1, step)):
                    r = rows[i]
                    sampled_lines.append(f"[{r['msg_time']}] {r['sender']}: {r['content']}")
                records_text = "\n".join(sampled_lines)
                self.progress.emit(f"记录较长，已均匀采样约 {len(sampled_lines)-2} 条进行分析…")

            self.progress.emit("正在生成关系画像…")

            prompt = f"""你是"镜"，一位情感对话分析师。你正在和 {user} 对话。{user} 的恋爱对象叫 {partner}。

请根据以下 {user} 和 {partner} 的聊天记录，生成一份关系画像。

要求：
1. 分析双方的性格特点、沟通风格、依恋模式
2. 识别关系中的互动模式（冲突模式、联结模式、修复模式）
3. 标注关键事件和转折点
4. 用中文撰写，结构清晰，500-1000字
5. 使用双方昵称（{user}、{partner}）
6. 不要编造聊天记录中没有的细节

输出格式：
## 双方画像
- {user}：（性格、沟通风格、依恋倾向）
- {partner}：（性格、沟通风格、依恋倾向）

## 关系模式
（冲突模式、联结模式、修复模式）

## 关键观察
（3-5条基于聊天记录的具体观察）"""

            with httpx.Client(**kwargs) as client:
                resp = client.post(
                    f"{self.settings['api_base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.settings["model"],
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": records_text},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 4096,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                final_profile = data["choices"][0]["message"]["content"]
                self.progress.emit(f"画像生成完成（基于 {total:,} 条记录）")
                self.finished.emit(final_profile)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ═════════════════════════════════════════════════════════
# 流式对话线程
# ═════════════════════════════════════════════════════════

class StreamThread(QThread):
    text_token = pyqtSignal(str)
    stream_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, settings, system_content, history, user_msg):
        super().__init__()
        self.settings = settings
        self.sc = system_content
        self.hist = history
        self.msg = user_msg

    def run(self):
        collected = ""
        try:
            kwargs = _get_httpx_kwargs(self.settings)
            kwargs["timeout"] = 120

            messages = [{"role": "system", "content": self.sc}]
            for h in self.hist:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": self.msg})

            with httpx.Client(**kwargs) as client:
                with client.stream(
                    "POST",
                    f"{self.settings['api_base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.settings["model"],
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.7,
                        "max_tokens": 4096,
                    },
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            d = line[6:]
                            if d == "[DONE]":
                                break
                            try:
                                chunk = json.loads(d)
                                token = chunk["choices"][0]["delta"].get("content", "")
                                if token:
                                    collected += token
                                    self.text_token.emit(token)
                            except Exception:
                                pass
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"{e}\n{traceback.format_exc()}")
        self.stream_done.emit(collected)


# ═════════════════════════════════════════════════════════
# 设置对话框
# ═════════════════════════════════════════════════════════

class SettingsDialog(QDialog):
    """首次使用或手动打开设置。"""

    def __init__(self, current_settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ 设置")
        self.setMinimumWidth(440)
        self.setStyleSheet(f"QDialog{{background:{CARD};}}")
        self._preset_configs = dict(current_settings.get("preset_configs", {}))
        self._last_preset = None
        self._init_done = False

        layout = QFormLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(24, 20, 24, 20)

        title = QLabel("配置 API 和双方信息")
        title.setStyleSheet(f"font-size:14px;font-weight:700;color:{TEXT};")
        layout.addRow(title)

        # ── 预设选择 ────────────────────────────
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["DeepSeek（推荐）", "硅基流动", "Ollama（本地）", "自定义"])
        self.preset_combo.setStyleSheet(
            f"QComboBox{{background:{INPUT_BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:6px;padding:6px 10px;font-size:13px;}}"
            f"QComboBox:hover{{border-color:{ACCENT};}}"
            f"QComboBox::drop-down{{border:none;}}"
        )
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        layout.addRow("服务商:", self.preset_combo)

        self.api_key_input = QLineEdit(current_settings.get("api_key", ""))
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        self.api_key_input.setStyleSheet(self._input_style())
        layout.addRow("API Key:", self.api_key_input)

        self.api_url_input = QLineEdit(current_settings.get("api_base_url", "https://api.deepseek.com/v1"))
        self.api_url_input.setStyleSheet(self._input_style())
        layout.addRow("API 地址:", self.api_url_input)

        self.model_input = QLineEdit(current_settings.get("model", "deepseek-v4-flash"))
        self.model_input.setStyleSheet(self._input_style())
        layout.addRow("模型:", self.model_input)

        # 根据当前设置选中预设（静默设置，不触发覆盖）
        self.preset_combo.blockSignals(True)
        api_url = current_settings.get("api_base_url", "")
        if "localhost:11434" in api_url or "127.0.0.1:11434" in api_url:
            self.preset_combo.setCurrentIndex(2)   # Ollama
        elif "siliconflow" in api_url:
            self.preset_combo.setCurrentIndex(1)   # 硅基流动
        elif "deepseek" in api_url:
            self.preset_combo.setCurrentIndex(0)   # DeepSeek
        else:
            self.preset_combo.setCurrentIndex(3)   # 自定义
        self.preset_combo.blockSignals(False)
        # 标记初始预设
        preset_keys = {0: "deepseek", 1: "siliconflow", 2: "ollama", 3: "custom"}
        self._last_preset = preset_keys.get(self.preset_combo.currentIndex(), "deepseek")
        self._init_done = True
        # 应用初始预设的 UI 状态（不清除已有 Key）
        self._apply_preset_ui(self.preset_combo.currentIndex())

        self.user_input = QLineEdit(current_settings.get("user_name", "我"))
        self.user_input.setStyleSheet(self._input_style())
        layout.addRow("你的昵称:", self.user_input)

        self.partner_input = QLineEdit(current_settings.get("partner_name", "对方"))
        self.partner_input.setStyleSheet(self._input_style())
        layout.addRow("TA 的昵称:", self.partner_input)

        # 分隔线
        sep = QLabel("")
        sep.setStyleSheet(f"border-top:1px solid {BORDER};margin:4px 0;")
        layout.addRow(sep)

        opt_label = QLabel("以下为可选设置（通常不需要改）")
        opt_label.setStyleSheet(f"color:{GRAY};font-size:11px;background:transparent;")
        layout.addRow(opt_label)

        self.proxy_input = QLineEdit(current_settings.get("proxy", ""))
        detected = _detect_proxy()
        ph = f"例如 http://127.0.0.1:7890"
        if detected:
            ph = f"已检测到: {detected}（自动使用，无需填写）"
        self.proxy_input.setPlaceholderText(ph)
        self.proxy_input.setStyleSheet(self._input_style())
        layout.addRow("代理 (可选):", self.proxy_input)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("取消")
        btn_cancel.setStyleSheet(f"QPushButton{{background:transparent;color:{GRAY};border:1px solid {BORDER};border-radius:6px;padding:6px 20px;font-size:13px;}}QPushButton:hover{{background:{SIDEBAR_HOVER};}}")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_save = QPushButton("保存")
        btn_save.setStyleSheet(f"QPushButton{{background:{ACCENT};color:white;border:none;border-radius:6px;padding:6px 20px;font-size:13px;font-weight:600;}}QPushButton:hover{{background:#8B7D6D;}}")
        btn_save.clicked.connect(self._save_and_accept)
        btn_row.addWidget(btn_save)
        layout.addRow(btn_row)

    def _save_and_accept(self):
        """验证并保存，然后关闭对话框。"""
        api_key = self.api_key_input.text().strip()
        api_url = self.api_url_input.text().strip()
        if not api_url:
            QMessageBox.warning(self, "请填写 API 地址", "API 地址不能为空。")
            return
        if "localhost" not in api_url and "127.0.0.1" not in api_url and not api_key:
            reply = QMessageBox.question(self, "API Key 为空", "你没有填写 API Key。确定要保存吗？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.accept()

    def _on_preset_changed(self, index: int):
        """切换预设：保存当前配置 → 恢复新预设配置（URL/模型/Key 全部独立）。"""
        presets = {
            0: {"url": "https://api.deepseek.com/v1", "model": "deepseek-v4-flash", "key_readonly": False, "key_name": "deepseek"},
            1: {"url": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen3-8B", "key_readonly": False, "key_name": "siliconflow"},
            2: {"url": "http://localhost:11434/v1", "model": "qwen3:8b", "key_readonly": True, "key_name": "ollama"},
            3: {"url": "", "model": "", "key_readonly": False, "key_name": "custom"},
        }
        p = presets.get(index, presets[3])

        # 保存当前配置到旧预设
        if self._last_preset and self._init_done:
            self._preset_configs[self._last_preset] = {
                "url": self.api_url_input.text().strip(),
                "model": self.model_input.text().strip(),
                "key": self.api_key_input.text().strip(),
            }

        # 恢复新预设配置
        saved = self._preset_configs.get(p["key_name"], {})
        self._last_preset = p["key_name"]

        if saved:
            self.api_url_input.setText(saved.get("url", p["url"]))
            self.model_input.setText(saved.get("model", p["model"]))
            self.api_key_input.setText(saved.get("key", ""))
        else:
            self.api_url_input.setText(p["url"])
            self.model_input.setText(p["model"])

        # 本地模型特殊处理
        self.api_key_input.setReadOnly(p["key_readonly"])
        if p["key_readonly"]:
            self.api_key_input.setText("ollama")
            self.api_key_input.setStyleSheet(self._input_style() + "QLineEdit{color:#999;}")
        else:
            self.api_key_input.setStyleSheet(self._input_style())

    def _apply_preset_ui(self, index: int):
        """初始加载时只应用 UI 状态，不清除已有值。"""
        presets = {
            0: {"key_readonly": False},
            1: {"key_readonly": False},
            2: {"key_readonly": True},
            3: {"key_readonly": False},
        }
        p = presets.get(index, presets[3])
        if p["key_readonly"]:
            self.api_key_input.setReadOnly(True)
            self.api_key_input.setStyleSheet(self._input_style() + "QLineEdit{color:#999;}")
        else:
            self.api_key_input.setReadOnly(False)
            self.api_key_input.setStyleSheet(self._input_style())

    def _input_style(self):
        return f"QLineEdit{{background:{INPUT_BG};color:{TEXT};border:1px solid {BORDER};border-radius:6px;padding:6px 10px;font-size:13px;}}QLineEdit:focus{{border-color:{ACCENT};}}"

    def get_settings(self) -> dict:
        # 保存当前预设的配置
        if self._last_preset:
            self._preset_configs[self._last_preset] = {
                "url": self.api_url_input.text().strip(),
                "model": self.model_input.text().strip(),
                "key": self.api_key_input.text().strip(),
            }
        return {
            "api_key": self.api_key_input.text().strip(),
            "api_base_url": self.api_url_input.text().strip().rstrip("/"),
            "model": self.model_input.text().strip(),
            "user_name": self.user_input.text().strip() or "我",
            "partner_name": self.partner_input.text().strip() or "对方",
            "proxy": self.proxy_input.text().strip(),
            "preset_configs": self._preset_configs,
        }


# ═════════════════════════════════════════════════════════
# 聊天区
# ═════════════════════════════════════════════════════════

class ChatArea(QWidget):
    def __init__(self):
        super().__init__()
        self.assistant_label = None
        self._history = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            f"QScrollArea{{background:transparent;border:none;}}"
            f"QScrollBar:vertical{{width:4px;background:transparent;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER};border-radius:2px;}}"
        )
        self.msg_container = QWidget()
        self.msg_container.setStyleSheet("background:transparent;")
        self.msg_layout = QVBoxLayout(self.msg_container)
        self.msg_layout.setContentsMargins(12, 12, 12, 12)
        self.msg_layout.setSpacing(8)
        self.msg_layout.addStretch()
        self.scroll.setWidget(self.msg_container)
        layout.addWidget(self.scroll, 1)

        bottom = QWidget()
        bottom.setFixedHeight(54)
        bottom.setStyleSheet("background:transparent;")
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(10, 6, 10, 6)
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("想说点什么…（Enter 发送，Shift+Enter 换行）")
        self.input_box.setMaximumHeight(36)
        self.input_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.input_box.setStyleSheet(
            f"QTextEdit{{background:{INPUT_BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:10px;padding:5px 10px;font-size:13px;}}"
            f"QTextEdit:focus{{border-color:{ACCENT};}}"
        )
        bl.addWidget(self.input_box, 1)
        self.btn_send = QPushButton("→")
        self.btn_send.setFixedSize(30, 30)
        self.btn_send.setStyleSheet(
            f"background:{ACCENT};color:white;border:none;border-radius:15px;font-size:13px;"
        )
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        bl.addWidget(self.btn_send)
        layout.addWidget(bottom)

    def add_msg(self, text: str, is_user: bool):
        if self.msg_layout.count() > 0:
            item = self.msg_layout.itemAt(self.msg_layout.count() - 1)
            if item and item.spacerItem():
                self.msg_layout.removeItem(item)
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(4, 0, 4, 0)
        rl.setSpacing(4)
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(420)
        bubble.setTextFormat(Qt.TextFormat.PlainText)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble.setCursor(Qt.CursorShape.IBeamCursor)
        if is_user:
            rl.addStretch()
            bubble.setStyleSheet(
                f"background:#D4C9BC;color:{TEXT};padding:10px 14px;"
                f"border-radius:10px;font-size:14px;line-height:1.6;white-space:pre-wrap;"
            )
        else:
            bubble.setStyleSheet(
                f"background:{WHITE};color:{TEXT};padding:10px 14px;"
                f"border-radius:10px;font-size:14px;line-height:1.6;white-space:pre-wrap;"
            )
            self.assistant_label = bubble
        rl.addWidget(bubble)
        if not is_user:
            rl.addStretch()
        self.msg_layout.addWidget(row)
        self.msg_layout.addStretch()
        QTimer.singleShot(50, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    def append_msg(self, text: str):
        if self.assistant_label:
            self.assistant_label.setText(self.assistant_label.text() + text)
            QTimer.singleShot(30, lambda: self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().maximum()
            ))

    def clear_msgs(self):
        while self.msg_layout.count() > 1:
            item = self.msg_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self.assistant_label = None
        self._history = []

    def add_to_history(self, role: str, content: str):
        self._history.append({"role": role, "content": content})
        if len(self._history) > 30:
            self._history = self._history[-30:]


# ═════════════════════════════════════════════════════════
# 主窗口
# ═════════════════════════════════════════════════════════

class JingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("镜 · 情感对话助手")
        self.setGeometry(150, 100, 880, 700)
        self.setMinimumSize(600, 450)
        self.is_streaming = False
        self.current_conv_id = None
        self._loading = False
        self._import_worker = None
        self._profile_worker = None
        self._msg_since_update = 0
        self.memory = MemoryIndex(DB_PATH, MEMORY_INDEX_PATH)

        self.settings = load_settings()

        if not self.settings.get("api_key"):
            self._first_run_setup()

        init_db()
        self._build_ui()
        self._init_convs()

    # ── 首次使用设置 ────────────────────────────────
    def _first_run_setup(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.settings = dlg.get_settings()
            save_settings(self.settings)
        else:
            QMessageBox.information(self, "提示", "你可以稍后通过「设置」按钮配置 API Key。")

    # ── 对话初始化 ─────────────────────────────────
    def _init_convs(self):
        convs = get_convs()
        if not convs:
            self.current_conv_id = create_conv()
        else:
            self.current_conv_id = convs[0]["id"]
        self._refresh_sidebar()
        self._load_current_conv()

    def _refresh_sidebar(self):
        self._loading = True
        self.conv_list.clear()
        for c in get_convs():
            item = QListWidgetItem(c["title"])
            item.setData(Qt.ItemDataRole.UserRole, c["id"])
            item.setSizeHint(QSize(0, 36))
            self.conv_list.addItem(item)
            if c["id"] == self.current_conv_id:
                self.conv_list.setCurrentItem(item)
        self.import_count_label.setText(self._import_count_text())
        self.btn_clear_import.setVisible(count_imported() > 0)
        self._loading = False

    def _load_current_conv(self):
        self.chat_area.clear_msgs()
        for msg in load_history(self.current_conv_id):
            is_user = msg["role"] == "user"
            self.chat_area.add_msg(msg["content"], is_user)
            self.chat_area.add_to_history(msg["role"], msg["content"])
        if not self.chat_area._history:
            profile_exists = bool(get_latest_profile())
            greeting = "✨ 你好，我是镜。有什么想聊的？" if profile_exists else "✨ 你好，我是镜。你可以先导入聊天记录让我了解你们的关系，也可以直接和我聊聊。"
            self.chat_area.add_msg(greeting, False)

    def _switch_conv(self, current, previous):
        if self._loading:
            return
        item = self.conv_list.currentItem()
        if not item:
            return
        cid = item.data(Qt.ItemDataRole.UserRole)
        if cid and cid != self.current_conv_id:
            self.current_conv_id = cid
            self._load_current_conv()

    # ── 对话操作 ───────────────────────────────────
    def _new_conv(self):
        self.current_conv_id = create_conv()
        self._refresh_sidebar()
        self._load_current_conv()

    def _delete_conv(self):
        item = self.conv_list.currentItem()
        if not item:
            return
        cid = item.data(Qt.ItemDataRole.UserRole)
        if not cid:
            return
        reply = QMessageBox.question(self, "删除对话", f"确定要删除「{item.text()}」吗？")
        if reply != QMessageBox.StandardButton.Yes:
            return
        delete_conv(cid)
        convs = get_convs()
        if not convs:
            self.current_conv_id = create_conv()
        else:
            self.current_conv_id = convs[0]["id"]
        self._refresh_sidebar()
        self._load_current_conv()

    def _clear_conv(self):
        clear_conv_msgs(self.current_conv_id)
        self._load_current_conv()

    def _build_memory_index(self):
        """后台线程构建向量索引。"""
        class IndexBuildThread(QThread):
            progress = pyqtSignal(str)
            finished = pyqtSignal(int)
            error = pyqtSignal(str)
            def __init__(self, memory):
                super().__init__()
                self.memory = memory
            def run(self):
                try:
                    def cb(msg):
                        self.progress.emit(msg)
                    count = self.memory.build(progress_callback=cb)
                    self.finished.emit(count)
                except Exception as e:
                    import traceback
                    self.error.emit(f"{e}\n{traceback.format_exc()}")

        self._index_thread = IndexBuildThread(self.memory)
        self._index_thread.progress.connect(
            lambda m: self.status_label.setText(m)
        )
        self._index_thread.finished.connect(
            lambda c: (self.status_label.setText(f"✓ 索引已构建（{c:,} 条）"), self._preload_model())
        )
        self._index_thread.error.connect(
            lambda e: self.status_label.setText(f"索引构建失败（不影响使用）: {e[:50]}")
        )
        self._index_thread.start()

    def _preload_model(self):
        """后台预热嵌入模型。"""
        class PreloadThread(QThread):
            def __init__(self, memory):
                super().__init__()
                self.memory = memory
            def run(self):
                try:
                    _ = self.memory.model
                except Exception:
                    pass
        self._preload = PreloadThread(self.memory)
        self._preload.finished.connect(lambda: self.status_label.setText("就绪"))
        self._preload.start()

    def _clear_imported_records(self):
        """清空导入的聊天记录和关联画像。"""
        total = count_imported()
        if total == 0:
            return
        reply = QMessageBox.question(
            self, "清空导入记录",
            f"确定要清空全部 {total:,} 条导入记录吗？\n关联的关系画像也会被清除。"
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        clear_imported()
        # 同时清空画像和索引
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM profiles WHERE source='import'")
        conn.commit()
        conn.close()
        # 清空向量索引
        if MEMORY_INDEX_PATH.exists():
            import shutil
            shutil.rmtree(str(MEMORY_INDEX_PATH), ignore_errors=True)
        self.memory = MemoryIndex(DB_PATH, MEMORY_INDEX_PATH)
        self.import_count_label.setText(self._import_count_text())
        self.btn_clear_import.setVisible(False)
        self.status_label.setText("✓ 已清空导入记录")

    def _clear_all(self):
        reply = QMessageBox.question(self, "清空全部", "确定要删除所有对话、导入记录、画像和索引吗？此操作不可恢复。")
        if reply != QMessageBox.StandardButton.Yes:
            return
        clear_all_db()
        self.memory = MemoryIndex(DB_PATH, MEMORY_INDEX_PATH)
        self.current_conv_id = create_conv()
        self._refresh_sidebar()
        self._load_current_conv()
        self.btn_clear_import.setVisible(False)
        self.status_label.setText("✓ 已清空全部数据")

    # ── 导入聊天记录（后台线程，不卡 UI）───────────
    def _import_records(self):
        """选择文件 → 后台导入 → 弹出进度对话框。"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择聊天记录文件",
            "",
            "聊天记录 (*.xlsx *.xls *.txt *.csv);;Excel 文件 (*.xlsx *.xls);;文本文件 (*.txt *.csv)",
        )
        if not filepath:
            return

        # 进度对话框
        self._progress_dlg = QProgressDialog("准备导入…", "取消", 0, 0, self)
        self._progress_dlg.setWindowTitle("导入聊天记录")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.setValue(0)
        self._progress_dlg.canceled.connect(self._cancel_import)

        # 启动后台导入
        self._import_worker = ImportWorker(
            filepath, self.settings["user_name"], self.settings["partner_name"]
        )
        self._import_worker.progress.connect(self._on_import_progress)
        self._import_worker.finished.connect(self._on_import_finished)
        self._import_worker.error.connect(self._on_import_error)
        self._import_worker.start()

    def _cancel_import(self):
        if self._import_worker and self._import_worker.isRunning():
            self._import_worker.terminate()
        if self._profile_worker and self._profile_worker.isRunning():
            self._profile_worker.terminate()
        if hasattr(self, '_progress_dlg'):
            self._progress_dlg.close()

    def _on_import_progress(self, msg: str):
        if hasattr(self, '_progress_dlg') and self._progress_dlg.isVisible():
            self._progress_dlg.setLabelText(msg)

    def _on_import_finished(self, count: int):
        if hasattr(self, '_progress_dlg'):
            self._progress_dlg.setLabelText(f"已导入 {count} 条记录，正在构建搜索索引…")
        total = count_imported()
        self.import_count_label.setText(self._import_count_text())
        self.btn_clear_import.setVisible(total > 0)
        self.status_label.setText(f"已导入 {count} 条（共 {total} 条），构建索引…")

        # 后台构建向量索引
        self._build_memory_index()
        # 然后启动画像生成
        self._start_profile_generation()

    def _on_import_error(self, msg: str):
        if hasattr(self, '_progress_dlg'):
            self._progress_dlg.close()
        QMessageBox.critical(self, "导入失败", f"读取文件时出错：\n{msg}")
        self.status_label.setText(f"✗ 导入失败: {msg[:50]}")

    # ── 画像生成（分块分析全部记录）───────────────
    def _start_profile_generation(self):
        """启动分块画像生成。"""
        total = count_imported()
        if total == 0:
            if hasattr(self, '_progress_dlg'):
                self._progress_dlg.close()
            self.status_label.setText("没有可分析的数据")
            return

        self._profile_worker = ProfileGenWorker(self.settings)
        self._profile_worker.progress.connect(self._on_profile_progress)
        self._profile_worker.finished.connect(self._on_profile_finished)
        self._profile_worker.error.connect(self._on_profile_error)
        self._profile_worker.start()

    def _on_profile_progress(self, msg: str):
        if hasattr(self, '_progress_dlg') and self._progress_dlg.isVisible():
            self._progress_dlg.setLabelText(msg)
        self.status_label.setText(msg)

    def _on_profile_finished(self, content: str):
        total = count_imported()
        save_profile("import", content, total)
        if hasattr(self, '_progress_dlg'):
            self._progress_dlg.close()
        self.status_label.setText(f"✓ 画像已生成（基于 {total} 条导入记录）")
        self.import_count_label.setText(self._import_count_text())
        self.btn_clear_import.setVisible(True)
        self.chat_area.add_msg(f"📋 关系画像已生成。我分析了全部 {total} 条聊天记录，现在对你们的关系有了比较深入的了解。", False)
        QMessageBox.information(self, "画像生成完成", f"已基于全部 {total} 条聊天记录生成关系画像。\n你现在可以和镜聊聊了。")

    def _on_profile_error(self, msg: str):
        # 写入日志文件方便排查
        import traceback
        with open(APP_DIR / "error.log", "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now()}] 画像生成错误:\n{msg}\n{traceback.format_exc()}\n")
        if hasattr(self, '_progress_dlg'):
            self._progress_dlg.close()
        self.status_label.setText(f"✗ 画像生成失败: {msg[:50]}")
        QMessageBox.warning(self, "画像生成失败", f"生成关系画像时出错：\n{msg}\n\n画像未生成，但你仍可以正常聊天。")

    # ── 后台更新画像（累计足够对话后自动触发）─────
    def _update_profile_background(self):
        """后台线程：从最近对话更新画像。不阻塞 UI。"""
        existing_profile = get_latest_profile()
        user = self.settings["user_name"]
        partner = self.settings["partner_name"]

        class UpdateProfileThread(QThread):
            finished = pyqtSignal(str)
            def __init__(self, settings, existing, user_name, partner_name):
                super().__init__()
                self.settings = settings
                self.existing = existing
                self.user_name = user_name
                self.partner_name = partner_name
            def run(self):
                try:
                    conn = sqlite3.connect(str(DB_PATH))
                    rows = conn.execute(
                        "SELECT role, content FROM messages ORDER BY id DESC LIMIT 60"
                    ).fetchall()
                    conn.close()
                    if len(rows) < 10:
                        return
                    chat_text = "\n".join(
                        f"{'用户' if r[0] == 'user' else '镜'}: {r[1]}" for r in reversed(rows)
                    )
                    existing_part = f"现有画像：\n{self.existing}\n\n" if self.existing else ""
                    prompt = f"""你是关系心理学家。以下是已有的画像和新对话。请更新画像。

【重要规则】
- 以现有画像为基础，逐段审视哪些描述仍然准确、哪些需要修正
- 如果新对话显示用户的习惯、态度、互动模式发生了变化，在画像对应位置替换旧描述
- 没变化的部分原样保留，不要删减
- 输出完整的更新后画像，不要只输出变化的部分
- 格式与现有画像保持一致

{existing_part}
═══════════════════════════════════

最新对话：
{chat_text}

请输出更新后的完整画像。"""
                    kwargs = _get_httpx_kwargs(self.settings)
                    kwargs["timeout"] = 120
                    with httpx.Client(**kwargs) as client:
                        resp = client.post(
                            f"{self.settings['api_base_url']}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.settings['api_key']}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": self.settings["model"],
                                "messages": [
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": "请根据最新对话更新画像。"},
                                ],
                                "temperature": 0.7,
                                "max_tokens": 4096,
                            },
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            content = data["choices"][0]["message"]["content"]
                            save_profile("chat", content, len(rows))
                            self.finished.emit(content)
                except Exception:
                    import traceback
                    # 后台更新失败静默处理，写入日志
                    with open(APP_DIR / "error.log", "a", encoding="utf-8") as f:
                        f.write(f"\n[bg_update] {traceback.format_exc()}\n")

        self._profile_worker = UpdateProfileThread(
            self.settings, existing_profile, user, partner
        )
        self._profile_worker.finished.connect(self._on_bg_update_done)
        self._profile_worker.start()
        self.status_label.setText("后台更新画像…")

    def _on_bg_update_done(self, content: str):
        self._msg_since_update = 0
        self.status_label.setText("✓ 画像已更新")

    # ── 关闭事件 ───────────────────────────────────
    def closeEvent(self, event):
        event.accept()

    # ── 设置 ───────────────────────────────────────
    def _open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_settings = dlg.get_settings()
            self.settings = new_settings
            save_settings(self.settings)
            self.status_label.setText("✓ 设置已保存")
            QMessageBox.information(
                self, "设置已保存",
                f"API 地址: {new_settings['api_base_url']}\n"
                f"模型: {new_settings['model']}\n"
                f"代理: {new_settings['proxy'] or '无'}"
            )

    # ── 构建界面 ───────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet(f"background:{BG};")

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 4)
        main_layout.setSpacing(6)

        content_row = QHBoxLayout()
        content_row.setSpacing(10)

        # ── 左侧栏 ──────────────────────────────
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet(
            f"QFrame{{background:{SIDEBAR_BG};border-radius:16px;border:1px solid {BORDER};}}"
        )
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 40))
        sidebar.setGraphicsEffect(shadow)

        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(6)

        title = QLabel("💬  镜")
        title.setStyleSheet(f"color:{TEXT};font-size:15px;font-weight:700;background:transparent;border:none;")
        sl.addWidget(title)

        btn_import = QPushButton("📥 导入聊天记录")
        btn_import.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;border:none;border-radius:8px;"
            f"padding:6px;font-size:12px;font-weight:600;}}"
            f"QPushButton:hover{{background:#8B7D6D;}}"
        )
        btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_import.clicked.connect(self._import_records)
        sl.addWidget(btn_import)

        self.import_count_label = QLabel(self._import_count_text())
        self.import_count_label.setStyleSheet(f"color:{GRAY};font-size:10px;background:transparent;padding:0 4px;")
        sl.addWidget(self.import_count_label)

        # 清空导入按钮（放在同一行）
        import_row = QHBoxLayout()
        import_row.setContentsMargins(0, 0, 0, 0)
        import_row.setSpacing(4)
        import_row.addWidget(self.import_count_label, 1)
        self.btn_clear_import = QPushButton("清空")
        self.btn_clear_import.setFixedSize(36, 18)
        self.btn_clear_import.setStyleSheet(
            f"QPushButton{{background:transparent;color:{DANGER};border:1px solid {BORDER};"
            f"border-radius:4px;font-size:9px;}}"
            f"QPushButton:hover{{background:#FDEDEC;}}"
        )
        self.btn_clear_import.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_import.clicked.connect(self._clear_imported_records)
        self.btn_clear_import.setVisible(count_imported() > 0)
        import_row.addWidget(self.btn_clear_import)
        sl.addLayout(import_row)

        btn_new = QPushButton("+ 新对话")
        btn_new.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;border:none;border-radius:8px;"
            f"padding:6px;font-size:12px;font-weight:600;}}"
            f"QPushButton:hover{{background:#8B7D6D;}}"
        )
        btn_new.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_new.clicked.connect(self._new_conv)
        sl.addWidget(btn_new)

        self.conv_list = QListWidget()
        self.conv_list.setStyleSheet(
            f"QListWidget{{background:transparent;border:none;font-size:12px;color:{TEXT};}}"
            f"QListWidget::item{{padding:8px 10px;border-radius:6px;}}"
            f"QListWidget::item:hover{{background:{SIDEBAR_HOVER};}}"
            f"QListWidget::item:selected{{background:{SIDEBAR_SELECTED};font-weight:600;}}"
        )
        self.conv_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.conv_list.currentItemChanged.connect(self._switch_conv)
        sl.addWidget(self.conv_list, 1)

        btn_clear = QPushButton("清空当前对话")
        btn_clear.setStyleSheet(
            f"QPushButton{{background:transparent;color:{GRAY};border:1px solid {BORDER};"
            f"border-radius:6px;padding:4px;font-size:11px;}}"
            f"QPushButton:hover{{background:{SIDEBAR_HOVER};}}"
        )
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.clicked.connect(self._clear_conv)
        sl.addWidget(btn_clear)

        btn_del = QPushButton("删除当前对话")
        btn_del.setStyleSheet(
            f"QPushButton{{background:transparent;color:{DANGER};border:1px solid {BORDER};"
            f"border-radius:6px;padding:4px;font-size:11px;}}"
            f"QPushButton:hover{{background:#FDEDEC;}}"
        )
        btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_del.clicked.connect(self._delete_conv)
        sl.addWidget(btn_del)

        btn_all = QPushButton("清空全部")
        btn_all.setStyleSheet(
            f"QPushButton{{background:transparent;color:{DANGER};border:1px solid {BORDER};"
            f"border-radius:6px;padding:4px;font-size:11px;}}"
            f"QPushButton:hover{{background:#FDEDEC;}}"
        )
        btn_all.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_all.clicked.connect(self._clear_all)
        sl.addWidget(btn_all)

        btn_settings = QPushButton("⚙ 设置")
        btn_settings.setStyleSheet(
            f"QPushButton{{background:transparent;color:{GRAY};border:1px solid {BORDER};"
            f"border-radius:6px;padding:4px;font-size:11px;}}"
            f"QPushButton:hover{{background:{SIDEBAR_HOVER};}}"
        )
        btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_settings.clicked.connect(self._open_settings)
        sl.addWidget(btn_settings)

        content_row.addWidget(sidebar)

        # ── 右侧聊天区 ──────────────────────────
        card = QFrame()
        card.setStyleSheet(f"QFrame{{background:{CARD};border-radius:20px;border:1px solid {BORDER};}}")
        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(32)
        shadow2.setOffset(0, 4)
        shadow2.setColor(QColor(0, 0, 0, 50))
        card.setGraphicsEffect(shadow2)

        cl = QVBoxLayout(card)
        cl.setContentsMargins(0, 0, 0, 0)
        self.chat_area = ChatArea()
        self.chat_area.btn_send.clicked.connect(self._send)
        self.chat_area.input_box.installEventFilter(self)
        cl.addWidget(self.chat_area)
        content_row.addWidget(card, 1)

        main_layout.addLayout(content_row, 1)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(f"color:{GRAY};font-size:10px;background:transparent;padding:2px 10px;")
        main_layout.addWidget(self.status_label)

    def _import_count_text(self) -> str:
        total = count_imported()
        if total == 0:
            return "暂无导入记录"
        profiles = get_all_profiles()
        if profiles:
            return f"已导入 {total:,} 条 · 已生成画像"
        return f"已导入 {total:,} 条"

    # ── 发送消息 ───────────────────────────────────
    def eventFilter(self, obj, event):
        if obj == self.chat_area.input_box and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self._send()
                return True
        return super().eventFilter(obj, event)

    def _send(self):
        if self.is_streaming:
            return
        text = self.chat_area.input_box.toPlainText().strip()
        if not text:
            return

        self.chat_area.input_box.clear()
        self.chat_area.input_box.setFocus()
        self.chat_area.add_msg(text, True)
        self.chat_area.add_to_history("user", text)
        save_msg(self.current_conv_id, "user", text)

        self.is_streaming = True
        self.chat_area.btn_send.setStyleSheet(
            f"background:{BORDER};color:{GRAY};border:none;border-radius:15px;font-size:13px;"
        )
        self.chat_area.add_msg("", False)
        self.status_label.setText("思考中…")

        # 从向量索引检索相关历史对话（本地运行，零 token 成本）
        system_prompt = build_system_prompt(self.settings)
        if self.memory.is_ready:
            try:
                memory_results = self.memory.search(text, top_k=15)
                memory_context = self.memory.format_for_llm(memory_results)
                if memory_context:
                    system_prompt += memory_context
            except Exception:
                pass  # 检索失败不影响对话

        self.st = StreamThread(self.settings, system_prompt, self.chat_area._history[:-1], text)
        self.st.text_token.connect(lambda t: self.chat_area.append_msg(t))
        self.st.stream_done.connect(self._done)
        self.st.error_occurred.connect(self._err)
        self.st.start()

    def _done(self, collected):
        self.chat_area.add_to_history("assistant", collected)
        save_msg(self.current_conv_id, "assistant", collected)
        self.is_streaming = False
        self.chat_area.btn_send.setStyleSheet(
            f"background:{ACCENT};color:white;border:none;border-radius:15px;font-size:13px;"
        )
        self._refresh_sidebar()
        self.status_label.setText("就绪")

        # 累计足够多的新对话后，后台自动更新画像
        self._msg_since_update += 2  # 用户消息 + AI 回复算一轮
        if self._msg_since_update >= 30 and not (self._profile_worker and self._profile_worker.isRunning()):
            self._update_profile_background()

    def _err(self, msg):
        # 写入日志文件
        import traceback
        with open(APP_DIR / "error.log", "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now()}] 对话错误:\n{msg}\n{traceback.format_exc()}\n")
        # 针对连接错误给出更友好的提示
        if "10061" in msg or "积极拒绝" in msg or "Connection refused" in msg:
            hint = "\n\n💡 提示：如果你开了代理软件（Clash/V2Ray 等），请检查代理是否正常运行。\n也可以打开「设置」→「代理（可选）」手动填写代理地址，例如 http://127.0.0.1:7890"
        elif "10060" in msg or "超时" in msg or "timeout" in msg:
            hint = "\n\n💡 提示：连接超时。如果你在国内使用海外 API，需要配置代理。\nDeepSeek 是国内服务，一般不需要代理——检查网络或防火墙设置。"
        elif "401" in msg or "Unauthorized" in msg:
            hint = "\n\n💡 提示：API Key 无效。请检查设置中的 API Key 是否正确。"
        elif "404" in msg:
            hint = "\n\n💡 提示：API 地址或模型名称错误。请检查设置。"
        else:
            hint = ""
        self.chat_area.append_msg(f"\n\n❌ 错误：{msg}{hint}")
        self.is_streaming = False
        self.chat_area.btn_send.setStyleSheet(
            f"background:{ACCENT};color:white;border:none;border-radius:15px;font-size:13px;"
        )
        self.status_label.setText(f"✗ {msg[:60]}")


# ═════════════════════════════════════════════════════════
# 启动
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = JingWindow()
    window.show()
    sys.exit(app.exec())
