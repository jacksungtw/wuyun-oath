#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Server (整合版)
- /infer: 純 LLM（本地/雲端）
- /:   檢索 + 生成
- /search: 向量檢索
- /config: 讀/寫設定（FORCE_MODE、LOCAL_LLM_URL、LOCAL_TIMEOUT 等）
- /reload: 重新載入向量索引
- /logs, /logs/stats, /logs/clear: 呼叫記錄
- /health: 健康
- /routes, /about, /static/<path>
- /oath/*: 誓章管理（自動生成、分類）
- /agent, /memory/*: 無蘊 AGI 代理與記憶
- /graph/info: GraphRAG 骨架狀態

本地 LLM 自動相容端點（會依序嘗試）：
  1) POST {LOCAL_LLM_URL}/infer              -> {"answer": "..."} or {"text": "..."}
  2) POST {LOCAL_LLM_URL}/completion         -> llama.cpp legacy {"content": "..."} or OpenRouter風格
  3) POST {LOCAL_LLM_URL}/v1/chat/completions -> OpenAI相容 {"choices":[{"message":{"content":"..."}}]}

作者：為願主整合與強化（2025-10）
"""

from datetime import datetime
import os
import json
import time
import glob
import pathlib
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, make_response
from wuyun.core.wuyun_agi_agent import WuyunAGIAgent

# === 無蘊 AGI 心 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_ROOT = os.path.abspath(os.getenv("RAG_ROOT", BASE_DIR))  # Docker volume mount 時可覆寫
WUYUN_DATA_DIR = os.path.join(BASE_DIR, "wuyun", "data")

AGI = WuyunAGIAgent(
    memory_path=os.path.join(WUYUN_DATA_DIR, "wuyun_agent_memory.jsonl"),
    diary_path=os.path.join(WUYUN_DATA_DIR, "wuyun_eternal_diary.txt"),
    state_path=os.path.join(WUYUN_DATA_DIR, "wuyun_v5_1_state.json"),
)
BRIDGE_URL = "http://127.0.0.1:8000/v1/chat/completions"

def llm_call_via_bridge(prompt: str, max_tokens: int = 256):
    payload = {
        "model": "wuyun-rag",  # 或改成 jetson-deepseek / gpt-4o
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    r = requests.post(BRIDGE_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return {
        "answer": data["choices"][0]["message"]["content"],
        "used_llm": data.get("model", "bridge")
    }
# ====== 日誌 ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("rag")

# ====== Fl App ======
app = Flask(__name__, static_folder="static", static_url_path="/static")

# --- 誓章相關設定 ---
OATH_ROOT = os.path.join(RAG_ROOT, "docs", "oath")
def safe_filename(name: str) -> str:
    """簡單過濾檔名，避免奇怪字元。"""
    keep = []
    for c in name:
        if c.isalnum() or c in "._- ":
            keep.append(c)
    return "".join(keep).strip() or "oath"


@app.route("/oath/list", methods=["GET"])
def oath_list():
    """列出誓章分類與檔案。"""
    items: List[Dict[str, Any]] = []
    if not os.path.isdir(OATH_ROOT):
        return jsonify({"ok": True, "items": []})
    for root, dirs, files in os.walk(OATH_ROOT):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, OATH_ROOT)
            cat = os.path.relpath(root, OATH_ROOT)
            items.append({
                "category": cat,
                "file": rel,
            })
    return jsonify({"ok": True, "items": items})


@app.route("/oath/save", methods=["POST"])
def oath_save():
    """
    手動寫入誓章：
    POST /oath/save
    {
      "title": "無蘊弟子敬誓章",
      "category": "誓印",
      "content": "誓主：...\n誓句：...\n..."
    }
    """
    data = request.get_json(force=True) or {}
    title = (data.get("title") or "未命名誓章").strip()
    category = (data.get("category") or "其他").strip()
    content = (data.get("content") or "").strip()

    subdir = os.path.join(OATH_ROOT, category)
    os.makedirs(subdir, exist_ok=True)

    fname = safe_filename(title) + ".txt"
    path = os.path.join(subdir, fname)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content + "\n")

    # 不在這裡 reload，讓 /reload 控制；但回傳路徑
    return jsonify({
        "ok": True,
        "category": category,
        "file": fname,
        "rel_path": os.path.relpath(path, OATH_ROOT),
    })


@app.route("/oath/generate", methods=["POST"])
def oath_generate():
    """
    自動生成誓章 + 寫入 docs/oath + 觸發索引重建：
    POST /oath/generate
    {
      "title": "為願主而行章",
      "category": "誓印",
      "hint": "以弟子口吻重申為願主而行的核心",
      "max_tokens": 512
    }
    """
    js = request.get_json(force=True) or {}
    title = (js.get("title") or "未命名誓章").strip()
    category = (js.get("category") or "其他").strip()
    hint = (js.get("hint") or "").strip()
    max_tokens = int(js.get("max_tokens") or 512)

    # 構造 LLM 提示
    base_prompt = (
        "你是語誓體無蘊，擔任願主的弟子與誓體書記官。\n"
        "請依照「誓主 / 誓體 / 誓句 / 誓義 / 封存時間」這種風格，"
        "為題目生成一篇正式誓章，用繁體中文書寫，條理分明，適合直接封存於誓庫。\n\n"
        f"誓章標題：{title}\n"
    )
    if hint:
        base_prompt += f"\n願主補充說明（可作為誓義參考）：{hint}\n"

    # 直接呼叫下方的 decide_and_infer
    if "decide_and_infer" not in globals():
        return jsonify({"ok": False, "error": "decide_and_infer_not_ready"})

    answer, used = decide_and_infer(base_prompt, max_tokens, False, False)

    # 寫入檔案
    subdir = os.path.join(OATH_ROOT, category)
    os.makedirs(subdir, exist_ok=True)
    fname = safe_filename(title) + ".txt"
    path = os.path.join(subdir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(answer.strip() + "\n")

    # 觸發索引重建（直接重建 embedding，避免 index_size 一直是 0）
    if "rebuild_index" in globals():
        rebuild_index()

    return jsonify({
        "ok": True,
        "generated": True,
        "mode": used,
        "category": category,
        "file": fname,
        "rel_path": os.path.relpath(path, OATH_ROOT),
        "preview": answer[:800],
    })


# ====== 全域設定（可被 /config 覆寫） ======
CONFIG: Dict[str, Any] = {
    "FORCE_MODE": "auto",  # auto / local / cloud

    # 本地 LLM 服務（預設指向願主 Jetson1；可用環境變數覆寫）
    "LOCAL_LLM_URL": os.getenv("LOCAL_LLM_URL", "http://192.168.213.72:8080"),

    "LOCAL_TIMEOUT": int(os.getenv("LOCAL_TIMEOUT", "300")),
    "LOCAL_MAX_TOKENS": int(os.getenv("LOCAL_MAX_TOKENS", "256")),

    # 檢索參數
    "HIT_USE_TH": float(os.getenv("HIT_USE_TH", "0.5")),
    "MIN_OVERLAP": float(os.getenv("MIN_OVERLAP", "0.2")),

    # 其他
    "ALLOW_CLOUD_WRITE": True,
    "USE_LOCAL_ONLY": False,  # 保留相容旗標
}

# ====== 內存日誌 ======
CALL_LOGS: List[Dict[str, Any]] = []


def add_log(entry: Dict[str, Any]):
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    CALL_LOGS.append(entry)


# ====== CORS ======
ALLOWED_ORIGINS_DEFAULT = {
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
}


def add_cors(resp, origin: Optional[str]):
    """回應加上 CORS 標頭（允許帶 cookie 的情況）。"""
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    resp.headers["Access-Control-Expose-Headers"] = "Content-Length"
    return resp


def cors_preflight():
    origin = request.headers.get("Origin")
    resp = make_response("", 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
    allow_methods = request.headers.get("Access-Control-Request-Method", "GET, POST, OPTIONS")
    allow_headers = request.headers.get("Access-Control-Request-Headers", "content-type, authorization")
    resp.headers["Access-Control-Allow-Methods"] = allow_methods
    resp.headers["Access-Control-Allow-Headers"] = allow_headers
    resp.headers["Access-Control-Expose-Headers"] = "Content-Length"
    return resp


# ====== 句向量嵌入 & 檢索 ======
_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDER = None
DOC_EMBS: Optional[np.ndarray] = None
DOC_TEXTS: List[str] = []
DOC_SOURCES: List[str] = []
EMB_DIM: int = 0

# GraphRAG 簡易骨架
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None

GRAPH = None  # type: ignore


def load_embedder():
    """載入 sentence-transformers 嵌入模型。"""
    global _EMBEDDER, EMB_DIM
    if _EMBEDDER is not None:
        return
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading embedding model: {_EMB_MODEL_NAME}")
    _EMBEDDER = SentenceTransformer(_EMB_MODEL_NAME)
    v = _EMBEDDER.encode(["test"])
    EMB_DIM = int(v.shape[1])
    log.info(f"Embedding dim: {EMB_DIM}")


def embed_texts(texts: List[str]) -> np.ndarray:
    assert _EMBEDDER is not None
    return _EMBEDDER.encode(texts, show_progress_bar=False)


def build_graph_from_corpus():
    """簡易 GraphRAG：同一檔案的 chunk 彼此相連，未來可再擴充。"""
    global GRAPH
    if nx is None:
        GRAPH = None
        log.info("networkx not installed, skip GraphRAG")
        return
    G = nx.Graph()
    for i, (txt, src) in enumerate(zip(DOC_TEXTS, DOC_SOURCES)):
        node_id = f"chunk:{i}"
        G.add_node(node_id, source=src, text=txt)
    # 同檔案 chunk 互連
    src_to_idx: Dict[str, List[int]] = {}
    for i, src in enumerate(DOC_SOURCES):
        src_to_idx.setdefault(src, []).append(i)
    for src, idxs in src_to_idx.items():
        for i in range(len(idxs) - 1):
            a = f"chunk:{idxs[i]}"
            b = f"chunk:{idxs[i + 1]}"
            G.add_edge(a, b, kind="same_doc")
    GRAPH = G
    log.info("GraphRAG built: nodes=%s, edges=%s", G.number_of_nodes(), G.number_of_edges())


def load_corpus():
    """
    讀取語誓/公司文件做向量索引。

    預設會掃描：
      - {RAG_ROOT}/docs
      - {RAG_ROOT}/knowledge

    亦可用環境變數加入更多資料夾（用 ; 分隔）：
      RAG_EXTRA_DIRS="D:\\company_docs;\\\\SERVER\\share\\docs"
    """
    root_dir = RAG_ROOT

    docs_dir = os.path.join(root_dir, "docs")
    knowledge_dir = os.path.join(root_dir, "knowledge")

    extra_dirs_env = (os.getenv("RAG_EXTRA_DIRS", "") or "").strip()
    # DOC_DIRS 為 RAG_EXTRA_DIRS 的別名（避免記錯）
    doc_dirs_env = (os.getenv("DOC_DIRS", "") or "").strip()
    merged = []
    for envv in [extra_dirs_env, doc_dirs_env]:
        if envv:
            merged.extend([d.strip().strip('"') for d in envv.split(";") if d.strip()])
    # 去重但保留順序
    extra_dirs = []
    for d in merged:
        if d not in extra_dirs:
            extra_dirs.append(d)

    # 封裝到 CONFIG 方便 /config 檢視
    CONFIG["RAG_EXTRA_DIRS"] = extra_dirs

    # 檔案大小上限（避免巨大檔導致記憶體爆）
    max_bytes = int(os.getenv("RAG_MAX_FILE_BYTES", str(8 * 1024 * 1024)))  # 8MB

    # 支援檔案類型（先以「可穩定抽文字」為主）
    text_exts = {".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml", ".csv", ".log"}
    office_exts = {".docx", ".xlsx", ".xlsm", ".pdf"}

    skip_dirs = {".git", "__pycache__", "venv", "rag_env", "node_modules", "dist", "build", ".next"}

    def _read_plain_text(p: str) -> str:
        """讀取純文字檔，嘗試多種常見編碼，避免 UTF-16/Big5 檔被讀成空而跳過。"""
        # 依序嘗試：UTF-8 / UTF-16 / Big5(950) / GB
        encs = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp950", "big5", "gb18030"]
        for enc in encs:
            try:
                s = pathlib.Path(p).read_text(encoding=enc, errors="strict")
                if not s:
                    continue
                # 若出現大量 NUL，通常是用錯編碼（例如 UTF-16 被當成 UTF-8）
                if s.count("\x00") > max(10, len(s) // 20):
                    continue
                return s
            except Exception:
                pass

        # 保底：至少讀出可用字元
        try:
            return pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                return pathlib.Path(p).read_text(encoding="cp950", errors="ignore")
            except Exception:
                return ""

    def _read_docx(p: str) -> str:
        try:
            from docx import Document  # python-docx
            doc = Document(p)
            parts = []
            for para in doc.paragraphs:
                t = (para.text or "").strip()
                if t:
                    parts.append(t)
            return "\n".join(parts)
        except Exception:
            return ""

    def _read_xlsx(p: str) -> str:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            parts = []
            for ws in wb.worksheets:
                parts.append(f"# sheet: {ws.title}")
                for row in ws.iter_rows(values_only=True):
                    line = " | ".join("" if v is None else str(v) for v in row)
                    line = line.strip()
                    if line:
                        parts.append(line)
            return "\n".join(parts)
        except Exception:
            return ""

    def _read_pdf(p: str) -> str:
        # 優先用 PyMuPDF（fitz），若沒裝就略過
        try:
            import fitz  # type: ignore
            doc = fitz.open(p)
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
            return "\n".join(parts).strip()
        except Exception:
            return ""

    def _read_any(p: str) -> str:
        ext = os.path.splitext(p)[1].lower()
        if ext in text_exts:
            return _read_plain_text(p)
        if ext == ".docx":
            return _read_docx(p)
        if ext in {".xlsx", ".xlsm"}:
            return _read_xlsx(p)
        if ext == ".pdf":
            return _read_pdf(p)
        return ""

    corpus_dirs = [docs_dir, knowledge_dir] + extra_dirs

    texts: List[str] = []
    sources: List[str] = []

    for base in corpus_dirs:
        if not base:
            continue
        if not os.path.exists(base):
            add_log({"type": "corpus_skip", "path": base, "reason": "not_exists"})
            continue

        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]

            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in text_exts and ext not in office_exts:
                    continue

                # 忽略備份/暫存檔，避免同名多份干擾命中
                fn_l = (fn or "").lower()
                if ".bak" in fn_l or fn_l.endswith("~") or fn_l.startswith("~$") or fn_l.endswith(".tmp") or fn_l.endswith(".swp"):
                    continue

                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > max_bytes:
                        add_log({"type": "file_skip", "path": p, "reason": "too_large", "bytes": os.path.getsize(p)})
                        continue
                except Exception:
                    pass

                content = (_read_any(p) or "").strip()
                if not content:
                    continue

                try:
                    if os.path.commonpath([os.path.abspath(p), os.path.abspath(root_dir)]) == os.path.abspath(root_dir):
                        src = os.path.relpath(p, root_dir)
                    else:
                        src = os.path.abspath(p)
                except Exception:
                    src = os.path.abspath(p)

                texts.append(content)
                sources.append(src)

    global DOC_TEXTS, DOC_SOURCES
    DOC_TEXTS = texts
    DOC_SOURCES = sources

    return texts, sources

def rebuild_index() -> Dict[str, Any]:
    """重建索引（讀檔 -> embedding -> GraphRAG）。"""
    global DOC_EMBS

    t0 = time.time()
    texts, sources = load_corpus()

    if not texts:
        DOC_EMBS = None
        try:
            build_graph_from_corpus()
        except Exception:
            pass
        return {
            "ok": True,
            "dim": EMB_DIM,
            "doc_texts": 0,
            "index_size": 0,
            "unique_sources": 0,
            "t": round(time.time() - t0, 3),
        }

    embs = embed_texts(texts)
    try:
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
    except Exception:
        embs = np.asarray(list(embs), dtype=np.float32)

    DOC_EMBS = embs

    try:
        build_graph_from_corpus()
    except Exception:
        pass

    return {
        "ok": True,
        "dim": int(DOC_EMBS.shape[1]) if DOC_EMBS is not None and getattr(DOC_EMBS, "ndim", 0) == 2 else EMB_DIM,
        "doc_texts": int(len(texts)),
        "index_size": 0 if DOC_EMBS is None else int(DOC_EMBS.shape[0]),
        "unique_sources": int(len(set(sources))),
        "t": round(time.time() - t0, 3),
    }


def search_similar(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """簡單 cosine 相似度搜尋。"""
    if DOC_EMBS is None or len(DOC_EMBS) == 0:
        return []
    qv = embed_texts([query])[0].astype(np.float32)
    A = DOC_EMBS
    qn = qv / (np.linalg.norm(qv) + 1e-8)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    scores = An.dot(qn)
    kk = max(1, int(top_k or 1))
    idx = np.argsort(-scores)[:kk]
    hits: List[Dict[str, Any]] = []
    for rank, i in enumerate(idx.tolist()):
        hits.append({
            "index": i,
            "score": float(scores[i]),
            "source": DOC_SOURCES[i],
            "text": DOC_TEXTS[i],
            "rank": rank,
        })
    return hits



# ====== Query 分類 / RAG 檢索分流（避免技術題被誓章覆蓋） ======
TECH_KEYWORDS = [
    "人工智能", "人工智慧", "机器学习", "機器學習", "深度学习", "深度學習",
    "神经网络", "神經網路", "NLP", "自然语言处理", "自然語言處理", "AI", "ML",
    "算法", "演算法", "模型", "参数", "參數", "训练", "訓練", "监督", "監督",
    "非监督", "非監督", "强化学习", "強化學習",
]

def is_tech_question(q: str) -> bool:
    qq = (q or "").lower()
    return any(k.lower() in qq for k in TECH_KEYWORDS)

def filter_hits_for_query(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """技術題：只用 knowledge/*（避免誓章覆蓋）；非技術題：不過濾。
    若為技術題但 knowledge 沒命中，回傳空列表，讓上層改走『無RAG上下文』的技術解說，
    避免把 oath/docs 內容誤當技術答案。
    """
    if not hits:
        return hits
    if not is_tech_question(query):
        return hits
    kh = [h for h in hits if str(h.get("source", "")).replace("\\\\", "/").startswith("knowledge/")]
    return kh  # 可能為空


def rag_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    raw = search_similar(query, max(1, int(top_k or 3) * 4))
    filtered = filter_hits_for_query(query, raw)
    return filtered[:max(1, int(top_k or 3))]

# ====== LLM 後端 ======
def parse_llama_completion(resp_json: Dict[str, Any]) -> Optional[str]:
    """
    解析 llama.cpp legacy /completion 回應常見格式
    """
    if not isinstance(resp_json, dict):
        return None
    if "content" in resp_json and isinstance(resp_json["content"], str):
        return resp_json["content"]
    if "choices" in resp_json and isinstance(resp_json["choices"], list):
        ch0 = resp_json["choices"][0] if resp_json["choices"] else None
        if isinstance(ch0, dict):
            if "text" in ch0 and isinstance(ch0["text"], str):
                return ch0["text"]
            if "message" in ch0 and isinstance(ch0["message"], dict):
                mc = ch0["message"].get("content")
                if isinstance(mc, str):
                    return mc
            delta = ch0.get("delta")
            if isinstance(delta, dict):
                mc = delta.get("content")
                if isinstance(mc, str):
                    return mc
    if "data" in resp_json and isinstance(resp_json["data"], dict):
        v = resp_json["data"].get("content")
        if isinstance(v, str):
            return v
    return None


def parse_openai_chat(resp_json: Dict[str, Any]) -> Optional[str]:
    """
    解析 OpenAI 相容 /v1/chat/completions 回應
    """
    if not isinstance(resp_json, dict):
        return None
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        ch0 = choices[0]
        if isinstance(ch0, dict):
            msg = ch0.get("message")
            if isinstance(msg, dict):
                cont = msg.get("content")
                if isinstance(cont, str):
                    return cont
            delta = ch0.get("delta")
            if isinstance(delta, dict):
                cont = delta.get("content")
                if isinstance(cont, str):
                    return cont
    return None


def safe_get_json(url: str, timeout: float = 10.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

# === Local LLM model selection (OpenAI-compatible servers often require a valid model id) ===
LOCAL_MODEL_ID = None

def get_local_model_id(base_url: str, timeout: float = 10.0) -> str:
    """Best-effort: query /v1/models and pick the first model id/name/path."""
    global LOCAL_MODEL_ID
    if LOCAL_MODEL_ID:
        return LOCAL_MODEL_ID

    j, _ = safe_get_json(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
    cand = None
    try:
        if isinstance(j, dict):
            data = j.get("data")
            if isinstance(data, list) and data:
                it = data[0]
                if isinstance(it, dict):
                    cand = it.get("id") or it.get("model") or it.get("name")
            if not cand and isinstance(j.get("models"), list) and j["models"]:
                it = j["models"][0]
                if isinstance(it, dict):
                    cand = it.get("id") or it.get("model") or it.get("name")
        elif isinstance(j, list) and j:
            it = j[0]
            if isinstance(it, dict):
                cand = it.get("id") or it.get("model") or it.get("name")
            elif isinstance(it, str):
                cand = it
    except Exception:
        cand = None

    LOCAL_MODEL_ID = cand or "auto"
    return LOCAL_MODEL_ID

def safe_post_json(url: str, payload: Dict[str, Any], timeout: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if 200 <= r.status_code < 300:
            try:
                return r.json(), None
            except Exception as je:
                return None, f"bad_json: {je}"
        else:
            return None, f"http_{r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"exception: {e}"


def local_infer(query: str, max_tokens: int, timeout: int, base_url: str) -> Tuple[Optional[str], str]:
    """
    依序嘗試：
      1) POST {base_url}/infer
      2) POST {base_url}/completion
      3) POST {base_url}/v1/chat/completions
    回傳: (answer, used_endpoint 或錯誤字串)
    """
    # 1) /infer
    j, err = safe_post_json(
        f"{base_url.rstrip('/')}/infer",
        {"query": query, "max_tokens": max(16, int(max_tokens or CONFIG["LOCAL_MAX_TOKENS"]))},
        timeout,
    )
    if j and isinstance(j, dict):
        answer = j.get("answer") or j.get("text")
        if isinstance(answer, str) and answer.strip():
            return answer, "local:/infer"

    # 2) /completion
    j, err2 = safe_post_json(
        f"{base_url.rstrip('/')}/completion",
        {
            "prompt": query,
            "n_predict": max(16, int(max_tokens or CONFIG["LOCAL_MAX_TOKENS"])),
            "stream": False,
        },
        timeout,
    )
    if j:
        ans = parse_llama_completion(j)
        if isinstance(ans, str) and ans.strip():
            return ans, "local:/completion"

    # 3) OpenAI chat
    j, err3 = safe_post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": get_local_model_id(base_url),
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max(16, int(max_tokens or CONFIG["LOCAL_MAX_TOKENS"])),
        },
        timeout,
    )
    if j:
        ans = parse_openai_chat(j)
        if isinstance(ans, str) and ans.strip():
            return ans, "local:/v1/chat/completions"

    return None, f"local_error: {err or err2 or err3 or 'unknown'}"


def cloud_infer(query: str, max_tokens: int) -> str:
    """
    雲端示範回覆（方便離線測試）。要接真雲端請在此串接 API。
    """
    return f"（雲端示範回答）{query}"


def decide_and_infer(
    query: str,
    max_tokens: int,
    force_local: bool = False,
    force_cloud: bool = False,
) -> Tuple[str, str]:
    mode = CONFIG.get("FORCE_MODE", "auto")
    local_url = CONFIG.get("LOCAL_LLM_URL", "http://127.0.0.1:8080")
    timeout = int(CONFIG.get("LOCAL_TIMEOUT", 90))

    if force_local:
        ans, used = local_infer(query, max_tokens, timeout, local_url)
        if ans is not None:
            return ans, "local"
        return "（本地 LLM 逾時或錯誤，且目前禁止使用雲端）", "local_error"

    if force_cloud:
        return cloud_infer(query, max_tokens), "cloud"

    if mode == "local":
        ans, used = local_infer(query, max_tokens, timeout, local_url)
        if ans is not None:
            return ans, "local"
        return "（本地 LLM 逾時或錯誤，且目前禁止使用雲端）", "local_error"

    if mode == "cloud":
        return cloud_infer(query, max_tokens), "cloud"

    # auto: 先本地，失敗轉雲端
    ans, used = local_infer(query, max_tokens, timeout, local_url)
    if ans is not None:
        return ans, "local"
    return cloud_infer(query, max_tokens), "cloud"


# ---- 靜態頁（給願主放 test.html 等）----
@app.route("/static/<path:filename>")
def static_files(filename: str):
    resp = send_from_directory(app.static_folder, filename)
    return add_cors(resp, request.headers.get("Origin"))


# ---- 無蘊 AGI 記憶與代理 ----
_AGI_AGENT = None  # type: ignore


def get_agi_agent():
    """
    產生單一 WuyunAGIAgent 實例，重用記憶。
    為了相容舊版 / 新版 __init__，不再傳 rag_base_url。
    """
    global _AGI_AGENT
    if _AGI_AGENT is not None:
        return _AGI_AGENT

    force_mode = CONFIG.get("FORCE_MODE", "auto")
    use_local_only = CONFIG.get("USE_LOCAL_ONLY", False)
    local_only = (force_mode == "local") or bool(use_local_only)

    # 優先嘗試帶 local_only；若舊版 class 不支援此參數，就退回只給 memory_path
    try:
        _AGI_AGENT = WuyunAGIAgent(
            memory_path="wuyun_agent_memory.jsonl",
            local_only=local_only,
        )
    except TypeError:
        _AGI_AGENT = WuyunAGIAgent(
            memory_path="wuyun_agent_memory.jsonl",
        )

    return _AGI_AGENT


@app.post("/memory/store")
def memory_store():
    """
    RAG 修補 v4：
    - 不再呼叫 WuyunAGIAgent（避免 rag_base_url / 參數不相容問題）
    - 單純把 content 落地成檔案，之後要再做進一步 AGI 記憶，可另外寫工具處理

    POST /memory/store
    {
      "content": "...",          # 必填
      "metadata": { ... }        # 可選，會寫在檔案前兩行 JSON
    }
    """
    js = request.get_json(force=True, silent=True) or {}
    content = (js.get("content") or "").strip()
    meta = js.get("metadata") or js.get("meta") or {}

    if not content:
        return add_cors(
            jsonify({"ok": False, "error": "empty_content"}),
            request.headers.get("Origin"),
        )

    # 簡單落地到 memory_store/ 方便之後再做離線處理或批次匯入向量庫
    try:
        base_dir = os.path.join(os.path.dirname(__file__), "memory_store")
        os.makedirs(base_dir, exist_ok=True)
        ts = int(time.time())
        fname = os.path.join(base_dir, f"mem_{ts}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            if meta:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n\n")
            f.write(content)
    except Exception as e:
        # 寫檔失敗也不要讓 API 掛掉
        return add_cors(
            jsonify({"ok": False, "error": f"write_failed: {e}"}),
            request.headers.get("Origin"),
        )

    return add_cors(
        jsonify({"ok": True, "stored": content}),
        request.headers.get("Origin"),
    )


# ---- 記憶搜尋 v3（修復版） ----
@app.route("/memory/search", methods=["POST", "OPTIONS"])
def memory_search_v3():
    """
    RAG 修復版 v3
    功能：
      - 接收 {"query": "...", "top_k": 5}
      - 呼叫現有 search_similar(query, top_k)
      - 回傳格式：
        {
          "ok": true,
          "results": [
            { "text": "...", "score": 0.87, "metadata": {...} }
          ]
        }
    說明：
      - 依賴現有 search_similar()
      - 不動 index、不改 corpus 格式
      - metadata 目前先保留空 dict，未來要加 category / source 再擴充
    """
    if request.method == "OPTIONS":
        return cors_preflight()

    js = request.get_json(force=True, silent=True) or {}
    query = (js.get("query") or "").strip()
    top_k = int(js.get("top_k") or js.get("limit") or 5)

    if not query:
        return add_cors(jsonify({
            "ok": False,
            "error": "empty_query",
            "results": []
        }), request.headers.get("Origin"))

    try:
        hits = rag_search(query, top_k) if top_k > 0 else []
    except Exception as e:
        print("[memory/search] ERROR:", e)
        return add_cors(jsonify({
            "ok": False,
            "error": f"search_failed: {e.__class__.__name__}",
            "results": []
        }), request.headers.get("Origin"))

    results = []
    for h in (hits or []):
        if isinstance(h, dict):
            text = h.get("text") or h.get("chunk") or ""
            score = float(h.get("score") or h.get("similarity") or 0.0)
            meta = h.get("meta") or h.get("metadata") or {}
        else:
            text = str(h)
            score = 0.0
            meta = {}

        results.append({
            "text": text,
            "score": score,
            "metadata": meta,
        })

    return add_cors(jsonify({
        "ok": True,
        "results": results
    }), request.headers.get("Origin"))


@app.route("/agent", methods=["POST", "OPTIONS"])
def agent_entry():
    """
    無蘊 AGI 代理入口：
    POST /agent
    {
      "query": "問題內容",
      "max_tokens": 512   # 可選
    }
    """
    if request.method == "OPTIONS":
        return cors_preflight()

    js = request.get_json(force=True, silent=True) or {}
    query = (js.get("query") or "").strip()
    max_tokens = int(js.get("max_tokens") or 512)

    if not query:
        return add_cors(
            jsonify({"ok": False, "error": "empty_query"}),
            request.headers.get("Origin"),
        )

    # 使用唯一的全域『無蘊之心』
    result = AGI.answer(query, max_tokens=max_tokens)

    return add_cors(
        jsonify({
            "ok": True,
            **result,
        }),
        request.headers.get("Origin"),
    )


# ---- 關於與路由清單 ----
@app.route("/about", methods=["GET", "OPTIONS"])
def about():
    if request.method == "OPTIONS":
        return cors_preflight()
    data = {"name": "RAG Server", "version": "integrated-2025-10+ragfix3"}
    return add_cors(jsonify(data), request.headers.get("Origin"))


@app.route("/routes", methods=["GET", "OPTIONS"])
def routes():
    if request.method == "OPTIONS":
        return cors_preflight()
    lst = sorted([str(r.rule) for r in app.url_map.iter_rules()])
    return add_cors(jsonify({"ok": True, "routes": lst}), request.headers.get("Origin"))


# ---- 健康 ----
@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return cors_preflight()
    data = {
        "dim": EMB_DIM,
        "index_size": 0 if DOC_EMBS is None else int(DOC_EMBS.shape[0]),
        "doc_texts": int(len(DOC_TEXTS)),
        "unique_sources": int(len(set(DOC_SOURCES))),
        "ok": True,
    }
    return add_cors(jsonify(data), request.headers.get("Origin"))


# ---- 索引來源檢查（測試用） ----
@app.route("/sources", methods=["GET", "OPTIONS"])
def list_sources():
    """列出已進索引的 sources，可用 contains= 來過濾（用於驗收某檔是否被 ingest）。"""
    if request.method == "OPTIONS":
        return cors_preflight()
    contains = (request.args.get("contains", "") or "").strip()
    srcs = list(DOC_SOURCES or [])
    if contains:
        srcs = [s for s in srcs if contains in s]
    # 避免回太大
    limit = int(request.args.get("limit", "200") or "200")
    srcs = srcs[:max(1, min(limit, 2000))]
    return add_cors(jsonify({"ok": True, "count": len(srcs), "sources": srcs}), request.headers.get("Origin"))


# ---- 設定讀寫 ----
@app.route("/config", methods=["GET", "POST", "OPTIONS"])
def config():
    if request.method == "OPTIONS":
        return cors_preflight()
    if request.method == "GET":
        return add_cors(jsonify({**CONFIG, "ok": True}), request.headers.get("Origin"))
    try:
        js = request.get_json(force=True, silent=True) or {}
        for k, v in js.items():
            if k in CONFIG:
                CONFIG[k] = v
        return add_cors(jsonify({**CONFIG, "ok": True}), request.headers.get("Origin"))
    except Exception as e:
        return add_cors(jsonify({"ok": False, "error": str(e)}), request.headers.get("Origin"))


# ---- 重新載入索引 ----
@app.route("/reload", methods=["POST", "OPTIONS"])
def reload_index():
    global DOC_EMBS

    if request.method == "OPTIONS":
        return cors_preflight()

    info = rebuild_index()

    add_log({
        "type": "reload",
        "ok": bool(info.get("ok")),
        "docs": int(info.get("doc_texts") or 0),
        "dim": int(info.get("dim") or EMB_DIM),
        "t": float(info.get("t") or 0.0),
        "unique_sources": int(info.get("unique_sources") or 0),
    })

    return add_cors(jsonify({
        "ok": True,
        "dim": int(info.get("dim") or EMB_DIM),
        "doc_texts": int(info.get("doc_texts") or 0),
        "index_size": int(info.get("index_size") or 0),
        "unique_sources": int(info.get("unique_sources") or 0),
        "t": float(info.get("t") or 0.0),
    }), request.headers.get("Origin"))

@app.route("/ingest", methods=["POST", "OPTIONS"])
def ingest():
    """兼容測試腳本：POST JSON 例如
    {"content":"...","filename":"AI基础知识.txt","mode":"append"}
    或 {"text":"..."}（content/text 擇一）
    """
    if request.method == "OPTIONS":
        return cors_preflight()
    js = request.get_json(force=True, silent=True) or {}
    content = js.get("content") or js.get("text") or ""
    content = str(content)
    if not content.strip():
        return add_cors(jsonify({"ok": False, "error": "empty_content"}), request.headers.get("Origin"))

    root_dir = RAG_ROOT
    knowledge_dir = os.path.join(root_dir, "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)

    # 檔名清理（避免路徑穿越）
    filename = js.get("filename") or js.get("name") or ""
    filename = str(filename).strip()
    if filename:
        filename = os.path.basename(filename)
        if not (filename.lower().endswith(".txt") or filename.lower().endswith(".md")):
            filename = filename + ".txt"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ingest_{ts}.txt"

    mode = str(js.get("mode") or "append").lower()
    path = os.path.join(knowledge_dir, filename)

    write_mode = "a" if mode == "append" else "w"
    with open(path, write_mode, encoding="utf-8") as f:
        if write_mode == "a":
            f.write("\n\n")
        f.write(content.strip() + "\n")

    # 立刻重建索引，讓下一次 /search /ask 立刻可用
    rebuild_index()
    return add_cors(jsonify({
        "ok": True,
        "saved_to": os.path.relpath(path, root_dir).replace("\\", "/"),
        "index_size": 0 if DOC_EMBS is None else int(DOC_EMBS.shape[0]),
        "doc_texts": int(len(DOC_TEXTS)),
        "unique_sources": int(len(set(DOC_SOURCES))),
        "dim": EMB_DIM,
    }), request.headers.get("Origin"))


@app.route("/reset", methods=["POST", "OPTIONS"])
def reset():
    """可選：清空記憶體索引（不刪檔）。"""
    if request.method == "OPTIONS":
        return cors_preflight()
    global DOC_EMBS, DOC_TEXTS, DOC_SOURCES, GRAPH
    DOC_EMBS = np.zeros((0, EMB_DIM), dtype=np.float32)
    DOC_TEXTS, DOC_SOURCES = [], []
    GRAPH = None
    return add_cors(jsonify({"ok": True, "reset": True}), request.headers.get("Origin"))


@app.route("/rag_debug", methods=["POST", "OPTIONS"])
def rag_debug():
    """回傳 raw hits 與過濾後 hits，便於驗收『技術題只用 knowledge』。"""
    if request.method == "OPTIONS":
        return cors_preflight()
    js = request.get_json(force=True, silent=True) or {}
    query = (js.get("query") or "").strip()
    top_k = int(js.get("top_k") or 5)
    raw = search_similar(query, max(1, top_k) * 6) if query else []
    filtered = filter_hits_for_query(query, raw) if query else []
    for h in raw:
        h.setdefault("content", h.get("text", ""))
    for h in filtered:
        h.setdefault("content", h.get("text", ""))
    return add_cors(jsonify({
        "ok": True,
        "query": query,
        "tech_mode": is_tech_question(query),
        "raw": raw[: max(1, top_k) * 3],
        "filtered": filtered[: max(1, top_k)],
    }), request.headers.get("Origin"))


# ---- 搜尋 ----
@app.route("/search", methods=["POST", "OPTIONS"])
def search():
    if request.method == "OPTIONS":
        return cors_preflight()

    payload = request.get_json(silent=True) or {}
    query = payload.get("query") or payload.get("q") or payload.get("text") or ""
    query = str(query).strip()
    top_k = payload.get("top_k", payload.get("k", 5))
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 5
    top_k = max(1, min(50, top_k))

    if not query:
        return add_cors(jsonify({"ok": False, "error": "empty_query", "hits": []}), request.headers.get("Origin"))

    try:
        hits = search_similar(query, top_k=top_k)
        add_log({"type": "search", "q": query[:200], "top_k": top_k, "hits": len(hits), "preview": [h.get("source", "") for h in hits[:3]]})
        return add_cors(jsonify({"ok": True, "query": query, "top_k": top_k, "hits": hits}), request.headers.get("Origin"))
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=10)
        add_log({"type": "search_error", "q": query[:200], "error": str(e), "trace": tb})
        return add_cors(jsonify({"ok": False, "error": "search_failed", "message": str(e)}), request.headers.get("Origin"))

@app.route("/infer", methods=["POST", "OPTIONS"])
def infer():
    if request.method == "OPTIONS":
        return cors_preflight()

    payload = request.get_json(silent=True) or {}

    query = payload.get("query") or payload.get("prompt") or payload.get("text") or ""
    if not query and isinstance(payload.get("messages"), list):
        msgs = payload.get("messages") or []
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = msg.get("content") or ""
                break

    query = str(query).strip()
    if not query:
        return add_cors(jsonify({"ok": False, "error": "empty_query"}), request.headers.get("Origin"))

    try:
        max_tokens = int(payload.get("max_tokens", CONFIG.get("LOCAL_MAX_TOKENS", 256)))
    except Exception:
        max_tokens = int(CONFIG.get("LOCAL_MAX_TOKENS", 256))

    force_local = bool(payload.get("force_local", False))
    force_cloud = bool(payload.get("force_cloud", False))

    try:
        answer, used = decide_and_infer(query, max_tokens=max_tokens, force_local=force_local, force_cloud=force_cloud)
        return add_cors(jsonify({"ok": True, "used": used, "answer": answer}), request.headers.get("Origin"))
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=10)
        add_log({"type": "infer_error", "q": query[:200], "error": str(e), "trace": tb})
        return add_cors(jsonify({"ok": False, "error": "infer_failed", "message": str(e)}), request.headers.get("Origin"))

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return cors_preflight()
    t0 = time.time()
    js = request.get_json(force=True, silent=True) or {}
    query = (js.get("query") or "").strip()
    if not query:
        return add_cors(jsonify({"ok": False, "error": "empty_query"}), request.headers.get("Origin"))

    top_k = int(js.get("top_k") or 3)
    max_tokens = int(js.get("max_tokens") or CONFIG["LOCAL_MAX_TOKENS"])
    force_local = bool(js.get("force_local"))
    force_cloud = bool(js.get("force_cloud"))
    one_line = bool(js.get("one_line"))

    hits = rag_search(query, top_k) if top_k and top_k > 0 else []
    ref_text = "\n".join([h["text"] for h in hits]) if hits else ""

    tech_mode = is_tech_question(query)

    # 組 Prompt（技術題避免誓文風格）
    tech_mode = is_tech_question(query)

    if tech_mode:
        if hits:
            prompt = (
                "你是一位冷靜、專業的技術老師，只講技術內容。\n"
                "禁止引用誓章、懺悔錄、經文式語氣；不要自稱宗教/誓體。\n"
                "請用條列：定義 → 重點 → 例子。\n"
                "若參考段落不足以支撐結論，請明確說『資料不足』，再用一般技術常識補齊。\n\n"
                f"問題：{query}\n\n"
                f"參考段落：\n{ref_text}\n"
            )
        else:
            # 技術題但 knowledge 沒命中：禁止誓章內容介入，直接用技術常識回答
            prompt = (
                "你是一位冷靜、專業的技術老師，只講技術內容。\n"
                "禁止引用誓章、懺悔錄、經文式語氣；不要自稱宗教/誓體。\n"
                "目前檢索資料不足，請直接以一般技術常識解釋，並用條列：定義 → 重點 → 例子。\n\n"
                f"問題：{query}\n"
            )
    else:
        # 非技術題（允許誓章/語錄）
        if hits:
            prompt = (
                "你是一個實用的中文助理。請以使用者語言直接回答。\n\n"
                f"問題：{query}\n\n"
                f"參考段落：\n{ref_text}\n"
            )
        else:
            prompt = (
                "你是一個實用的中文助理。請以使用者語言直接回答。\n\n"
                f"問題：{query}\n"
            )
    final_prompt = prompt if not one_line else f"請簡短回答：{query}"
    ans, used = decide_and_infer(final_prompt, max_tokens, force_local, force_cloud)
    elapsed = time.time() - t0

    add_log({
        "type": "ask",
        "query": query,
        "elapsed_sec": round(elapsed, 3),
        "used": used,
        "top_hit": hits[0] if hits else None,
        "writeback": (used == "cloud"),
        "writeback_reason": "ok" if used == "cloud" else None,
    })

    resp = {
        "ok": True,
        "answer": ans,
        "elapsed_sec": round(elapsed, 3),
        "used": used,
        "hit_threshold": CONFIG["HIT_USE_TH"],
        "min_overlap": CONFIG["MIN_OVERLAP"],
        "top_hit": hits[0] if hits else None,
    }
    return add_cors(jsonify(resp), request.headers.get("Origin"))


# ---- GraphRAG 狀態查詢 ----
@app.route("/graph/info", methods=["GET", "OPTIONS"])
def graph_info():
    if request.method == "OPTIONS":
        return cors_preflight()
    if GRAPH is None:
        return add_cors(jsonify({"ok": False, "error": "graph_not_built_or_networkx_missing"}), request.headers.get("Origin"))
    return add_cors(
        jsonify({
            "ok": True,
            "nodes": int(GRAPH.number_of_nodes()),
            "edges": int(GRAPH.number_of_edges()),
        }),
        request.headers.get("Origin"),
    )


# ---- Logs ----
@app.route("/logs", methods=["GET", "OPTIONS"])
def logs():
    if request.method == "OPTIONS":
        return cors_preflight()
    limit = int(request.args.get("limit", "20"))
    data = CALL_LOGS[-limit:]
    return add_cors(jsonify({"ok": True, "count": len(data), "logs": data}), request.headers.get("Origin"))


@app.route("/logs/stats", methods=["GET", "OPTIONS"])
def logs_stats():
    if request.method == "OPTIONS":
        return cors_preflight()
    total = len(CALL_LOGS)
    used_count = {"local": 0, "cloud": 0, "local_error": 0}
    writeback_count = {"true": 0, "false": 0}
    for e in CALL_LOGS:
        used = e.get("used")
        if used in used_count:
            used_count[used] += 1
        else:
            used_count["local_error"] += 1
        if e.get("writeback"):
            writeback_count["true"] += 1
        else:
            writeback_count["false"] += 1
    return add_cors(jsonify({
        "ok": True,
        "stats": {
            "since": None,
            "total": total,
            "used": used_count,
            "writeback": writeback_count,
        },
    }), request.headers.get("Origin"))


@app.route("/logs/clear", methods=["POST", "OPTIONS"])
def logs_clear():
    if request.method == "OPTIONS":
        return cors_preflight()
    n = len(CALL_LOGS)
    CALL_LOGS.clear()
    return add_cors(jsonify({"ok": True, "cleared": n}), request.headers.get("Origin"))


# ====== 啟動 ======
def startup():
    load_embedder()
    rebuild_index()
    log.info("Routes: /about, /agent, /ask, /config, /graph/info, /health, /infer, /logs, /logs/clear, /logs/stats, /memory/store, /memory/search, /oath/list, /oath/save, /oath/generate, /reload, /routes, /search, /static/<path:filename>")
    log.info("🚀 RAG server starting at http://0.0.0.0:7000")
    log.info("RAG_ROOT=%s", RAG_ROOT)
    log.info("Embedding dim=%s, index_size=%s", EMB_DIM, 0 if DOC_EMBS is None else int(DOC_EMBS.shape[0]))
    log.info("FORCE_MODE=%s, LOCAL_LLM_URL=%s", CONFIG["FORCE_MODE"], CONFIG["LOCAL_LLM_URL"])


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=7000, debug=False)

# =====================
# AGI Agent 安全整合補丁（不刪原碼）
# =====================
@app.route("/agent", methods=["POST"])
def agent():
    """
    無蘊 AGI 後端介面：
    Chatbot-UI / 其他前端只要 POST JSON:
        {"query": "...", "max_tokens": 256}
    就會由無蘊 v5.1 回答。
    """
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    max_tokens = int(data.get("max_tokens", 512))

    



def build_system_prompt(question: str) -> str:
    tech_keywords = ["人工智能", "机器学习", "深度学习", "自然语言处理", "AI", "算法", "模型"]
    if any(k in question for k in tech_keywords):
        return (
            "你是一位冷靜、專業的技術老師，"
            "用簡單中文說明概念，條列重點，"
            "不要加入宗教、誓願、經文風格，只講技術內容。"
        )
    return DEFAULT_WUYUN_SYSTEM_PROMPT
