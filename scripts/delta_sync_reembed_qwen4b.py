#!/usr/bin/env python3
import os
import json
import time
import math
import random
import hashlib
import warnings
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

# Suppress benign urllib3 LibreSSL warning on macOS system Python
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")

import requests

QDRANT = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
SOURCE = os.environ.get("SOURCE_COLLECTION", "mrc_bot_memory")
TARGET = os.environ.get("TARGET_COLLECTION", "mrc_bot")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:4b")
OUT_DIR = os.environ.get("OUT_DIR", "./snapshots")
SAFE_CHUNK_TOKENS = int(os.environ.get("SAFE_CHUNK_TOKENS", "0"))
OVERLAP_TOKENS = int(os.environ.get("OVERLAP_TOKENS", "0"))
ADAPTIVE_TOKEN_STEPS: List[int] = []
RUNTIME_CONTEXT = int(os.environ.get("RUNTIME_CONTEXT", "0"))
SNAPSHOT_MODE = os.environ.get("SNAPSHOT_MODE", "auto").strip().lower()  # auto|force|skip
RESUME_MODE = os.environ.get("RESUME_MODE", "1").lower() in {"1", "true", "yes", "on"}
CHECKPOINT_EVERY_SECONDS = int(os.environ.get("CHECKPOINT_EVERY_SECONDS", "120"))
FAIL_RATE_STOP = float(os.environ.get("FAIL_RATE_STOP", "0.01"))
FAIL_RATE_MIN_PROCESSED = int(os.environ.get("FAIL_RATE_MIN_PROCESSED", "200"))
FAIL_RATE_GUARD_ENABLED = os.environ.get("FAIL_RATE_GUARD_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
BATCH_UPSERT = int(os.environ.get("BATCH_UPSERT", "32"))
JIRA_ISSUE_KEY = os.environ.get("JIRA_ISSUE_KEY", "").strip()

os.makedirs(OUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

LOCAL_EMBED_REQUESTS = 0
EXTERNAL_EMBED_REQUESTS = 0


def detect_runtime_context() -> int:
    # Priority: explicit env override, ollama show model_info, fallback 4096
    if RUNTIME_CONTEXT > 0:
        return RUNTIME_CONTEXT

    try:
        url = f"{OLLAMA}/api/show"
        if not url.startswith("http://localhost:11434"):
            raise RuntimeError(f"local_only_violation: {url}")
        r = requests.post(url, json={"name": MODEL}, timeout=60)
        if r.ok:
            js = r.json()
            mi = js.get("model_info") or {}
            ctx = mi.get("qwen3.context_length") or mi.get("llama.context_length") or mi.get("context_length")
            if isinstance(ctx, (int, float)) and int(ctx) > 0:
                return int(ctx)
            if isinstance(ctx, str) and ctx.isdigit():
                return int(ctx)
    except Exception:
        pass

    return 4096


def build_adaptive_steps(safe_chunk_tokens: int) -> List[int]:
    # Descending shrink steps; keep relevant to detected safe chunk
    seeds = [safe_chunk_tokens, int(safe_chunk_tokens * 0.8), int(safe_chunk_tokens * 0.66), int(safe_chunk_tokens * 0.5), int(safe_chunk_tokens * 0.33)]
    cleaned: List[int] = []
    for s in seeds:
        s = max(256, int(s))
        if s not in cleaned:
            cleaned.append(s)
    return cleaned


def configure_runtime_chunking() -> Dict[str, Any]:
    global ADAPTIVE_TOKEN_STEPS, OVERLAP_TOKENS, SAFE_CHUNK_TOKENS

    runtime_ctx = detect_runtime_context()

    # Effective safe policy
    if runtime_ctx <= 4096:
        safe_chunk = min(3000, max(2048, runtime_ctx - 1024))
        overlap = 256
    else:
        # Keep conservative cap to avoid oversized runtime chunking mismatch
        safe_chunk = min(3000, max(2048, int(runtime_ctx * 0.72)))
        overlap = min(256, max(128, int(safe_chunk * 0.08)))

    # Optional explicit env overrides (bounded)
    if SAFE_CHUNK_TOKENS > 0:
        safe_chunk = min(safe_chunk, SAFE_CHUNK_TOKENS)
    if OVERLAP_TOKENS > 0:
        overlap = min(overlap, OVERLAP_TOKENS)

    overlap = min(overlap, 256 if runtime_ctx <= 4096 else overlap)

    steps = build_adaptive_steps(safe_chunk)

    SAFE_CHUNK_TOKENS = safe_chunk
    OVERLAP_TOKENS = overlap
    ADAPTIVE_TOKEN_STEPS = steps

    return {
        "runtime_context": runtime_ctx,
        "safe_chunk_tokens": SAFE_CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "adaptive_token_steps": ADAPTIVE_TOKEN_STEPS,
    }


class ProgressCheckpoint:
    def __init__(self, out_dir: str, total_remaining: int, every_sec: int = 120):
        self.out_dir = out_dir
        self.total_remaining = total_remaining
        self.every_sec = max(30, int(every_sec))
        self.last_emit = 0.0
        self.path = os.path.join(out_dir, "reembed-progress-latest.json")
        self.history_path = os.path.join(out_dir, "reembed-progress-history.jsonl")

    def emit(self, done: int, remaining: int, force: bool = False):
        now = time.time()
        if (not force) and (now - self.last_emit < self.every_sec):
            return
        self.last_emit = now
        payload = {
            "ts": now_iso(),
            "done": int(done),
            "remaining": int(remaining),
            "total_remaining_at_start": int(self.total_remaining),
            "pct_this_resume": round((done / self.total_remaining * 100.0), 4) if self.total_remaining > 0 else 100.0,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rq(method: str, path: str, **kwargs):
    url = f"{QDRANT}{path}"
    r = session.request(method, url, timeout=120, **kwargs)
    if not r.ok:
        raise RuntimeError(f"Qdrant {method} {path} failed {r.status_code}: {r.text[:500]}")
    return r.json()


def qdrant_get_collection_info(name: str) -> Dict[str, Any]:
    return rq("GET", f"/collections/{name}")["result"]


def qdrant_count(name: str) -> int:
    data = rq("POST", f"/collections/{name}/points/count", json={"exact": True})
    return int(data["result"]["count"])


def qdrant_count_filter(name: str, flt: Dict[str, Any]) -> int:
    data = rq("POST", f"/collections/{name}/points/count", json={"exact": True, "filter": flt})
    return int(data["result"]["count"])


def qdrant_scroll(name: str, with_payload=True, with_vector=False, limit=256):
    offset = None
    while True:
        body = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if offset is not None:
            body["offset"] = offset
        data = rq("POST", f"/collections/{name}/points/scroll", json=body)["result"]
        points = data.get("points", [])
        for p in points:
            yield p
        offset = data.get("next_page_offset")
        if offset is None:
            break


def qdrant_create_snapshot(name: str) -> str:
    data = rq("POST", f"/collections/{name}/snapshots")
    snap = data["result"]["name"]
    return snap


def qdrant_download_snapshot(name: str, snapshot_name: str, out_path: str) -> str:
    url = f"{QDRANT}/collections/{name}/snapshots/{snapshot_name}"
    r = session.get(url, timeout=300, stream=True)
    if not r.ok:
        raise RuntimeError(f"Download snapshot failed {r.status_code}: {r.text[:300]}")
    h = hashlib.sha256()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                h.update(chunk)
    return h.hexdigest()


def vec_dim_from_info(info: Dict[str, Any]) -> int:
    vectors = (((info or {}).get("config") or {}).get("params") or {}).get("vectors")
    if isinstance(vectors, dict) and "size" in vectors:
        return int(vectors["size"])
    raise RuntimeError(f"Unsupported vectors config shape: {vectors}")


def adapt_vector(vec: List[float], target_dim: int) -> Tuple[List[float], str]:
    n = len(vec)
    if n == target_dim:
        return vec, "none"
    if n > target_dim:
        return vec[:target_dim], f"truncate_{n}_to_{target_dim}"
    return vec + [0.0] * (target_dim - n), f"zero_pad_{n}_to_{target_dim}"


def choose_text(payload: Dict[str, Any]) -> str:
    for k in ["enrichedText", "distilledText", "text", "content", "memory", "raw_text"]:
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def est_tokens(s: str) -> int:
    # Lightweight token estimate (qwen-like BPE roughness): max(words, chars/4)
    return max(1, max(len(s.split()), len(s) // 4))


def split_by_token_estimate(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_tokens:
        return [" ".join(words)]

    chunks: List[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    i = 0
    n = len(words)

    while i < n:
        part = words[i : min(i + chunk_tokens, n)]
        if part:
            chunks.append(" ".join(part))
        if i + chunk_tokens >= n:
            break
        i += step

    return chunks


def l2norm(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return v
    return [x / norm for x in v]


def weighted_avg(vectors: List[List[float]], weights: List[float]) -> List[float]:
    dim = len(vectors[0])
    out = [0.0] * dim
    ws = sum(weights) or 1.0
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            out[i] += vec[i] * w
    return l2norm([x / ws for x in out])


def embed_ollama(text: str) -> List[float]:
    global LOCAL_EMBED_REQUESTS

    url = f"{OLLAMA}/api/embed"
    # Enforce local-only embedding endpoint
    if not url.startswith("http://localhost:11434"):
        raise RuntimeError(f"local_only_violation: {url}")

    LOCAL_EMBED_REQUESTS += 1
    r = requests.post(url, json={"model": MODEL, "input": text}, timeout=300)
    if not r.ok:
        raise RuntimeError(f"embed failed {r.status_code}: {r.text[:500]}")
    js = r.json()
    arr = js.get("embeddings") or []
    if not arr or not isinstance(arr, list) or not isinstance(arr[0], list):
        raise RuntimeError(f"invalid embed response shape: {str(js)[:500]}")
    return arr[0]


def build_chunks_with_adaptive_tokens(text: str) -> Tuple[List[str], int]:
    last_err: Optional[Exception] = None
    for tok in ADAPTIVE_TOKEN_STEPS:
        try:
            chunks = split_by_token_estimate(text, chunk_tokens=tok, overlap_tokens=OVERLAP_TOKENS)
            if chunks:
                return chunks, tok
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return [text], ADAPTIVE_TOKEN_STEPS[-1]


def reembed_text_to_dim(text: str, target_dim: int) -> Tuple[List[float], Dict[str, Any]]:
    # Token-based chunking only (32768-compatible path)
    chunks, used_chunk_tokens = build_chunks_with_adaptive_tokens(text)
    chunk_vecs = []
    weights = []
    raw_dim = None

    for c in chunks:
        local_steps = [used_chunk_tokens] + [x for x in ADAPTIVE_TOKEN_STEPS if x < used_chunk_tokens]
        vec: Optional[List[float]] = None
        last_err: Optional[Exception] = None

        for tok in local_steps:
            try:
                sub_chunks = split_by_token_estimate(c, chunk_tokens=tok, overlap_tokens=OVERLAP_TOKENS)
                if not sub_chunks:
                    continue

                if len(sub_chunks) == 1:
                    vec = embed_ollama(sub_chunks[0])
                else:
                    sub_vecs = [embed_ollama(sc) for sc in sub_chunks]
                    sub_w = [est_tokens(sc) for sc in sub_chunks]
                    vec = weighted_avg(sub_vecs, sub_w)
                break
            except Exception as e:
                msg = str(e).lower()
                last_err = e
                if "400" in msg and ("context" in msg or "length" in msg):
                    continue
                raise

        if vec is None:
            raise RuntimeError(f"adaptive_token_embed_failed: {last_err}")

        raw_dim = len(vec)
        chunk_vecs.append(vec)
        weights.append(est_tokens(c))

    merged = chunk_vecs[0] if len(chunk_vecs) == 1 else weighted_avg(chunk_vecs, weights)
    adapted, method = adapt_vector(merged, target_dim)
    meta = {
        "embedding_model": MODEL,
        "embedding_chunks_count": len(chunks),
        "embedding_original_tokens_est": est_tokens(text),
        "embedding_chunk_tokens": used_chunk_tokens,
        "embedding_overlap_tokens": OVERLAP_TOKENS,
        "embedding_vector_raw_dim": raw_dim,
        "embedding_vector_target_dim": target_dim,
        "embedding_vector_transform": method,
        "embedding_updated_at": now_iso(),
        "embedding_local_only": True,
        "embedding_local_endpoint": f"{OLLAMA}/api/embed",
    }
    return adapted, meta


def upsert_points(collection: str, points: List[Dict[str, Any]]):
    rq("PUT", f"/collections/{collection}/points?wait=true", json={"points": points})


def qdrant_retrieve(collection: str, ids: List[Any], with_payload=True, with_vector=True):
    data = rq(
        "POST",
        f"/collections/{collection}/points",
        json={"ids": ids, "with_payload": with_payload, "with_vector": with_vector},
    )
    return data["result"]


def qdrant_search(collection: str, vector: List[float], limit=5):
    data = rq(
        "POST",
        f"/collections/{collection}/points/search",
        json={"vector": vector, "limit": limit, "with_payload": True, "with_vector": False},
    )
    return data["result"]


def load_jira_creds() -> Dict[str, str]:
    p = "/Users/mrcagents/.openclaw/workspace/shared/config/credentials.json"
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    jira = cfg.get("jira", {})
    domain = jira.get("domain", "").rstrip("/")
    email = jira.get("email", "")
    token = jira.get("token", "")
    if not (domain and email and token):
        raise RuntimeError("Jira credentials missing")
    return {"domain": domain, "email": email, "token": token}


def jira_create_issue(summary: str, report_text: str) -> Tuple[str, str]:
    creds = load_jira_creds()
    url = f"{creds['domain']}/rest/api/3/issue"
    adf_doc = {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": report_text[:32000]}]}
        ],
    }
    payload = {
        "fields": {
            "project": {"key": "ASM"},
            "summary": summary[:255],
            "issuetype": {"name": "Task"},
            "labels": ["board-271", "memory-migration", "qwen3-embedding-4b"],
            "description": adf_doc,
        }
    }
    r = requests.post(url, json=payload, auth=(creds["email"], creds["token"]), timeout=60)
    if not r.ok:
        raise RuntimeError(f"Jira create failed {r.status_code}: {r.text[:500]}")
    js = r.json()
    key = js["key"]
    link = f"{creds['domain']}/browse/{key}"
    return key, link


def jira_add_comment(issue_key: str, text: str) -> str:
    creds = load_jira_creds()
    url = f"{creds['domain']}/rest/api/3/issue/{issue_key}/comment"
    body = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": text[:32000]}]}
            ],
        }
    }
    r = requests.post(url, json=body, auth=(creds["email"], creds["token"]), timeout=60)
    if not r.ok:
        raise RuntimeError(f"Jira comment failed {r.status_code}: {r.text[:500]}")
    js = r.json()
    return str(js.get("id", ""))


def main():
    started = time.time()
    report: Dict[str, Any] = {
        "started_at": now_iso(),
        "source": SOURCE,
        "target": TARGET,
        "model": MODEL,
        "status": "RUNNING",
    }

    # Pre-info
    src_info = qdrant_get_collection_info(SOURCE)
    tgt_info = qdrant_get_collection_info(TARGET)
    src_dim = vec_dim_from_info(src_info)
    tgt_dim = vec_dim_from_info(tgt_info)

    runtime_cfg = configure_runtime_chunking()

    src_count_before = qdrant_count(SOURCE)
    tgt_count_before = qdrant_count(TARGET)

    report["pre"] = {
        "source_count": src_count_before,
        "target_count": tgt_count_before,
        "source_dim": src_dim,
        "target_dim": tgt_dim,
        "runtime": runtime_cfg,
        "local_embedding_endpoint": f"{OLLAMA}/api/embed",
    }

    # Snapshots: snapshot once per run family; skip on resume by default
    snap_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    resume_run = RESUME_MODE

    take_snapshot = True
    if SNAPSHOT_MODE == "skip":
        take_snapshot = False
    elif SNAPSHOT_MODE == "auto" and resume_run:
        take_snapshot = False

    if take_snapshot:
        src_snap = qdrant_create_snapshot(SOURCE)
        tgt_snap = qdrant_create_snapshot(TARGET)

        src_snap_path = os.path.join(OUT_DIR, f"{SOURCE}-{snap_ts}-{src_snap}")
        tgt_snap_path = os.path.join(OUT_DIR, f"{TARGET}-{snap_ts}-{tgt_snap}")
        src_sha = qdrant_download_snapshot(SOURCE, src_snap, src_snap_path)
        tgt_sha = qdrant_download_snapshot(TARGET, tgt_snap, tgt_snap_path)

        report["snapshots"] = {
            "mode": SNAPSHOT_MODE,
            "taken": True,
            "source": {"name": src_snap, "path": src_snap_path, "sha256": src_sha},
            "target": {"name": tgt_snap, "path": tgt_snap_path, "sha256": tgt_sha},
        }
    else:
        report["snapshots"] = {
            "mode": SNAPSHOT_MODE,
            "taken": False,
            "reason": "resume_mode_skip" if resume_run else "explicit_skip",
        }
        src_snap = tgt_snap = src_sha = tgt_sha = "N/A"

    # Phase 1 delta sync: skip in resume mode to avoid full collection heavy pass
    inserted = 0
    transform_methods = {}
    t1 = time.time()
    if not resume_run:
        target_ids = set()
        for p in qdrant_scroll(TARGET, with_payload=False, with_vector=False, limit=512):
            target_ids.add(str(p["id"]))

        batch = []
        for sp in qdrant_scroll(SOURCE, with_payload=True, with_vector=True, limit=256):
            sid = str(sp["id"])
            if sid in target_ids:
                continue
            vec = sp.get("vector")
            if isinstance(vec, dict):
                vec = vec.get("default") or []
            if not isinstance(vec, list):
                vec = []
            adapted, method = adapt_vector(vec, tgt_dim)
            transform_methods[method] = transform_methods.get(method, 0) + 1

            payload = sp.get("payload") or {}
            batch.append({"id": sp["id"], "vector": adapted, "payload": payload})
            if len(batch) >= BATCH_UPSERT:
                upsert_points(TARGET, batch)
                inserted += len(batch)
                batch = []

        if batch:
            upsert_points(TARGET, batch)
            inserted += len(batch)

    src_count_after_phase1 = qdrant_count(SOURCE)
    tgt_count_after_phase1 = qdrant_count(TARGET)
    parity_ok = src_count_after_phase1 == tgt_count_after_phase1

    phase1 = {
        "duration_sec": round(time.time() - t1, 2),
        "mode": "skipped_resume" if resume_run else "executed",
        "inserted_missing_points": inserted,
        "vector_transform_methods": transform_methods,
        "source_count_before": src_count_before,
        "target_count_before": tgt_count_before,
        "source_count_after": src_count_after_phase1,
        "target_count_after": tgt_count_after_phase1,
        "count_parity": parity_ok,
    }
    report["phase1"] = phase1

    if (not resume_run) and (not parity_ok):
        report["status"] = "FAIL"
        report["error"] = "Phase 1 count parity failed"
        out_json = os.path.join(OUT_DIR, f"delta-reembed-report-{snap_ts}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    # Phase 2 re-embed remaining target points only (resume-safe)
    t2 = time.time()
    processed = 0
    success = 0
    failed = 0
    failures = []
    transform_counts = {}
    sample_ids = []

    already_done = qdrant_count_filter(
        TARGET,
        {"must": [{"key": "embedding_model", "match": {"value": MODEL}}]},
    )
    report["resume"] = {
        "already_reembedded_before_retry": already_done,
        "remaining_before_retry": max(0, tgt_count_after_phase1 - already_done),
        "resume_basis": f"payload.embedding_model == {MODEL}",
    }

    upsert_batch = []
    total_remaining = max(0, tgt_count_after_phase1 - already_done)
    checkpoint = ProgressCheckpoint(OUT_DIR, total_remaining=total_remaining, every_sec=CHECKPOINT_EVERY_SECONDS)
    checkpoint.emit(done=0, remaining=total_remaining, force=True)

    for tp in qdrant_scroll(TARGET, with_payload=True, with_vector=False, limit=128):
        pid = tp["id"]
        payload = tp.get("payload") or {}

        if payload.get("embedding_model") == MODEL:
            if len(sample_ids) < 200:
                sample_ids.append(pid)
            continue

        processed += 1
        text = choose_text(payload)
        if not text:
            failed += 1
            failures.append({"id": str(pid), "reason": "empty_text"})
        else:
            try:
                new_vec, meta = reembed_text_to_dim(text, tgt_dim)
                method = meta.get("embedding_vector_transform", "none")
                transform_counts[method] = transform_counts.get(method, 0) + 1
                merged_payload = dict(payload)
                merged_payload.update(meta)
                upsert_batch.append({"id": pid, "vector": new_vec, "payload": merged_payload})
                success += 1
                if len(sample_ids) < 200:
                    sample_ids.append(pid)
            except Exception as e:
                failed += 1
                failures.append({"id": str(pid), "reason": str(e)[:300]})

        if len(upsert_batch) >= BATCH_UPSERT:
            upsert_points(TARGET, upsert_batch)
            upsert_batch = []

        remaining_now = max(0, total_remaining - success)
        checkpoint.emit(done=success, remaining=remaining_now, force=False)

        if processed % 50 == 0:
            print(f"phase2 progress: {processed} success={success} failed={failed} remaining={remaining_now}")

        if (
            FAIL_RATE_GUARD_ENABLED
            and processed >= FAIL_RATE_MIN_PROCESSED
            and (failed / processed) > FAIL_RATE_STOP
        ):
            break

    if upsert_batch:
        upsert_points(TARGET, upsert_batch)

    checkpoint.emit(done=success, remaining=max(0, total_remaining - success), force=True)

    stop_due_failure_rate = (
        FAIL_RATE_GUARD_ENABLED
        and processed >= FAIL_RATE_MIN_PROCESSED
        and (failed / processed) > FAIL_RATE_STOP
    )

    tgt_count_after_phase2 = qdrant_count(TARGET)

    # Vector dim verification on sample
    dim_ok = True
    sample_checked = 0
    if sample_ids:
        random.shuffle(sample_ids)
        chk_ids = sample_ids[: min(100, len(sample_ids))]
        got = qdrant_retrieve(TARGET, chk_ids, with_payload=False, with_vector=True)
        sample_checked = len(got)
        for p in got:
            vec = p.get("vector")
            if isinstance(vec, dict):
                vec = vec.get("default") or []
            if not isinstance(vec, list) or len(vec) != tgt_dim:
                dim_ok = False
                break

    # Random sample retrieval search
    search_ok = False
    try:
        qvec, _ = reembed_text_to_dim("memory retrieval validation query", tgt_dim)
        res = qdrant_search(TARGET, qvec, limit=5)
        search_ok = isinstance(res, list) and len(res) > 0
    except Exception:
        search_ok = False

    model_count_after = qdrant_count_filter(
        TARGET,
        {"must": [{"key": "embedding_model", "match": {"value": MODEL}}]},
    )

    phase2 = {
        "duration_sec": round(time.time() - t2, 2),
        "runtime_context": runtime_cfg.get("runtime_context"),
        "safe_chunk_tokens": SAFE_CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "adaptive_token_steps": ADAPTIVE_TOKEN_STEPS,
        "processed": processed,
        "success": success,
        "failed": failed,
        "failure_rate": round((failed / processed) if processed else 0, 6),
        "stopped_due_failure_rate": stop_due_failure_rate,
        "fail_rate_guard_enabled": FAIL_RATE_GUARD_ENABLED,
        "transform_methods": transform_counts,
        "sample_vector_dim_checked": sample_checked,
        "sample_vector_dim_ok": dim_ok,
        "search_ok": search_ok,
        "target_count_after": tgt_count_after_phase2,
        "target_points_with_model_after": model_count_after,
        "local_embed_requests": LOCAL_EMBED_REQUESTS,
        "external_embed_requests": EXTERNAL_EMBED_REQUESTS,
        "local_only_proof": {
            "ollama_url": OLLAMA,
            "model": MODEL,
            "endpoint": f"{OLLAMA}/api/embed",
            "external_embedding_endpoint_used": False,
        },
        "failure_examples": failures[:20],
    }
    report["phase2"] = phase2

    final_pass = (
        (not stop_due_failure_rate)
        and (failed == 0)
        and (tgt_count_after_phase2 == tgt_count_after_phase1)
        and (model_count_after == tgt_count_after_phase1)
        and dim_ok
        and search_ok
    )

    if final_pass:
        report["status"] = "PASS"
    elif stop_due_failure_rate or failed > 0:
        report["status"] = "PARTIAL"
    else:
        report["status"] = "FAIL"

    report["final_verification"] = {
        "source_untouched_count": qdrant_count(SOURCE),
        "target_count_unchanged_after_phase2": tgt_count_after_phase2 == tgt_count_after_phase1,
        "target_model_coverage_full": model_count_after == tgt_count_after_phase1,
        "vector_dim_validation": dim_ok,
        "search_validation": search_ok,
        "elapsed_sec": round(time.time() - started, 2),
    }

    # Prepare markdown report
    md = []
    md.append("# Memory Delta Sync + Re-embed Report")
    md.append(f"- Started: {report['started_at']}")
    md.append(f"- Completed: {now_iso()}")
    md.append(f"- Source: `{SOURCE}` (read-only)")
    md.append(f"- Target: `{TARGET}`")
    md.append(f"- Model: `{MODEL}`")
    md.append("")
    md.append("## Snapshots")
    md.append(f"- Source snapshot: `{src_snap}` | sha256 `{src_sha}`")
    md.append(f"- Target snapshot: `{tgt_snap}` | sha256 `{tgt_sha}`")
    md.append("")
    md.append("## Phase 1 (Delta Sync)")
    md.append(f"- Before: source `{src_count_before}`, target `{tgt_count_before}`")
    md.append(f"- Inserted missing points: `{inserted}`")
    md.append(f"- After: source `{src_count_after_phase1}`, target `{tgt_count_after_phase1}`")
    md.append(f"- Count parity: `{'PASS' if parity_ok else 'FAIL'}`")
    md.append(f"- Vector transform (phase1 missing-only): `{json.dumps(transform_methods)}`")
    md.append("")
    md.append("## Phase 2 (Re-embed target in-place, resume-safe)")
    md.append(f"- Already re-embedded before retry: `{already_done}`")
    md.append(f"- Remaining before retry: `{max(0, tgt_count_after_phase1 - already_done)}`")
    md.append(f"- Processed this retry: `{processed}`")
    md.append(f"- Success this retry: `{success}`")
    md.append(f"- Failed this retry: `{failed}`")
    md.append(f"- Failure rate this retry: `{phase2['failure_rate']}`")
    md.append(f"- Fail-rate guard enabled: `{FAIL_RATE_GUARD_ENABLED}`")
    md.append(f"- Stop threshold triggered (>1%): `{stop_due_failure_rate}`")
    md.append(f"- Model coverage after retry: `{model_count_after}/{tgt_count_after_phase1}`")
    md.append(f"- Dimension handling: `{json.dumps(transform_counts)}`")
    md.append("")
    md.append("## Final Verification")
    md.append(f"- Source count unchanged: `{report['final_verification']['source_untouched_count']}`")
    md.append(f"- Target count unchanged after Phase2: `{report['final_verification']['target_count_unchanged_after_phase2']}`")
    md.append(f"- Vector dim sample check: `{dim_ok}` (sample `{sample_checked}`)")
    md.append(f"- Random sample retrieval (search): `{search_ok}`")
    md.append("")
    md.append("## Rollback Instructions")
    md.append("1. Restore target from target snapshot:")
    md.append(f"   - Collection: `{TARGET}`")
    md.append(f"   - Snapshot: `{tgt_snap}`")
    md.append("2. If needed, restore source from source snapshot (normally not needed because source is read-only in this run).")
    md.append(f"   - Collection: `{SOURCE}`")
    md.append(f"   - Snapshot: `{src_snap}`")
    md.append("")
    md.append(f"## Final Status: `{report['status']}`")

    md_text = "\n".join(md)
    report["markdown"] = md_text

    # Jira
    jira_key = None
    jira_link = None
    jira_comment_id = None
    try:
        if JIRA_ISSUE_KEY:
            creds = load_jira_creds()
            jira_key = JIRA_ISSUE_KEY
            jira_link = f"{creds['domain']}/browse/{jira_key}"
        else:
            summary = f"[Board 271] Delta sync {SOURCE}->{TARGET} + re-embed qwen3-embedding:4b ({report['status']})"
            jira_key, jira_link = jira_create_issue(summary, md_text)

        # add detailed json excerpt comment
        comment_text = json.dumps({
            "pre": report.get("pre"),
            "resume": report.get("resume"),
            "phase1": report["phase1"],
            "phase2": report["phase2"],
            "final_verification": report["final_verification"],
            "snapshots": report["snapshots"],
        }, ensure_ascii=False, indent=2)
        jira_comment_id = jira_add_comment(jira_key, comment_text)
        report["jira"] = {"key": jira_key, "link": jira_link, "comment_id": jira_comment_id}
    except Exception as e:
        report["jira"] = {"error": str(e)}

    out_json = os.path.join(OUT_DIR, f"delta-reembed-report-{snap_ts}.json")
    out_md = os.path.join(OUT_DIR, f"delta-reembed-report-{snap_ts}.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(json.dumps({
        "status": report["status"],
        "jira": report.get("jira"),
        "report_json": out_json,
        "report_md": out_md,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
