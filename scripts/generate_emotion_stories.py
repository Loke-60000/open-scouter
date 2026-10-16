#!/usr/bin/env python3
"""Generate the full paper-scale emotion-story dataset (resumable, runs for weeks).

Reproduces the data recipe from "Emotion Concepts and their Function in a Large
Language Model" (Sofroniew, Lindsey et al., arXiv:2604.07729):

    171 emotions x 100 topics x 12 one-paragraph stories  (~205k stories)

For each (emotion, topic) the model writes 12 short paragraph stories in which a
character clearly experiences the emotion *without naming it*. Output feeds the
emotion-vector builder (`scouter-emotions` / `build_emotion_vectors`).

This is a multi-day job, so it is built to be **interrupted and resumed freely**:

  * Work is sharded per emotion: ``inputs/emotion_stories/<emotion>.json``.
  * Each shard records which topics are done; a re-run skips completed work.
  * Shards are written atomically (tmp + os.replace) so a kill never corrupts one.
  * Failed calls are retried with backoff; a permanently failing (emotion, topic)
    is simply left for the next run instead of crashing the job.

Run it, kill it, run it again — it converges to complete. When every shard is
full, ``--assemble`` merges them into one ``inputs/emotions.json``.

    # generate (resume-safe; re-run as many times as needed)
    python scripts/generate_emotion_stories.py --workers 6

    # check progress at any time
    python scripts/generate_emotion_stories.py --status

    # once complete, build the final emotions.json
    python scripts/generate_emotion_stories.py --assemble --out inputs/emotions.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx

from generate_inputs import EMOTION_NAMES, _slug  # noqa: E402  (scripts/ on sys.path)

DEFAULT_MODEL = "deepseek-v4-flash:cloud"
TOPICS_FILE = "_topics.json"


def salvage_stories(text: str, *, min_words: int = 8) -> list[str]:
    """Recover complete story strings from a JSON response, even if truncated.

    The model is asked for ``{"stories": [...]}``. If the array is cut off
    mid-token-budget we still keep every complete quoted string, so a partial
    response is never wasted.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        items = obj.get("stories", []) if isinstance(obj, dict) else obj
        if isinstance(items, list):
            return [str(s).strip() for s in items if len(str(s).split()) >= min_words]
    except json.JSONDecodeError:
        pass
    m = re.search(r'"stories"\s*:\s*\[(.*)', text, re.S)
    body = m.group(1) if m else text
    out = []
    for raw in re.findall(r'"((?:[^"\\]|\\.)*)"', body):
        try:
            s = json.loads('"' + raw + '"')
        except json.JSONDecodeError:
            continue
        if len(s.split()) >= min_words:
            out.append(s.strip())
    return out


def ollama_call(model: str, prompt: str, *, host: str, num_predict: int, timeout: float) -> str:
    resp = httpx.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0.9, "num_predict": num_predict},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def story_prompt(emotion: str, topic: str, n: int) -> str:
    pretty = emotion.replace("_", " ")
    return (
        f"Write {n} short stories, one paragraph each (4-6 sentences), on the topic "
        f'of "{topic}". In every story a character clearly experiences the emotion of '
        f"being {pretty}, shown through their actions, body, and thoughts, WITHOUT ever "
        f'using the word "{pretty}" or an obvious synonym. Vary the characters and '
        f'situations across the {n} stories. Return JSON: '
        '{"stories": ["paragraph one", "paragraph two", ...]} '
        f"with exactly {n} strings."
    )


class StoryGenerator:
    """Resumable generator over the (emotion, topic) grid."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.emotions = [_slug(e) for e in EMOTION_NAMES]
        if args.limit_emotions:
            self.emotions = self.emotions[: args.limit_emotions]
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._done = 0
        self._done_lock = threading.Lock()
        self._start = 0.0

    # ---- shard I/O (atomic, per-emotion) ----------------------------------
    def _shard_path(self, emotion: str) -> Path:
        return self.out_dir / f"{emotion}.json"

    def _load_shard(self, emotion: str) -> dict[str, list[str]]:
        path = self._shard_path(emotion)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            return dict(data.get("topics", {}))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_shard(self, emotion: str, topics: dict[str, list[str]]) -> None:
        path = self._shard_path(emotion)
        tmp = path.with_suffix(".json.tmp")
        payload = {"name": emotion, "topics": topics}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
        os.replace(tmp, path)  # atomic: a kill mid-write leaves the old shard intact

    # ---- topics -----------------------------------------------------------
    def ensure_topics(self) -> list[str]:
        path = self.out_dir / TOPICS_FILE
        if path.exists():
            topics = json.loads(path.read_text())
            if len(topics) >= self.args.topics:
                return topics[: self.args.topics]
        prompt = (
            f"List {self.args.topics} diverse, concrete story topics or settings "
            "(everyday life, work, travel, relationships, nature, history, etc.) "
            "suitable as backdrops for short character stories. Keep each to a few "
            'words. Return JSON: {"topics": ["...", "..."]}.'
        )
        for _ in range(5):
            txt = ollama_call(
                self.args.model, prompt, host=self.args.host,
                num_predict=2048, timeout=self.args.timeout,
            )
            try:
                obj = json.loads(txt)
                topics = obj.get("topics", obj) if isinstance(obj, dict) else obj
            except json.JSONDecodeError:
                topics = salvage_stories(txt, min_words=1)
            topics = [str(t).strip() for t in topics if str(t).strip()]
            if len(topics) >= self.args.topics:
                path.write_text(json.dumps(topics, ensure_ascii=False, indent=1) + "\n")
                return topics[: self.args.topics]
        raise RuntimeError("could not generate enough topics")

    # ---- the work grid ----------------------------------------------------
    def pending(self, topics: list[str]) -> list[tuple[str, str]]:
        """All (emotion, topic) pairs not yet satisfied."""
        need = self.args.stories_per_topic
        work = []
        for emo in self.emotions:
            have = self._load_shard(emo)
            for topic in topics:
                if len(have.get(topic, [])) < need:
                    work.append((emo, topic))
        return work

    def _generate_pair(self, emotion: str, topic: str) -> Optional[list[str]]:
        need = self.args.stories_per_topic
        prompt = story_prompt(emotion, topic, need)
        delay = 2.0
        for attempt in range(self.args.retries):
            try:
                txt = ollama_call(
                    self.args.model, prompt, host=self.args.host,
                    num_predict=self.args.num_predict, timeout=self.args.timeout,
                )
                stories = salvage_stories(txt)
                if len(stories) >= max(1, need // 2):
                    return stories[:need]
            except (httpx.HTTPError, OSError):
                pass
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        return None

    def _store_pair(self, emotion: str, topic: str, stories: list[str]) -> None:
        with self._locks[emotion]:
            have = self._load_shard(emotion)
            have[topic] = stories
            self._write_shard(emotion, have)

    def _worker(self, emotion: str, topic: str, total: int) -> None:
        stories = self._generate_pair(emotion, topic)
        if stories:
            self._store_pair(emotion, topic, stories)
        with self._done_lock:
            self._done += 1
            n = self._done
        if n % 20 == 0 or n == total:
            elapsed = time.time() - self._start
            rate = n / elapsed if elapsed else 0
            eta = (total - n) / rate if rate else 0
            ok = "" if stories else " (failed, will retry next run)"
            print(
                f"[{n}/{total}] {emotion}/{topic[:30]!r} "
                f"{rate*60:.0f}/min ETA {eta/3600:.1f}h{ok}",
                flush=True,
            )

    def run(self) -> None:
        topics = self.ensure_topics()
        work = self.pending(topics)
        total = len(work)
        if not total:
            print("nothing to do — all shards complete. Run --assemble to merge.")
            return
        print(
            f"generating {self.args.stories_per_topic} stories for {total} "
            f"(emotion, topic) pairs with {self.args.workers} workers "
            f"({self.args.model}). Resume-safe: kill and re-run any time.",
            flush=True,
        )
        self._start = time.time()
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = [pool.submit(self._worker, e, t, total) for e, t in work]
            try:
                for _ in as_completed(futures):
                    pass
            except KeyboardInterrupt:
                print("\ninterrupted — completed work is saved; re-run to resume.", flush=True)
                for f in futures:
                    f.cancel()

    # ---- reporting / assembly --------------------------------------------
    def status(self) -> None:
        topics_path = self.out_dir / TOPICS_FILE
        topics = json.loads(topics_path.read_text()) if topics_path.exists() else []
        topics = topics[: self.args.topics]  # grid only uses the first --topics
        need = self.args.stories_per_topic
        n_topics = len(topics) or self.args.topics
        total_pairs = len(self.emotions) * n_topics
        done_pairs = 0
        total_stories = 0
        complete_emotions = 0
        for emo in self.emotions:
            have = self._load_shard(emo)
            full = sum(1 for t in topics if len(have.get(t, [])) >= need) if topics else 0
            done_pairs += full
            total_stories += sum(len(v) for v in have.values())
            if topics and full == n_topics:
                complete_emotions += 1
        pct = 100 * done_pairs / total_pairs if total_pairs else 0
        print(f"topics:            {len(topics)}/{self.args.topics}")
        print(f"(emotion,topic):   {done_pairs}/{total_pairs} pairs done ({pct:.1f}%)")
        print(f"emotions complete: {complete_emotions}/{len(self.emotions)}")
        print(f"stories on disk:   {total_stories}")

    def assemble(self, out_path: str) -> None:
        records = []
        total = 0
        for emo in self.emotions:
            have = self._load_shard(emo)
            stories: list[str] = []
            for v in have.values():
                stories.extend(v)
            if stories:
                records.append({"name": emo, "scenarios": stories})
                total += len(stories)
        Path(out_path).write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {out_path}: {len(records)} emotions, {total} stories")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--out-dir", default="inputs/emotion_stories")
    ap.add_argument("--topics", type=int, default=100)
    ap.add_argument("--limit-emotions", type=int, default=0, help="cap emotions (0 = all 171)")
    ap.add_argument("--stories-per-topic", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--num-predict", type=int, default=4096)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--assemble", action="store_true", help="merge shards into --out and exit")
    ap.add_argument("--out", default="inputs/emotions.json", help="assemble target")
    args = ap.parse_args()

    gen = StoryGenerator(args)
    if args.status:
        gen.status()
    elif args.assemble:
        gen.assemble(args.out)
    else:
        gen.run()


if __name__ == "__main__":
    main()
