"""Run the paired LoCoMo retrieval-reflection experiment against frozen memory."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
from threading import local
from time import perf_counter
from types import ModuleType
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "7da777f56a15db81bb261d296c89cad5915e8d67"
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
EMBEDDING_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
CONVERSATIONS = ("conv-26", "conv-42", "conv-43")
PILOT_INDICES = (0, 1, 2, 3, 4, 5, 7, 14, 22, 82, 83, 84)
SEED = 5660
ROUNDS = 3
WORKERS = 3


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze_questions(raw):
    rng = random.Random(SEED)
    manifest = {}
    for conversation in CONVERSATIONS:
        selected = []
        for category in range(1, 5):
            candidates = [
                index for index, qa in enumerate(raw[conversation]["qa"])
                if qa["category"] == category and qa.get("answer") is not None
                and str(qa["answer"]).strip()
                and (conversation != "conv-26" or index not in PILOT_INDICES)
            ]
            selected.extend(rng.sample(candidates, 10))
        manifest[conversation] = sorted(selected)
    return manifest


def load_upstream(key, embedding):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(REPO))
    config = ModuleType("config")
    config.__dict__.update(runpy.run_path(str(REPO / "config.py.example")))
    config.OPENAI_API_KEY = key
    config.OPENAI_BASE_URL = BASE_URL
    config.LLM_MODEL = MODEL
    config.EMBEDDING_MODEL = str(embedding.resolve())
    sys.modules["config"] = config

    import nltk
    import torch

    torch.set_num_threads(4)
    with patch.object(nltk, "download"), patch(
        "sentence_transformers.SentenceTransformer", return_value=None
    ):
        import test_locomo10 as benchmark
    # The upstream F1 is independent of the auxiliary model-based metrics.
    benchmark.calculate_rouge_scores = lambda *_: {}
    benchmark.calculate_bleu_scores = lambda *_: {}
    benchmark.calculate_bert_scores = lambda *_: {}
    benchmark.calculate_meteor_score = lambda *_: None
    return benchmark


def install_telemetry(system):
    """Observe the upstream pipeline without changing decisions or results."""
    active = local()
    llm = system.llm_client
    retriever = system.hybrid_retriever
    create = llm.client.chat.completions.create
    chat = llm.chat_completion
    parse = llm.extract_json
    merge = retriever._merge_and_deduplicate_entries
    completeness = retriever._analyze_information_completeness
    supplemental = retriever._generate_missing_info_queries
    keyword = retriever._keyword_search
    structured = retriever._structured_search

    def request_hook(_request):
        call = getattr(active, "request_call", None)
        if call is not None:
            call["http_attempts"].append(None)

    def response_hook(response):
        call = getattr(active, "request_call", None)
        if call is not None:
            call["http_attempts"][-1] = response.status_code

    http_client = llm.client._client
    http_client.event_hooks["request"].append(request_hook)
    http_client.event_hooks["response"].append(response_hook)

    def create_with_usage(**kwargs):
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "thinking": {"type": "disabled"}}
        if kwargs.get("stream"):
            kwargs["stream_options"] = {"include_usage": True}
        trace = getattr(active, "trace", None)
        if trace is None:
            return create(**kwargs)
        call = {
            "stage": active.stage,
            "logical_call": trace["logical_calls"],
            "http_attempts": [],
            "usage": None,
        }
        trace["api_calls"].append(call)
        active.request_call = call
        try:
            response = create(**kwargs)
        finally:
            active.request_call = None

        def record(chunk):
            if getattr(chunk, "usage", None) is not None:
                call["usage"] = chunk.usage.model_dump()
                call["model"] = chunk.model

        if not kwargs.get("stream"):
            record(response)
            return response

        def stream():
            with response:
                for chunk in response:
                    record(chunk)
                    yield chunk

        return stream()

    def observed_chat(*args, **kwargs):
        trace = getattr(active, "trace", None)
        if trace is not None:
            trace["logical_calls"] += 1
        return chat(*args, **kwargs)

    def observed_parse(*args, **kwargs):
        result = parse(*args, **kwargs)
        check = getattr(active, "reflection_check", None)
        if check is not None:
            if not isinstance(result, dict) or "assessment" not in result:
                check["source"] = "missing_assessment"
            elif result["assessment"] in ("complete", "incomplete"):
                check["source"] = "model"
            else:
                check["source"] = "unexpected_assessment"
            if isinstance(result, dict):
                check["reported_coverage"] = result.get("coverage_percentage")
        return result

    def observed_merge(entries):
        merged = merge(entries)
        trace = getattr(active, "trace", None)
        if trace is not None and trace["initial_memory_ids"] is None:
            trace["initial_memory_ids"] = [entry.entry_id for entry in merged]
        return merged

    def observed_completeness(*args, **kwargs):
        check = {"source": "call_or_parse_error"}
        active.trace["reflection_checks"].append(check)
        previous = active.stage
        active.stage = "completeness_check"
        active.reflection_check = check
        try:
            outcome = completeness(*args, **kwargs)
            check["decision"] = outcome
            return outcome
        finally:
            active.reflection_check = None
            active.stage = previous

    def observed_supplemental(*args, **kwargs):
        previous = active.stage
        active.stage = "additional_query_generation"
        try:
            queries = supplemental(*args, **kwargs)
            active.trace["supplemental_queries"].append(queries)
            return queries
        finally:
            active.stage = previous

    def observed_channel(name, search):
        def observe(*args, **kwargs):
            result = search(*args, **kwargs)
            trace = getattr(active, "trace", None)
            if trace is not None:
                trace[name] = len(result)
            return result
        return observe

    llm.client.chat.completions.create = create_with_usage
    llm.chat_completion = observed_chat
    llm.extract_json = observed_parse
    retriever._merge_and_deduplicate_entries = observed_merge
    retriever._analyze_information_completeness = observed_completeness
    retriever._generate_missing_info_queries = observed_supplemental
    retriever._keyword_search = observed_channel("keyword_matches", keyword)
    retriever._structured_search = observed_channel("structured_matches", structured)
    return active


def fingerprint(table):
    rows = table.to_arrow().to_pylist()
    rows.sort(key=lambda row: row["entry_id"])
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest(), rows


def check_index(store, probe):
    matches = store.keyword_search([probe], top_k=5)
    if not matches:
        raise RuntimeError(f"Tantivy index probe failed for {probe!r}")
    return len(matches)


def prepare_snapshot(benchmark, root, conversation, sample, pilot_snapshot):
    path = root / "snapshots" / conversation
    database = path / "lancedb"
    metadata = path / "snapshot.json"
    if metadata.is_file():
        return database, json.loads(metadata.read_text(encoding="utf-8"))
    if database.exists():
        raise RuntimeError(f"Incomplete snapshot at {database}; inspect before retrying")
    path.mkdir(parents=True, exist_ok=True)
    reuse = conversation == "conv-26" and pilot_snapshot is not None
    if reuse:
        shutil.copytree(pilot_snapshot, database)
    system = benchmark.SimpleMemSystem(db_path=str(database), clear_db=False)
    install_telemetry(system)
    tester = benchmark.LoCoMoTester(system, "")
    dialogue = tester.convert_to_dialogues(sample)
    store = system.vector_store
    build_seconds = None
    if not reuse:
        start = perf_counter()
        system.add_dialogues(dialogue)
        system.finalize()
        if not store._fts_initialized:
            store._init_fts_index()
        build_seconds = perf_counter() - start
    probe = sample.conversation.speaker_a.split()[0]
    matches = check_index(store, probe)
    snapshot_hash, rows = fingerprint(store.table)
    entries = [
        {key: value for key, value in row.items() if key != "vector"}
        for row in rows
    ]
    info = {
        "sha256": snapshot_hash, "rows": len(rows), "turns": len(dialogue),
        "reused_pilot_snapshot": reuse, "build_seconds": build_seconds,
        "keyword_probe": probe, "keyword_probe_matches": matches,
    }
    save(path / "memories.json", entries)
    save(metadata, info)
    print(f"SNAPSHOT {conversation} rows={len(rows)} fts={matches} reused={reuse}", flush=True)
    return database, info


def run_question(benchmark, system, active, index, qa):
    trace = {
        "logical_calls": 0, "api_calls": [], "initial_memory_ids": None,
        "reflection_checks": [], "supplemental_queries": [],
    }
    active.trace = trace
    active.stage = "initial_retrieval"
    start = perf_counter()
    retrieval_seconds = None
    answer = None
    contexts = []
    failure = None
    try:
        contexts = system.hybrid_retriever.retrieve(qa.question)
        retrieval_seconds = perf_counter() - start
        active.stage = "answer"
        answer = system.answer_generator.generate_answer(qa.question, contexts)
        if not answer or answer == "Failed to generate answer":
            failure = "no_scorable_answer"
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        elapsed = perf_counter() - start
        active.trace = None
    final_ids = [entry.entry_id for entry in contexts]
    initial_ids = set(trace["initial_memory_ids"] or ())
    score = 0.0 if failure else benchmark.calculate_metrics(answer, qa.final_answer)["f1"]
    return {
        "index": index, "category": qa.category, "question": qa.question,
        "reference": qa.final_answer, "evidence": qa.evidence,
        "answer": answer, "failure": failure, "f1": score,
        "retrieval_seconds": retrieval_seconds,
        "answer_seconds": None if retrieval_seconds is None else elapsed - retrieval_seconds,
        "total_seconds": elapsed, "final_memory_ids": final_ids,
        "added_memory_ids": [item for item in final_ids if item not in initial_ids],
        **trace,
    }


def run_group(benchmark, root, group, database, snapshot, sample, indices, reflection):
    group.mkdir(parents=True, exist_ok=True)
    meta = group / "group.json"
    expected = {"reflection": reflection, "indices": list(indices), "snapshot_sha256": snapshot["sha256"]}
    if meta.is_file() and json.loads(meta.read_text(encoding="utf-8")) != expected:
        raise RuntimeError(f"Existing run configuration differs at {group}")
    save(meta, expected)
    pending = [index for index in indices if not (group / f"qa-{index}.json").is_file()]
    if not pending and (group / "complete.json").is_file():
        return
    system = benchmark.SimpleMemSystem(
        db_path=str(database), clear_db=False, enable_planning=True, enable_reflection=reflection
    )
    store = system.vector_store
    check_index(store, snapshot["keyword_probe"])
    if fingerprint(store.table)[0] != snapshot["sha256"]:
        raise RuntimeError("Memory snapshot differs before question evaluation")
    active = install_telemetry(system)
    print(f"GROUP {group.relative_to(root)} START {len(pending)} pending", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as workers:
        futures = [
            workers.submit(run_question, benchmark, system, active, index, sample.qa[index])
            for index in pending
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            save(group / f"qa-{row['index']}.json", row)
            if completed % 10 == 0 or completed == len(pending):
                print(f"GROUP {group.relative_to(root)} PROGRESS {completed}/{len(pending)}", flush=True)
    if fingerprint(store.table)[0] != snapshot["sha256"]:
        raise RuntimeError("Memory snapshot changed during question evaluation")
    save(group / "complete.json", expected)


def validate_pilot(root):
    for name in ("enabled", "disabled"):
        group = root / "pilot" / name
        if not (group / "complete.json").is_file():
            raise RuntimeError(f"Pilot group {name} did not complete")
        rows = [json.loads((group / f"qa-{index}.json").read_text(encoding="utf-8")) for index in PILOT_INDICES]
        if any(
            row["failure"] or not row["initial_memory_ids"] or "keyword_matches" not in row
            or "structured_matches" not in row or not row["api_calls"]
            or any(call["usage"] is None or not call["http_attempts"] for call in row["api_calls"])
            or not any(call["stage"] == "answer" for call in row["api_calls"])
            for row in rows
        ):
            raise RuntimeError(f"Pilot group {name} failed question or usage checks")
        if name == "disabled" and any(row["reflection_checks"] or row["supplemental_queries"] for row in rows):
            raise RuntimeError("Disabled reflection ran during pilot")
        if name == "enabled" and (
            not any(row["reflection_checks"] for row in rows)
            or not any(any(row["supplemental_queries"]) for row in rows)
        ):
            raise RuntimeError("Pilot did not exercise both checking and supplemental search")
    print("PILOT PASSED 24/24", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--reuse-pilot-snapshot", type=Path)
    parser.add_argument("--mode", choices=("manifest", "pilot", "formal", "all"), default="all")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if digest(args.dataset) != DATA_SHA256:
        raise SystemExit("LoCoMo dataset SHA-256 differs from the pinned version")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    raw = {item["sample_id"]: item for item in dataset}
    manifest = freeze_questions(raw)
    frozen = {"seed": SEED, "data_sha256": DATA_SHA256, "questions": manifest}
    manifest_path = root / "questions.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != frozen:
        raise SystemExit("Existing question manifest differs")
    save(manifest_path, frozen)
    print(f"MANIFEST {sum(map(len, manifest.values()))} questions", flush=True)
    if args.mode == "manifest":
        return

    from dotenv import dotenv_values
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and args.key_file:
        key = dotenv_values(args.key_file).get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY or pass a local --key-file")
    if digest(args.embedding / "model.safetensors") != EMBEDDING_SHA256:
        raise SystemExit("Qwen3 embedding SHA-256 differs from the pinned version")
    tracked = ("main.py", "config.py.example", "core", "database", "models", "utils", "test_locomo10.py")
    subprocess.run(["git", "diff", "--quiet", SOURCE_COMMIT, "--", *tracked], cwd=REPO, check=True)
    settings = {
        "simplemem_commit": SOURCE_COMMIT, "model": MODEL, "thinking": "disabled",
        "base_url": BASE_URL, "embedding_sha256": EMBEDDING_SHA256,
        "config_sha256": digest(REPO / "config.py.example"),
        "runner_sha256": digest(Path(__file__)), "workers": WORKERS,
        "rounds": ROUNDS, "reflection_round_limit": 2,
    }
    settings_path = root / "settings.json"
    if settings_path.is_file() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise SystemExit("Existing run settings differ")
    save(settings_path, settings)
    benchmark = load_upstream(key, args.embedding)
    parsed = dict(zip(raw, benchmark.load_locomo_dataset(args.dataset.resolve()), strict=True))
    if args.mode in {"pilot", "all"}:
        sample = parsed["conv-26"]
        database, snapshot = prepare_snapshot(
            benchmark, root, "conv-26", sample, args.reuse_pilot_snapshot
        )
        for name, reflection in (("enabled", True), ("disabled", False)):
            run_group(benchmark, root, root / "pilot" / name, database, snapshot, sample, PILOT_INDICES, reflection)
    validate_pilot(root)
    if args.mode == "pilot":
        return

    snapshots = {
        conversation: prepare_snapshot(
            benchmark, root, conversation, parsed[conversation], args.reuse_pilot_snapshot
        )
        for conversation in CONVERSATIONS
    }
    for number in range(1, ROUNDS + 1):
        for index, conversation in enumerate(CONVERSATIONS):
            order = ("enabled", "disabled") if (number + index) % 2 else ("disabled", "enabled")
            database, snapshot = snapshots[conversation]
            for name in order:
                group = root / f"round-{number}" / conversation / name
                run_group(
                    benchmark, root, group, database, snapshot,
                    parsed[conversation], manifest[conversation], name == "enabled",
                )
    print("FORMAL RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
