"""Summarize complete paired reflection runs using the frozen question manifest."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

from run import CONVERSATIONS, ROUNDS, save


STAGES = ("initial_retrieval", "completeness_check", "additional_query_generation", "answer")
GROUPS = ("enabled", "disabled")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pair_rows(root, manifest, number):
    for conversation in CONVERSATIONS:
        base = root / f"round-{number}" / conversation
        groups = [base / name for name in GROUPS]
        if not all((group / "complete.json").is_file() for group in groups):
            raise ValueError(f"Incomplete round {number}, {conversation}")
        metadata = [read(group / "complete.json") for group in groups]
        if metadata[0]["snapshot_sha256"] != metadata[1]["snapshot_sha256"]:
            raise ValueError(f"Snapshot mismatch in {base}")
        if [item["reflection"] for item in metadata] != [True, False]:
            raise ValueError(f"Reflection flags differ from design in {base}")
        for index in manifest[conversation]:
            enabled, disabled = (read(group / f"qa-{index}.json") for group in groups)
            if any(
                enabled[key] != disabled[key] for key in ("index", "category", "question", "reference")
            ) or enabled["index"] != index:
                raise ValueError(f"Question mismatch in {base}, question {index}")
            if disabled["reflection_checks"] or disabled["supplemental_queries"]:
                raise ValueError(f"Disabled reflection ran in {base}, question {index}")
            yield conversation, index, enabled, disabled


def mean_or_none(items):
    return mean(items) if items else None


def f1_summary(pairs):
    categories = defaultdict(list)
    for _, _, enabled, disabled in pairs:
        categories[enabled["category"]].append((enabled["f1"] * 100, disabled["f1"] * 100))
    result = {}
    for category, rows in sorted(categories.items()):
        baseline, ablation = map(mean, zip(*rows))
        result[category] = {
            "n": len(rows), "enabled": baseline, "disabled": ablation,
            "delta_pp": baseline - ablation,
        }
    baseline = mean_or_none([row["enabled"] for row in result.values()])
    ablation = mean_or_none([row["disabled"] for row in result.values()])
    return {
        "n": len(pairs), "enabled": baseline, "disabled": ablation,
        "delta_pp": None if baseline is None else baseline - ablation,
        "categories": result,
    }


def token_usage(row, field, stage=None):
    calls = [call for call in row["api_calls"] if stage is None or call["stage"] == stage]
    if any(call["usage"] is None or call["usage"].get(field) is None for call in calls):
        return None
    return sum(call["usage"][field] for call in calls)


def metric_values(row):
    result = {
        "retrieval_seconds": row["retrieval_seconds"],
        "answer_seconds": row["answer_seconds"],
        "total_seconds": row["total_seconds"],
        "logical_calls": row["logical_calls"],
        "http_attempts": sum(len(call["http_attempts"]) for call in row["api_calls"]),
        "prompt_tokens": token_usage(row, "prompt_tokens"),
        "completion_tokens": token_usage(row, "completion_tokens"),
        "total_tokens": token_usage(row, "total_tokens"),
    }
    for stage in STAGES:
        result[f"{stage}_http_attempts"] = sum(
            len(call["http_attempts"]) for call in row["api_calls"] if call["stage"] == stage
        )
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            result[f"{stage}_{field}"] = token_usage(row, field, stage)
    return result


def efficiency_summary(pairs):
    samples = defaultdict(list)
    for _, _, enabled, disabled in pairs:
        before, after = metric_values(enabled), metric_values(disabled)
        for name, value in before.items():
            if value is not None and after[name] is not None:
                samples[name].append((value, after[name]))
            else:
                samples.setdefault(name, [])
    result = {}
    for name, values in samples.items():
        left, right = tuple(map(mean, zip(*values))) if values else (None, None)
        result[name] = {
            "paired_n": len(values), "enabled": left, "disabled": right,
            "delta": None if left is None else left - right,
        }
    return result


def diagnostics(pairs):
    decisions = Counter()
    initial_jaccard = []
    same_order = 0
    supplemented = 0
    added_memory = 0
    failures = Counter()
    usage_missing = Counter()
    for _, _, enabled, disabled in pairs:
        a = enabled["initial_memory_ids"] or []
        b = disabled["initial_memory_ids"] or []
        union = set(a) | set(b)
        if union:
            initial_jaccard.append(len(set(a) & set(b)) / len(union))
            same_order += int(a == b)
        for check in enabled["reflection_checks"]:
            decisions[(check["source"], check.get("decision"))] += 1
        supplemented += int(any(enabled["supplemental_queries"]))
        added_memory += int(bool(enabled["added_memory_ids"]))
        for name, row in zip(GROUPS, (enabled, disabled)):
            failures[name] += int(bool(row["failure"]))
            usage_missing[name] += sum(call["usage"] is None for call in row["api_calls"])
    return {
        "failures": dict(failures), "missing_usage_calls": dict(usage_missing),
        "reflection_decisions": [
            {"source": source, "decision": decision, "count": count}
            for (source, decision), count in decisions.items()
        ],
        "questions_with_supplemental_queries": supplemented,
        "questions_with_added_memory": added_memory,
        "initial_jaccard_mean": mean_or_none(initial_jaccard),
        "initial_overlap_n": len(initial_jaccard),
        "initial_same_order": same_order,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory
    manifest = read(root / "questions.json")["questions"]
    per_round = {}
    all_pairs = []
    per_question = defaultdict(list)
    for number in range(1, ROUNDS + 1):
        pairs = list(pair_rows(root, manifest, number))
        all_pairs.extend(pairs)
        per_round[number] = {"f1": f1_summary(pairs), "efficiency": efficiency_summary(pairs)}
        for conversation, index, enabled, disabled in pairs:
            per_question[(conversation, index)].append((enabled["f1"] - disabled["f1"]) * 100)
    round_deltas = [item["f1"]["delta_pp"] for item in per_round.values()]
    question_deltas = [mean(items) for items in per_question.values()]
    summary = {
        "unique_questions": len(per_question),
        "paired_observations": len(all_pairs),
        "per_round": per_round,
        "round_delta_mean_pp": mean(round_deltas),
        "round_delta_sample_sd_pp": stdev(round_deltas),
        "f1": f1_summary(all_pairs),
        "f1_both_succeeded": f1_summary([
            pair for pair in all_pairs if not (pair[2]["failure"] or pair[3]["failure"])
        ]),
        "per_conversation_f1": {
            conversation: f1_summary([pair for pair in all_pairs if pair[0] == conversation])
            for conversation in CONVERSATIONS
        },
        "question_mean_deltas": {
            "improved": sum(item > 1e-9 for item in question_deltas),
            "degraded": sum(item < -1e-9 for item in question_deltas),
            "unchanged": sum(abs(item) <= 1e-9 for item in question_deltas),
        },
        "efficiency": efficiency_summary(all_pairs),
        "diagnostics": diagnostics(all_pairs),
    }
    save(root / "summary.json", summary)
    print(json.dumps({
        "f1": summary["f1"], "round_delta_sample_sd_pp": summary["round_delta_sample_sd_pp"],
        "efficiency": {
            key: summary["efficiency"][key]
            for key in ("total_seconds", "logical_calls", "http_attempts", "total_tokens", "answer_prompt_tokens")
        },
        "diagnostics": summary["diagnostics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
