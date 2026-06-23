# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def get_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f_score(precision: float, recall: float) -> float:
    return safe_divide(2.0 * precision * recall, precision + recall)


def corpus_bleu(samples: list[tuple[list[str], list[str]]], max_order: int) -> float:
    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    pred_length = 0
    ref_length = 0

    for pred, ref in samples:
        pred_length += len(pred)
        ref_length += len(ref)
        for order in range(1, max_order + 1):
            pred_ngrams = get_ngrams(pred, order)
            ref_ngrams = get_ngrams(ref, order)
            overlap = pred_ngrams & ref_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_by_order[order - 1] += max(len(pred) - order + 1, 0)

    precisions = []
    for matches, possible in zip(matches_by_order, possible_by_order):
        if possible == 0:
            precisions.append(0.0)
        elif matches == 0:
            precisions.append(1.0 / (possible * 2.0))
        else:
            precisions.append(matches / possible)

    if min(precisions) <= 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(precision) for precision in precisions) / max_order)

    brevity_penalty = 1.0
    if pred_length == 0:
        brevity_penalty = 0.0
    elif pred_length < ref_length:
        brevity_penalty = math.exp(1.0 - ref_length / pred_length)

    return 100.0 * geo_mean * brevity_penalty


def meteor_score(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0

    ref_positions: dict[str, list[int]] = defaultdict(list)
    for idx, token in enumerate(ref):
        ref_positions[token].append(idx)

    used_ref_positions = set()
    matched_pairs = []
    for pred_idx, token in enumerate(pred):
        for ref_idx in ref_positions.get(token, []):
            if ref_idx not in used_ref_positions:
                used_ref_positions.add(ref_idx)
                matched_pairs.append((pred_idx, ref_idx))
                break

    matches = len(matched_pairs)
    if matches == 0:
        return 0.0

    precision = matches / len(pred)
    recall = matches / len(ref)
    f_mean = safe_divide(10.0 * precision * recall, recall + 9.0 * precision)

    matched_pairs.sort()
    chunks = 1
    for idx in range(1, len(matched_pairs)):
        prev_pred, prev_ref = matched_pairs[idx - 1]
        cur_pred, cur_ref = matched_pairs[idx]
        if cur_pred != prev_pred + 1 or cur_ref != prev_ref + 1:
            chunks += 1

    penalty = 0.5 * (chunks / matches) ** 3
    return 100.0 * f_mean * (1.0 - penalty)


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for col, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[col - 1] + 1)
            else:
                current.append(max(previous[col], current[-1]))

        previous = current

    return previous[-1]


def rouge_n_score(pred: list[str], ref: list[str], order: int) -> float:
    pred_ngrams = get_ngrams(pred, order)
    ref_ngrams = get_ngrams(ref, order)
    if not pred_ngrams or not ref_ngrams:
        return 0.0

    overlap = sum((pred_ngrams & ref_ngrams).values())
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    return 100.0 * f_score(precision, recall)


def rouge_l_score(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0

    lcs = lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 100.0 * f_score(precision, recall)


def cider_document_frequency(samples: list[tuple[list[str], list[str]]]) -> list[Counter[tuple[str, ...]]]:
    dfs = [Counter() for _ in range(4)]
    for _, ref in samples:
        for order in range(1, 5):
            dfs[order - 1].update(set(get_ngrams(ref, order)))

    return dfs


def cider_vector(
    tokens: list[str],
    order: int,
    document_frequency: Counter[tuple[str, ...]],
    document_count: int,
) -> dict[tuple[str, ...], float]:
    counts = get_ngrams(tokens, order)
    total = sum(counts.values())
    if total == 0:
        return {}

    vector = {}
    for ngram, count in counts.items():
        tf = count / total
        # Common CIDEr implementations use reference-side document frequency.
        # Avoid add-one smoothing here: high-frequency ngrams should go to zero weight, not negative weight.
        idf = math.log(max(1, document_count) / max(1, document_frequency.get(ngram, 0)))
        vector[ngram] = tf * idf

    return vector


def cosine_similarity(left: dict[tuple[str, ...], float], right: dict[tuple[str, ...], float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return safe_divide(dot, left_norm * right_norm)


def cider_score(samples: list[tuple[list[str], list[str]]]) -> float:
    if not samples:
        return 0.0

    document_frequency = cider_document_frequency(samples)
    document_count = len(samples)
    scores = []
    for pred, ref in samples:
        order_scores = []
        for order in range(1, 5):
            pred_vector = cider_vector(pred, order, document_frequency[order - 1], document_count)
            ref_vector = cider_vector(ref, order, document_frequency[order - 1], document_count)
            order_scores.append(cosine_similarity(pred_vector, ref_vector))

        scores.append(10.0 * sum(order_scores) / 4.0)

    return sum(scores) / len(scores)


def load_predictions(path: Path) -> list[tuple[list[str], list[str]]]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if "predict" not in item or "label" not in item:
                raise ValueError(f"Line {line_number} must contain `predict` and `label` fields.")

            samples.append((tokenize(item["predict"]), tokenize(item["label"])))

    if not samples:
        raise ValueError(f"No valid prediction samples found in {path}.")

    return samples


def evaluate(path: Path) -> dict[str, float | int]:
    samples = load_predictions(path)
    meteor_scores = [meteor_score(pred, ref) for pred, ref in samples]
    rouge_1_scores = [rouge_n_score(pred, ref, 1) for pred, ref in samples]
    rouge_2_scores = [rouge_n_score(pred, ref, 2) for pred, ref in samples]
    rouge_l_scores = [rouge_l_score(pred, ref) for pred, ref in samples]
    pred_lengths = [len(pred) for pred, _ in samples]
    ref_lengths = [len(ref) for _, ref in samples]

    return {
        "num_samples": len(samples),
        "bleu_1": corpus_bleu(samples, 1),
        "bleu_2": corpus_bleu(samples, 2),
        "bleu_3": corpus_bleu(samples, 3),
        "bleu_4": corpus_bleu(samples, 4),
        "meteor": sum(meteor_scores) / len(meteor_scores),
        "rouge_1": sum(rouge_1_scores) / len(rouge_1_scores),
        "rouge_2": sum(rouge_2_scores) / len(rouge_2_scores),
        "rouge_l": sum(rouge_l_scores) / len(rouge_l_scores),
        "cider": cider_score(samples),
        "exact_match": 100.0 * sum(pred == ref for pred, ref in samples) / len(samples),
        "empty_predictions": sum(not pred for pred, _ in samples),
        "avg_prediction_length": sum(pred_lengths) / len(pred_lengths),
        "avg_reference_length": sum(ref_lengths) / len(ref_lengths),
        "length_ratio": safe_divide(sum(pred_lengths), sum(ref_lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate English caption predictions from generated_predictions.jsonl.")
    parser.add_argument("prediction_file", type=Path, help="Path to generated_predictions.jsonl.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save metric JSON. Defaults to predictions_score_en.json next to the prediction file.",
    )
    args = parser.parse_args()

    metrics = evaluate(args.prediction_file)
    output_path = args.output or args.prediction_file.with_name("predictions_score_en.json")
    output_path.write_text(json.dumps(metrics, indent=4) + "\n", encoding="utf-8")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nScore file saved to {output_path}")


if __name__ == "__main__":
    main()
