#!/usr/bin/env python3
"""Convert medical imaging datasets to ShareGPT OpenAI format for LlamaFactory."""

import argparse
import csv
import json
import random
import re
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

ROCO_SPLITS = ("train", "validation", "test")

ROCO_INSTRUCTION_CATEGORIES = {
    "caption": [
        "Describe this radiology image.",
        "Provide a caption for this image.",
        "Generate a radiology report summary.",
        "Write a description of this medical image.",
    ],
    "finding": [
        "What findings are present in this image?",
        "Describe the radiological findings.",
        "What abnormalities can be observed?",
        "Summarize the imaging findings.",
    ],
    "interpretation": [
        "Interpret this medical image.",
        "What does this scan demonstrate?",
        "What is shown in this study?",
        "Analyze this radiological image.",
    ],
    "report_generation": [
        "Generate a radiology report for this image.",
        "Write the findings section of a radiology report.",
        "Produce a diagnostic description of this image.",
    ],
}

ROCO_CATEGORY_WEIGHTS = {
    "caption": 1.5,
    "finding": 1.5,
    "interpretation": 1.0,
    "report_generation": 0.1,
}

VQA_RAD_SPLIT_FILES = {
    "train": "train-00000-of-00001-eb8844602202be60.parquet",
    "test": "test-00000-of-00001-e5bc3d208bb4deeb.parquet",
}

PMC_VQA_VARIANTS = {
    "v1": {
        "splits": {
            "train": "train.csv",
            "test": "test.csv",
            "test_clean": "test_clean.csv",
        },
        "images_zip": "images.zip",
        "images_subdir": "images",
        "prefix": "pmc_vqa",
    },
    "v2": {
        "splits": {
            "train": "train_2.csv",
            "test": "test_2.csv",
        },
        "images_zip": "images_2.zip",
        "images_subdir": "figures",
        "prefix": "pmc_vqa2",
    },
}

RADIOLOGY_KEYWORDS = re.compile(
    r"\b("
    r"x-?ray|radiograph|mammograph|computed tomography|\bct\b|mri|magnetic resonance|"
    r"ultrasound|sonograph|\bpet\b|\bspect\b|fluoroscop|angiograph|tomosynthesis|"
    r"radiolog|scintigraph|dexa|\bdxa\b|echocardiograph|doppler|cbct|cone beam|"
    r"perfusion|diffusion weighted|time of flight|\bdsa\b|scintimammograph|mammography"
    r")\b",
    re.IGNORECASE,
)
NON_RADIOLOGY_KEYWORDS = re.compile(
    r"\b("
    r"histolog|patholog|biopsy|h&e|hematoxylin|eosin|immunohistochem|\bihc\b|"
    r"microtome|paraffin section|tissue section|masson|trichrome|"
    r"microscop|electron microscop|\btem\b|\bsem\b|confocal|clsm|bright.?field|"
    r"phase.?contrast|immunofluorescen|mitochondri|golgi complex|cell nuclei|"
    r"in vitro|405nm excitation|two.?photon|widefield"
    r")\b",
    re.IGNORECASE,
)


def save_rgb_image(image, image_path: Path, labels_only: bool) -> None:
    if not labels_only or not image_path.exists():
        image.convert("RGB").save(image_path, format="JPEG", quality=95)


def write_sharegpt_jsonl(
    output_file: Path,
    records: Iterator[dict],
    desc: str,
) -> int:
    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for record in tqdm(records, desc=desc):
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_image_rel_path(dataset_prefix: str, split: str, filename: str) -> str:
    return f"{dataset_prefix}/images/{split}/{filename}"


def build_sharegpt_record(user_content: str, assistant_content: str, image_rel_path: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_rel_path],
    }


def choose_roco_instruction(
    image_id: str,
    rng: random.Random,
    category_weights: dict[str, float] | None = None,
) -> str:
    categories = list(ROCO_INSTRUCTION_CATEGORIES)
    weights = (
        [category_weights.get(name, 1.0) for name in categories]
        if category_weights
        else [1.0] * len(categories)
    )
    category_rng = random.Random(rng.randint(0, 2**32 - 1) ^ hash(image_id))
    category = category_rng.choices(categories, weights=weights, k=1)[0]
    return category_rng.choice(ROCO_INSTRUCTION_CATEGORIES[category])


def convert_roco_sample(
    sample: dict,
    split: str,
    images_dir: Path,
    rng: random.Random,
    labels_only: bool,
    category_weights: dict[str, float] | None,
    dataset_prefix: str,
) -> dict:
    image_id = sample["image_id"]
    caption = sample["caption"].strip()
    instruction = choose_roco_instruction(image_id, rng, category_weights)

    image_filename = f"{image_id}.jpg"
    image_rel_path = build_image_rel_path(dataset_prefix, split, image_filename)
    save_rgb_image(sample["image"], images_dir / split / image_filename, labels_only)

    return build_sharegpt_record(f"<image>{instruction}", caption, image_rel_path)


def convert_vqa_rad_sample(
    sample: dict,
    split: str,
    index: int,
    images_dir: Path,
    labels_only: bool,
    dataset_prefix: str,
) -> dict:
    question = sample["question"].strip()
    answer = sample["answer"].strip()
    image_filename = f"img_{index}.jpg"
    image_rel_path = build_image_rel_path(dataset_prefix, split, image_filename)
    save_rgb_image(sample["image"], images_dir / split / image_filename, labels_only)
    return build_sharegpt_record(f"<image>{question}", answer, image_rel_path)


def load_roco_split(source_dir: Path, split: str) -> Dataset:
    return load_dataset(str(source_dir), split=split)


def load_vqa_rad_split(source_dir: Path, split: str) -> Dataset:
    parquet_path = source_dir / "data" / VQA_RAD_SPLIT_FILES[split]
    return load_dataset("parquet", data_files=str(parquet_path), split="train")


def load_pmc_vqa_split(source_dir: Path, variant: str, split: str) -> list[dict]:
    csv_name = PMC_VQA_VARIANTS[variant]["splits"][split]
    with (source_dir / csv_name).open(encoding="utf-8") as fin:
        return list(csv.DictReader(fin))


def format_pmc_vqa_question(row: dict, open_ended: bool = False) -> str:
    question = row["Question"].strip()
    if open_ended:
        return question
    choices = [row[f"Choice {label}"].strip() for label in ("A", "B", "C", "D")]
    return "\n".join([question, *choices])


def format_pmc_vqa_answer(row: dict) -> str:
    answer = row["Answer"].strip()
    if len(answer) == 1 and answer in "ABCD":
        choice = row[f"Choice {answer}"].strip()
        prefix = f"{answer}:"
        if choice.startswith(prefix):
            return choice[len(prefix) :].strip()
        return choice
    return answer


def pmc_vqa_sample_text(row: dict) -> str:
    return f"{row['Caption']} {row['Question']} " + " ".join(row[f"Choice {label}"] for label in "ABCD")


def is_radiology_sample(row: dict) -> bool:
    text = pmc_vqa_sample_text(row)
    return bool(RADIOLOGY_KEYWORDS.search(text)) and not bool(NON_RADIOLOGY_KEYWORDS.search(text))


def link_image(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() or dest_path.is_symlink():
        return
    dest_path.symlink_to(source_path.resolve())


def ensure_pmc_vqa_image(
    figure_path: str,
    split: str,
    images_dir: Path,
    source_images_dir: Path,
    zip_archive: zipfile.ZipFile | None,
    zip_member_prefix: str,
    labels_only: bool,
    dataset_prefix: str,
) -> str:
    image_rel_path = build_image_rel_path(dataset_prefix, split, figure_path)
    dest_path = images_dir / split / figure_path
    if labels_only and (dest_path.exists() or dest_path.is_symlink()):
        return image_rel_path

    source_path = source_images_dir / figure_path
    if source_path.is_file():
        link_image(source_path, dest_path)
        return image_rel_path

    if zip_archive is None:
        raise FileNotFoundError(f"Image not found for {figure_path}")

    zip_member = f"{zip_member_prefix}/{figure_path}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zip_archive.open(zip_member) as src, dest_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return image_rel_path


def convert_pmc_vqa_sample(
    row: dict,
    split: str,
    images_dir: Path,
    source_images_dir: Path,
    zip_archive: zipfile.ZipFile | None,
    zip_member_prefix: str,
    labels_only: bool,
    dataset_prefix: str,
    open_ended: bool = False,
) -> dict:
    figure_path = row["Figure_path"].strip()
    question = format_pmc_vqa_question(row, open_ended)
    answer = format_pmc_vqa_answer(row)
    image_rel_path = ensure_pmc_vqa_image(
        figure_path,
        split,
        images_dir,
        source_images_dir,
        zip_archive,
        zip_member_prefix,
        labels_only,
        dataset_prefix,
    )
    return build_sharegpt_record(f"<image>{question}", answer, image_rel_path)


def convert_dataset(
    dataset_name: str,
    source_dir: Path,
    output_dir: Path,
    labels_only: bool,
    seed: int = 42,
    variant: str = "v1",
    open_ended: bool = False,
    radiology_only: bool = False,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    dataset_prefix = output_dir.name
    totals: dict[str, int] = {}
    zip_archive: zipfile.ZipFile | None = None

    if dataset_name == "roco":
        splits = ROCO_SPLITS
        prefix = "roco"
        category_weights = ROCO_CATEGORY_WEIGHTS

        def load_split(split: str) -> Dataset:
            return load_roco_split(source_dir, split)

        def iter_records(split: str, dataset: Dataset) -> Iterator[dict]:
            rng = random.Random(seed + hash(split) % 10000)
            for sample in dataset:
                yield convert_roco_sample(
                    sample, split, images_dir, rng, labels_only, category_weights, dataset_prefix
                )

    elif dataset_name == "vqa-rad":
        splits = tuple(VQA_RAD_SPLIT_FILES)
        prefix = "vqa_rad"

        def load_split(split: str) -> Dataset:
            return load_vqa_rad_split(source_dir, split)

        def iter_records(split: str, dataset: Dataset) -> Iterator[dict]:
            for index, sample in enumerate(dataset):
                yield convert_vqa_rad_sample(sample, split, index, images_dir, labels_only, dataset_prefix)

    elif dataset_name == "pmc-vqa":
        if variant not in PMC_VQA_VARIANTS:
            raise ValueError(f"Unsupported PMC-VQA variant: {variant}")

        config = PMC_VQA_VARIANTS[variant]
        splits = tuple(config["splits"])
        prefix = f"{config['prefix']}_rad" if radiology_only else config["prefix"]
        source_images_dir = source_dir / config["images_subdir"]
        zip_path = source_dir / config["images_zip"]

        if not source_images_dir.is_dir() and zip_path.is_file():
            zip_archive = zipfile.ZipFile(zip_path)

        def load_split(split: str) -> list[dict]:
            rows = load_pmc_vqa_split(source_dir, variant, split)
            if radiology_only:
                rows = [row for row in rows if is_radiology_sample(row)]
            return rows

        def iter_records(split: str, dataset: list[dict]) -> Iterator[dict]:
            for sample in dataset:
                yield convert_pmc_vqa_sample(
                    sample,
                    split,
                    images_dir,
                    source_images_dir,
                    zip_archive,
                    config["images_subdir"],
                    labels_only,
                    dataset_prefix,
                    open_ended,
                )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    try:
        for split in splits:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            dataset = load_split(split)
            totals[split] = write_sharegpt_jsonl(
                output_dir / f"{prefix}_{split}.jsonl",
                iter_records(split, dataset),
                desc=f"Converting {dataset_name} {split}",
            )
    finally:
        if dataset_name == "pmc-vqa" and zip_archive is not None:
            zip_archive.close()

    return totals


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only regenerate jsonl labels; skip saving images that already exist.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert medical imaging datasets to ShareGPT OpenAI format for LlamaFactory.",
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    roco_parser = subparsers.add_parser("roco", help="Convert ROCOv2-radiology.")
    roco_parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/kk/datas/ROCOv2-radiology"),
        help="Path to ROCOv2-radiology dataset directory.",
    )
    roco_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/kk/LlamaFactory/data/roco-sharegpt"),
        help="Output directory for converted dataset.",
    )
    roco_parser.add_argument("--seed", type=int, default=42, help="Random seed for instruction sampling.")
    add_common_args(roco_parser)

    vqa_parser = subparsers.add_parser("vqa-rad", help="Convert VQA-RAD parquet dataset.")
    vqa_parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/kk/datas/vqa-rad"),
        help="Path to VQA-RAD dataset directory.",
    )
    vqa_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/kk/LlamaFactory/data/vqa-rad-sharegpt"),
        help="Output directory for converted dataset.",
    )
    add_common_args(vqa_parser)

    pmc_parser = subparsers.add_parser("pmc-vqa", help="Convert PMC-VQA CSV dataset.")
    pmc_parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/kk/LlamaFactory/datas/PMC-VQA"),
        help="Path to PMC-VQA dataset directory.",
    )
    pmc_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/kk/LlamaFactory/data/pmc-vqa-sharegpt"),
        help="Output directory for converted dataset.",
    )
    pmc_parser.add_argument(
        "--variant",
        choices=tuple(PMC_VQA_VARIANTS),
        default="v2",
        help="PMC-VQA version: v1 (train/test/test_clean) or v2 (train_2/test_2).",
    )
    pmc_parser.add_argument(
        "--open-ended",
        action="store_true",
        help="Open-ended VQA: only include the question, without multiple-choice options.",
    )
    pmc_parser.add_argument(
        "--radiology-only",
        action="store_true",
        help="Keep radiology samples only; output files use the _rad suffix (e.g. pmc_vqa2_rad_train.jsonl).",
    )
    add_common_args(pmc_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    totals = convert_dataset(
        dataset_name=args.dataset,
        source_dir=args.source,
        output_dir=args.output,
        labels_only=args.labels_only,
        seed=getattr(args, "seed", 42),
        variant=getattr(args, "variant", "v2"),
        open_ended=getattr(args, "open_ended", False),
        radiology_only=getattr(args, "radiology_only", False),
    )

    print(f"Conversion complete. Output saved to {args.output}")
    for split, count in totals.items():
        print(f"  {split}: {count} samples")


if __name__ == "__main__":
    main()
