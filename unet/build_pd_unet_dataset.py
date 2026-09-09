"""Build a PD dataset variant whose images are replaced by nnU-Net segmentations.

The generated dataset keeps the original directory hierarchy and metadata schema,
but every source image is replaced by a predicted segmentation mask saved as PNG.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unet.predict import load_predictor, predict_single


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
JSON_FILES = ("patients_dataset_en.json", "patients_dataset_test.json")


def collect_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def target_image_path(source_root: Path, output_root: Path, image_path: Path) -> Path:
    relative_path = image_path.relative_to(source_root)
    return (output_root / relative_path).with_suffix(".png")


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def rewrite_json_paths(data, source_name: str, output_name: str) -> None:
    for patient in data:
        image_paths = patient.get("image_paths", {})
        for region, paths in image_paths.items():
            rewritten_paths = []
            for image_path in paths:
                relative = Path(image_path)
                if relative.parts and relative.parts[0] == source_name:
                    relative = Path(output_name, *relative.parts[1:])
                rewritten_paths.append(str(relative.with_suffix(".png")))
            image_paths[region] = rewritten_paths


def build_readme(source_readme: Path, output_readme: Path, output_name: str, image_count: int) -> None:
    content = source_readme.read_text(encoding="utf-8")
    content = content.replace("pd_dataset_en/", f"{output_name}/")
    content = content.replace("PD Dataset EN", "PD Dataset UNet EN")
    content = content.replace("# 帕金森病眼部图像数据集 (PD Dataset EN)", "# 帕金森病分割图像数据集 (PD Dataset UNet EN)")
    note = (
        "\n## 🤖 UNet 分割版本说明\n\n"
        f"本数据集由训练完成的 nnU-Net 5-fold ensemble 自动生成，共替换 {image_count} 张图像。"
        " 原始 RGB 图像已被对应的角膜分割结果替换，文件统一保存为 PNG。"
        " 元信息结构与原始数据集保持一致，仅 image_paths 路径前缀改为当前数据集名。\n"
    )
    if "## 🤖 UNet 分割版本说明" not in content:
        content += note
    output_readme.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pd_dataset_unet_en from pd_dataset_en using trained nnU-Net")
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "pd_dataset_en",
        help="Source dataset root",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "pd_dataset_unet_en",
        help="Target dataset root",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "nnunet_data" / "nnUNet_results" / "Dataset001_Cornea",
        help="Trained nnU-Net model directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the target dataset first if it already exists",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    predictor = load_predictor(model_dir=args.model_dir, fold="all")
    image_paths = collect_images(source_root)
    if not image_paths:
        raise RuntimeError(f"No images found under: {source_root}")

    skipped_count = 0
    for index, image_path in enumerate(image_paths, start=1):
        out_path = target_image_path(source_root, output_root, image_path)
        if out_path.exists() and not args.overwrite:
            skipped_count += 1
            if index % 100 == 0 or index == len(image_paths):
                print(
                    f"[Progress] {index}/{len(image_paths)} images visited "
                    f"(reused {skipped_count} existing masks)"
                )
            continue

        mask = predict_single(image_path=image_path, predictor=predictor)
        save_mask(mask, out_path)
        if index % 100 == 0 or index == len(image_paths):
            print(
                f"[Progress] {index}/{len(image_paths)} images visited "
                f"(reused {skipped_count} existing masks)"
            )

    for json_name in JSON_FILES:
        source_json = source_root / json_name
        if not source_json.exists():
            continue
        data = json.loads(source_json.read_text(encoding="utf-8"))
        rewrite_json_paths(data, source_root.name, output_root.name)
        (output_root / json_name).write_text(
            json.dumps(data, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )

    source_readme = source_root / "README.md"
    if source_readme.exists():
        build_readme(source_readme, output_root / "README.md", output_root.name, len(image_paths))

    print(f"[Done] Built dataset: {output_root}")
    print(f"[Done] Images processed: {len(image_paths)}")
    print(f"[Done] Existing masks reused: {skipped_count}")


if __name__ == "__main__":
    main()
