"""按 meta_upload.json 列出的源 jsonl，把非语义零宽字符 strip 掉，写到镜像目录。

源数据不会被修改；输出每行与源数据一一对应（空行、JSON 解析失败行也原样保留），
仅对每行 JSON 内的字符串值递归调用 str.translate 删掉以下码点：

  U+2060  WORD JOINER
  U+FEFF  ZERO WIDTH NO-BREAK SPACE / BOM
  U+00AD  SOFT HYPHEN

判定标准与 detact.py 的 NO_SEMANTIC_ZW 集合保持一致。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


DEFAULT_META_PATH = "/mnt/afs_toolcall/sunhao4/GarbledDataDetection/meta/meta_upload.json"
DEFAULT_OUTPUT_ROOT = "/mnt/afs/sunhao4/cleaned_source"
DEFAULT_PROCESSES = max(1, (os.cpu_count() or 1))

NO_SEMANTIC_ZW = frozenset({0x2060, 0xFEFF, 0x00AD})
TRANSLATE_TABLE = {cp: None for cp in NO_SEMANTIC_ZW}


try:
	from tqdm import tqdm
except ImportError:
	tqdm = None


def strip_zw(value):
	if isinstance(value, str):
		return value.translate(TRANSLATE_TABLE)
	if isinstance(value, list):
		return [strip_zw(v) for v in value]
	if isinstance(value, dict):
		return {k: strip_zw(v) for k, v in value.items()}
	return value


def common_dir_prefix(paths):
	"""返回 paths 的最长公共目录前缀（保证以 / 结尾，可能为空字符串）。"""
	if not paths:
		return ""
	prefix = os.path.commonpath(paths)
	if not prefix:
		return ""
	if not prefix.endswith(os.sep):
		prefix += os.sep
	return prefix


def resolve_output_path(annotation_path, common_prefix, output_root):
	"""把源路径相对于公共前缀的部分原样挂到 output_root 下。"""
	if common_prefix and annotation_path.startswith(common_prefix):
		rel = annotation_path[len(common_prefix):]
	else:
		rel = os.path.basename(annotation_path)
	return os.path.join(output_root, rel)


def clean_file(annotation_path, output_path):
	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	line_count = 0
	stripped_lines = 0
	stripped_chars = 0
	json_error_lines = 0
	empty_lines = 0
	started_at = time.time()

	with open(annotation_path, "r", encoding="utf-8") as src, \
		open(output_path, "w", encoding="utf-8", newline="") as dst:
		for raw_line in src:
			line_count += 1

			payload = raw_line.rstrip("\n")
			if not payload.strip():
				dst.write(raw_line)
				empty_lines += 1
				continue

			try:
				data = json.loads(payload)
			except json.JSONDecodeError:
				dst.write(raw_line)
				json_error_lines += 1
				continue

			cleaned = strip_zw(data)
			cleaned_text = json.dumps(cleaned, ensure_ascii=False)

			if cleaned_text != payload:
				stripped_lines += 1
				before = sum(payload.count(chr(cp)) for cp in NO_SEMANTIC_ZW)
				after = sum(cleaned_text.count(chr(cp)) for cp in NO_SEMANTIC_ZW)
				stripped_chars += max(0, before - after)

			dst.write(cleaned_text)
			dst.write("\n")

	return {
		"annotation_path": annotation_path,
		"output_path": output_path,
		"line_count": line_count,
		"stripped_lines": stripped_lines,
		"stripped_chars": stripped_chars,
		"json_error_lines": json_error_lines,
		"empty_lines": empty_lines,
		"elapsed_seconds": round(time.time() - started_at, 3),
	}


def load_tasks(meta_path):
	with open(meta_path, "r", encoding="utf-8") as handle:
		data = json.load(handle)

	tasks = []
	for meta_key, meta_item in data.items():
		if not isinstance(meta_item, dict):
			continue
		ann_fields = []
		for field in ("annotation", "tj_annotation"):
			path = meta_item.get(field)
			if path:
				ann_fields.append((field, path))
		for field, path in ann_fields:
			suffix = f"__{field}" if len(ann_fields) > 1 else ""
			tasks.append({"meta_key": meta_key + suffix, "annotation_path": path})
	return tasks


def write_cleaned_meta(meta_path, common_prefix, output_root, output_meta_path):
	"""读原 meta，把每个 annotation 替换为清洗后的新路径，写到 output_meta_path。
	默认 output_meta_path == meta_path，即直接覆盖原 meta。
	"""
	with open(meta_path, "r", encoding="utf-8") as handle:
		meta = json.load(handle)

	updated = 0
	for meta_item in meta.values():
		if not isinstance(meta_item, dict):
			continue
		for field in ("annotation", "tj_annotation"):
			annotation = meta_item.get(field)
			if not annotation:
				continue
			meta_item[field] = resolve_output_path(annotation, common_prefix, output_root)
			updated += 1
		updated += 1

	output_dir = os.path.dirname(output_meta_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(output_meta_path, "w", encoding="utf-8") as handle:
		json.dump(meta, handle, ensure_ascii=False, indent=4)
	return updated


def parse_args():
	parser = argparse.ArgumentParser(description="Strip NO_SEMANTIC_ZW chars from source jsonl files; line count preserved.")
	parser.add_argument("--meta-path", default=DEFAULT_META_PATH, help="meta_upload.json 路径")
	parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="清洗后数据的输出根目录（在用户目录下，不会触碰源数据）")
	parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES, help="并发进程数（每个进程处理一个文件）")
	parser.add_argument("--overwrite", action="store_true", help="若输出文件已存在，默认会跳过；加此开关则覆盖重写")
	parser.add_argument("--meta-output", default=None, help="清洗后的 meta 写入路径；默认直接覆盖 --meta-path 指向的原文件")
	return parser.parse_args()


def main():
	args = parse_args()
	started_at = time.time()

	if not os.path.exists(args.meta_path):
		raise FileNotFoundError(f"meta file not found: {args.meta_path}")

	tasks = load_tasks(args.meta_path)
	if not tasks:
		print("No tasks found in meta file.", file=sys.stderr)
		return 0

	annotation_paths = [t["annotation_path"] for t in tasks]
	common_prefix = common_dir_prefix(annotation_paths)

	plans = []
	skipped = []
	missing = []
	for task in tasks:
		src = task["annotation_path"]
		dst = resolve_output_path(src, common_prefix, args.output_root)
		if not os.path.exists(src):
			missing.append({"meta_key": task["meta_key"], "annotation_path": src})
			continue
		if os.path.exists(dst) and not args.overwrite:
			skipped.append({"meta_key": task["meta_key"], "output_path": dst})
			continue
		plans.append({"meta_key": task["meta_key"], "src": src, "dst": dst})

	for item in missing:
		print(f"[missing] {item['meta_key']} -> {item['annotation_path']}", file=sys.stderr)
	for item in skipped:
		print(f"[skip-existing] {item['meta_key']} -> {item['output_path']} (use --overwrite to redo)")

	if not plans:
		print("Nothing to do.")
		return 0

	max_workers = max(1, min(args.processes, len(plans)))
	results = []
	progress = None
	if tqdm is not None:
		progress = tqdm(total=len(plans), desc="cleaning", unit="file")

	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		future_to_plan = {
			executor.submit(clean_file, p["src"], p["dst"]): p for p in plans
		}
		for future in as_completed(future_to_plan):
			plan = future_to_plan[future]
			try:
				result = future.result()
			except Exception as exc:
				result = {
					"annotation_path": plan["src"],
					"output_path": plan["dst"],
					"error": repr(exc),
				}
			result["meta_key"] = plan["meta_key"]
			results.append(result)
			if progress is not None:
				progress.update(1)
				if "error" in result:
					progress.set_postfix(last=plan["meta_key"], err=1)
				else:
					progress.set_postfix(last=plan["meta_key"], stripped=result["stripped_lines"])

	if progress is not None:
		progress.close()

	results.sort(key=lambda r: r.get("meta_key", ""))
	totals = {
		"line_count": sum(r.get("line_count", 0) for r in results),
		"stripped_lines": sum(r.get("stripped_lines", 0) for r in results),
		"stripped_chars": sum(r.get("stripped_chars", 0) for r in results),
		"json_error_lines": sum(r.get("json_error_lines", 0) for r in results),
		"empty_lines": sum(r.get("empty_lines", 0) for r in results),
	}

	meta_output_path = args.meta_output or args.meta_path
	updated_meta_entries = write_cleaned_meta(
		args.meta_path, common_prefix, args.output_root, meta_output_path
	)

	summary_path = os.path.join(args.output_root, "clean_zerowidth_summary.json")
	os.makedirs(args.output_root, exist_ok=True)
	with open(summary_path, "w", encoding="utf-8") as handle:
		json.dump(
			{
				"meta_path": args.meta_path,
				"meta_output_path": meta_output_path,
				"meta_entries_updated": updated_meta_entries,
				"output_root": args.output_root,
				"common_prefix": common_prefix,
				"started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
				"finished_at": datetime.now().isoformat(timespec="seconds"),
				"processes": max_workers,
				"missing": missing,
				"skipped_existing": skipped,
				"totals": totals,
				"files": results,
			},
			handle,
			ensure_ascii=False,
			indent=2,
		)

	print(
		"Done: "
		f"files={len(results)}, "
		f"lines={totals['line_count']}, "
		f"stripped_lines={totals['stripped_lines']}, "
		f"stripped_chars={totals['stripped_chars']}, "
		f"json_errors={totals['json_error_lines']}, "
		f"meta={meta_output_path}, "
		f"summary={summary_path}"
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
