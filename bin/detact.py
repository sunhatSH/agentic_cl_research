import argparse
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import Manager


# =====================================================================
# 用户可调参数（命令行可覆写，下方 parse_args 一一对应）
# =====================================================================
DEFAULT_META_PATH = "/mnt/afs_toolcall/sunhao4/GarbledDataDetection/meta/meta_upload.json"
DEFAULT_OUTPUT_DIR = "/mnt/afs_toolcall/sunhao4/GarbledDataDetection/info_v2"
DEFAULT_DATASET_PROCESSES = os.cpu_count() or 1
DEFAULT_THREADS_PER_DATASET = max(2, min(8, os.cpu_count() or 2))  # kept for CLI compat; ignored
DEFAULT_BATCH_SIZE = 200  # 历史参数，新调度（按 chunk）下已不再使用，仅保留 CLI 兼容性
DEFAULT_CHUNK_LINES = 50_000  # 单 chunk 目标行数；大文件按此切分到多个 worker，零长尾

# 阈值（0~1）
DEFAULT_GARBLE_THRESHOLD = 0.05          # 整条对话的总脏占比阈值；> 阈值整条丢弃
DEFAULT_SINGLE_MSG_THRESHOLD = 0.20      # 单条非 tool message 的脏占比阈值
DEFAULT_TOOL_RESPONSE_THRESHOLD = 0.95   # 单条 role=='tool' message 的脏占比阈值（更宽松）

# 触发后的丢弃策略：drop_all（整条丢，drop_from=0）或 drop_tail（从触发点起丢，drop_from=idx）
DEFAULT_SINGLE_MSG_POLICY = "drop_all"        # 同时作用于 single_msg / tool_response 触发
DEFAULT_WEBSEARCH_ERROR_POLICY = "drop_all"   # 单独作用于 web_search status=error 触发
POLICY_CHOICES = ("drop_all", "drop_tail")

# =====================================================================
# 内部常量（一般不需要改）
# =====================================================================
WEBSEARCH_TOOL_NAME = "web_search"
TARGET_FIELDS = ("content", "reasoning_content")
CLASS_CLEAN = "clean"
CLASS_DIRTY_KEPT = "dirty_kept"
CLASS_DISCARDED = "discarded"
REASON_TOTAL = "total_threshold"
REASON_SINGLE = "single_msg_threshold"
REASON_TOOL_RESPONSE = "tool_response_threshold"
REASON_WEBSEARCH = "websearch_error"


try:
	from tqdm import tqdm
except ImportError:
	tqdm = None


ALLOWED_CONTROL = frozenset({0x09, 0x0A, 0x0D})
ALLOWED_FORMAT = frozenset()

# 严格无语义的零宽字符：在任何语义判定前直接 strip。
# 仅包含 Unicode 标准里被明确定义为"纯排版/已废弃"且不承载任何语言/词法语义的字符。
# 不放进来的反例：U+200B ZWSP（Thai/Lao/Khmer/缅文里是词边界）、U+200C ZWNJ、U+200D ZWJ、
# U+200E LRM / U+200F RLM、U+202A-E、U+2066-9、U+FE00-F、U+E0100-1EF、tag chars 等都有语义，
# 应当作为脏字符走 get_char_category 流程。
NO_SEMANTIC_ZW = frozenset({
	0x2060,  # WORD JOINER：纯禁止换行提示，无语言语义
	0xFEFF,  # ZERO WIDTH NO-BREAK SPACE：文件首是 BOM，文中是 Unicode 3.2 已废弃用法
	0x00AD,  # SOFT HYPHEN：纯排版换行提示，无词法语义
})


def should_strip(char):
	"""Stage A：是否属于"确定无语义"的零宽字符。这些字符在所有统计、分类、清洗里都不出现。"""
	return ord(char) in NO_SEMANTIC_ZW


def get_char_category(char):
	code = ord(char)
	cat = unicodedata.category(char)

	if char == "�":
		return "garbled_replacement"
	if cat == "Cc":
		if code in ALLOWED_CONTROL:
			return "whitespace"
		return "garbled_control"
	if cat == "Cf":
		if code in ALLOWED_FORMAT:
			return "format_allowed"
		return "garbled_format"
	if cat == "Cs":
		return "garbled_surrogate"
	if cat == "Co":
		return "garbled_pua"
	if cat == "Cn":
		return "garbled_unassigned"
	if 0xFDD0 <= code <= 0xFDEF or (code & 0xFFFE) == 0xFFFE:
		return "garbled_noncharacter"
	if 0x20000 <= code <= 0x3134F:
		return "garbled_rare_cjk"
	if 0x00C0 <= code <= 0x00FF:
		return "garbled_mojibake"

	bucket = cat[0]
	if bucket == "L":
		return "letter"
	if bucket == "M":
		return "combining"
	if bucket == "N":
		return "number"
	if bucket == "P":
		return "punctuation"
	if bucket == "S":
		return "symbol"
	if bucket == "Z":
		return "separator"

	return "garbled_unknown"


def is_garbled(char):
	"""Stage B：纯字符级判定，与阈值/数据集无关。"""
	return get_char_category(char).startswith("garbled")


# analyze_text 热路径：每条数据可能数千到数万字符，整库 1.4M 行 → 数十亿次字符判定。
# 上面 should_strip/is_garbled/get_char_category 每次都要 ord+unicodedata.category+多次函数调用，
# 实测在中文 / JSON 主导的数据上耗时 30+ 分钟。
# 这里改成 per-process dict cache：分类一次后把结果（0=strip / 1=normal / 2=garbled）记下来，
# 后续命中直接 dict.get，对一份字符种数有限（~1e4 量级）的语料近乎 O(1)。
# helper（should_strip / is_garbled / get_char_category）保留对外语义不变。
_CLS_STRIP = 0
_CLS_NORMAL = 1
_CLS_GARBLED = 2
_CHAR_CLASS_CACHE = {}


def _classify_char(ch, _strip_set=NO_SEMANTIC_ZW, _allowed_cc=ALLOWED_CONTROL, _category=unicodedata.category):
	code = ord(ch)
	if code in _strip_set:
		return _CLS_STRIP
	if ch == "�":
		return _CLS_GARBLED
	cat = _category(ch)
	if cat == "Cc":
		return _CLS_NORMAL if code in _allowed_cc else _CLS_GARBLED
	if cat == "Cf" or cat == "Cs" or cat == "Co" or cat == "Cn":
		return _CLS_GARBLED
	if 0xFDD0 <= code <= 0xFDEF or (code & 0xFFFE) == 0xFFFE:
		return _CLS_GARBLED
	if 0x20000 <= code <= 0x3134F:
		return _CLS_GARBLED
	if 0x00C0 <= code <= 0x00FF:
		return _CLS_GARBLED
	return _CLS_NORMAL


def analyze_text(text):
	"""Stage C：单遍扫描。返回 (total_chars, garble_chars, unique_garbled_list)。
	NO_SEMANTIC_ZW 字符直接 strip，不计入 total。
	"""
	cache = _CHAR_CLASS_CACHE
	classify = _classify_char
	cache_get = cache.get
	total = 0
	garble_count = 0
	seen = None
	unique_list = None

	for ch in text:
		c = cache_get(ch)
		if c is None:
			c = classify(ch)
			cache[ch] = c
		if c == _CLS_NORMAL:
			total += 1
		elif c == _CLS_GARBLED:
			total += 1
			garble_count += 1
			if seen is None:
				seen = set()
				unique_list = []
			if ch not in seen:
				seen.add(ch)
				unique_list.append(ch)
		# c == _CLS_STRIP: 跳过，不计入 total。

	return total, garble_count, unique_list if unique_list is not None else []


def iter_target_fields(messages):
	if not isinstance(messages, list):
		return

	for message in messages:
		if not isinstance(message, dict):
			continue
		for field_name in TARGET_FIELDS:
			field_value = message.get(field_name)
			if isinstance(field_value, str) and field_value:
				yield field_value


def is_websearch_error(message):
	"""检测一条 message 是否是 web_search 工具的错误返回。
	判定标准：role=='tool' AND name=='web_search' AND content 是 JSON 且 status=='error'。
	"""
	if not isinstance(message, dict):
		return False
	if message.get("role") != "tool":
		return False
	if message.get("name") != WEBSEARCH_TOOL_NAME:
		return False
	content = message.get("content")
	if not isinstance(content, str) or not content:
		return False
	try:
		parsed = json.loads(content)
	except (ValueError, TypeError):
		return False
	if not isinstance(parsed, dict):
		return False
	return parsed.get("status") == "error"


def analyze_messages(messages):
	"""按 message 粒度逐条扫描，返回 per-message 统计列表。"""
	results = []
	if not isinstance(messages, list):
		return results

	for idx, message in enumerate(messages):
		entry = {
			"idx": idx,
			"role": message.get("role") if isinstance(message, dict) else None,
			"name": message.get("name") if isinstance(message, dict) else None,
			"total_chars": 0,
			"garble_chars": 0,
			"garbled_unique": [],
			"is_websearch_error": False,
		}
		if not isinstance(message, dict):
			results.append(entry)
			continue

		seen = set()
		for field_name in TARGET_FIELDS:
			value = message.get(field_name)
			if not isinstance(value, str) or not value:
				continue
			t, g, ulist = analyze_text(value)
			entry["total_chars"] += t
			entry["garble_chars"] += g
			for ch in ulist:
				if ch not in seen:
					seen.add(ch)
					entry["garbled_unique"].append(ch)

		entry["is_websearch_error"] = is_websearch_error(message)
		results.append(entry)

	return results


def evaluate_record(
	data,
	total_threshold,
	single_threshold,
	tool_response_threshold,
	single_policy,
	websearch_policy,
):
	"""Stage C+D 合并：返回分类决策与诊断信息。

	决策顺序：
	  1. 找最早触发 single_msg_threshold（role!='tool'）的 message
	  2. 找最早触发 tool_response_threshold（role=='tool'）的 message
	  3. 找最早触发 web_search status=error 的 message
	  4. 三者各自按 policy 算 drop_from（single+tool 共用 single_policy；websearch 用 websearch_policy）
	     再取 (drop_from, idx) 字典序最小的一个作为最终结果
	  5. 若上述都未触发，再看全行总阈值；超阈 → discarded(drop_from=0)，否则 dirty_kept / clean
	"""
	messages = data.get("messages") if isinstance(data, dict) else None
	per_msg = analyze_messages(messages)

	total_chars = sum(s["total_chars"] for s in per_msg)
	garble_chars = sum(s["garble_chars"] for s in per_msg)
	seen = set()
	garbled_unique = []
	for s in per_msg:
		for ch in s["garbled_unique"]:
			if ch not in seen:
				seen.add(ch)
				garbled_unique.append(ch)

	earliest_single = None        # 非 tool message 触发 single_threshold
	earliest_tool = None          # tool message 触发 tool_response_threshold
	earliest_websearch = None     # web_search status=error
	for s in per_msg:
		if earliest_websearch is None and s["is_websearch_error"]:
			earliest_websearch = s["idx"]
		if s["total_chars"] > 0:
			ratio = s["garble_chars"] / s["total_chars"]
			if s["role"] == "tool":
				if earliest_tool is None and ratio > tool_response_threshold:
					earliest_tool = s["idx"]
			else:
				if earliest_single is None and ratio > single_threshold:
					earliest_single = s["idx"]

	candidates = []
	if earliest_single is not None:
		df = 0 if single_policy == "drop_all" else earliest_single
		candidates.append((df, earliest_single, REASON_SINGLE))
	if earliest_tool is not None:
		df = 0 if single_policy == "drop_all" else earliest_tool
		candidates.append((df, earliest_tool, REASON_TOOL_RESPONSE))
	if earliest_websearch is not None:
		df = 0 if websearch_policy == "drop_all" else earliest_websearch
		candidates.append((df, earliest_websearch, REASON_WEBSEARCH))

	drop_from = None
	reason = None
	if candidates:
		drop_from, _, reason = min(candidates, key=lambda x: (x[0], x[1]))
		classification = CLASS_DISCARDED
	else:
		if garble_chars == 0:
			classification = CLASS_CLEAN
		elif total_chars <= 0:
			classification = CLASS_DIRTY_KEPT
		elif (garble_chars / total_chars) > total_threshold:
			classification = CLASS_DISCARDED
			drop_from = 0
			reason = REASON_TOTAL
		else:
			classification = CLASS_DIRTY_KEPT

	return {
		"classification": classification,
		"drop_from": drop_from,
		"discard_reason": reason,
		"total_chars": total_chars,
		"garble_chars": garble_chars,
		"garbled_unique": garbled_unique,
		"message_count": len(per_msg),
		"earliest_single_msg_idx": earliest_single,
		"earliest_tool_response_idx": earliest_tool,
		"earliest_websearch_error_idx": earliest_websearch,
	}


def sanitize_filename(name):
	cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
	cleaned = cleaned.strip("._")
	return cleaned or "dataset"


def extract_source_preview(data, max_len=20):
	if not isinstance(data, dict):
		return ""
	messages = data.get("messages", [])
	if not isinstance(messages, list):
		return ""
	for message in messages:
		if isinstance(message, dict) and message.get("role") == "user":
			content = message.get("content")
			if isinstance(content, str) and content:
				return content[:max_len]
	for message in messages:
		if isinstance(message, dict):
			for field_name in TARGET_FIELDS:
				value = message.get(field_name)
				if isinstance(value, str) and value:
					return value[:max_len]
	return ""


def build_info_record(dataset_name, line_number, evaluation, source_preview):
	total_chars = evaluation["total_chars"]
	garble_chars = evaluation["garble_chars"]
	dirty_ratio = round(garble_chars / total_chars, 4) if total_chars > 0 else 0.0
	return {
		"dataset_name": dataset_name,
		"line_number": line_number,
		"classification": evaluation["classification"],
		"total_chars": total_chars,
		"garble_char_count": garble_chars,
		"dirty_ratio": dirty_ratio,
		"garbled_characters": evaluation["garbled_unique"],
		"message_count": evaluation["message_count"],
		"drop_from": evaluation["drop_from"],
		"discard_reason": evaluation["discard_reason"],
		"earliest_single_msg_idx": evaluation["earliest_single_msg_idx"],
		"earliest_tool_response_idx": evaluation["earliest_tool_response_idx"],
		"earliest_websearch_error_idx": evaluation["earliest_websearch_error_idx"],
		"source_preview": source_preview,
	}


class _JsonlWriter:
	def __init__(self, output_path):
		self.output_path = output_path
		self._lock = threading.Lock()
		output_dir = os.path.dirname(output_path)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)
		with open(self.output_path, "w", encoding="utf-8"):
			pass

	def write(self, record):
		payload = json.dumps(record, ensure_ascii=False)
		with self._lock:
			with open(self.output_path, "a", encoding="utf-8") as handle:
				handle.write(payload)
				handle.write("\n")


class ResultWriter:
	"""四路 jsonl 输出：
	- info：仅脏数据（class 2 + class 3）的诊断行
	- clean_index：class 1 的 line_number
	- acceptable_index：class 2 的 line_number（删脏字符后可用）
	- discarded_index：class 3 的 line_number（脏占比超阈值，被丢弃）
	"""

	def __init__(self, info_path, clean_index_path, acceptable_index_path, discarded_index_path):
		self.info_path = info_path
		self.clean_index_path = clean_index_path
		self.acceptable_index_path = acceptable_index_path
		self.discarded_index_path = discarded_index_path
		self._info = _JsonlWriter(info_path)
		self._clean_index = _JsonlWriter(clean_index_path)
		self._acceptable_index = _JsonlWriter(acceptable_index_path)
		self._discarded_index = _JsonlWriter(discarded_index_path)

	def write_info(self, record):
		self._info.write(record)

	def write_clean_index(self, record):
		self._clean_index.write(record)

	def write_acceptable_index(self, record):
		self._acceptable_index.write(record)

	def write_discarded_index(self, record):
		self._discarded_index.write(record)


class ProgressReporter:
	"""Per-dataset 进度条 + 总进度条。chunk 调度下：每个 dataset 一根 bar，
	chunk 的 `progress` / `hit` 事件按 meta_key 聚合到对应 bar；
	所有 chunk 都完成时才把 dataset 标记为 done。"""

	def __init__(self, tasks, chunks_per_dataset):
		self._lock = threading.Lock()
		self._closed = False
		self._hit_counts = {task["meta_key"]: 0 for task in tasks}
		self._chunks_remaining = dict(chunks_per_dataset)  # meta_key -> 还差几个 chunk 完成
		self._bars = {}
		self._total_bar = None
		self._total_datasets = len(tasks)
		self._done_datasets = 0
		self._total_hits = 0
		# 文本打印用的聚合计数（独立于 tqdm，后台/非 TTY 也能看到进度）
		self._scanned_lines = 0
		self._total_expected_lines = sum(t["expected_lines"] or 0 for t in tasks)
		self._total_chunks = sum(chunks_per_dataset.values())
		self._done_chunks = 0
		self._started_at = time.time()

		if tqdm is not None:
			self._total_bar = tqdm(
				total=self._total_expected_lines if self._total_expected_lines else None,
				desc=f"TOTAL ({self._total_datasets} datasets)",
				unit="line",
				position=0,
				leave=True,
			)
			for index, task in enumerate(tasks):
				meta_key = task["meta_key"]
				label = meta_key if len(meta_key) <= 28 else meta_key[:25] + "..."
				n_chunks = chunks_per_dataset.get(meta_key, 1)
				if n_chunks > 1:
					label = f"{label} [×{n_chunks}]"
				self._bars[meta_key] = tqdm(
					total=task["expected_lines"] if task["expected_lines"] else None,
					desc=label,
					unit="line",
					position=index + 1,
					leave=True,
				)

	def handle_event(self, event):
		event_type = event.get("type")
		meta_key = event.get("meta_key")
		if event_type == "progress":
			line_delta = event.get("line_delta", 0)
			with self._lock:
				self._scanned_lines += line_delta
				bar = self._bars.get(meta_key)
				if bar is not None:
					bar.update(line_delta)
				if self._total_bar is not None:
					self._total_bar.update(line_delta)
		elif event_type == "hit":
			with self._lock:
				hit_delta = event.get("hit_delta", 0)
				self._hit_counts[meta_key] = self._hit_counts.get(meta_key, 0) + hit_delta
				self._total_hits += hit_delta
				bar = self._bars.get(meta_key)
				if bar is not None:
					bar.set_postfix(hits=self._hit_counts[meta_key])
				if self._total_bar is not None:
					self._total_bar.set_postfix(
						datasets=f"{self._done_datasets}/{self._total_datasets}",
						hits=self._total_hits,
					)
		elif event_type == "chunk_done":
			with self._lock:
				self._done_chunks += 1
				remaining = self._chunks_remaining.get(meta_key)
				if remaining is not None:
					remaining -= 1
					self._chunks_remaining[meta_key] = remaining
					if remaining <= 0:
						self._done_datasets += 1
						bar = self._bars.get(meta_key)
						if bar is not None:
							bar.set_postfix(done=True, hits=self._hit_counts.get(meta_key, 0))
						if self._total_bar is not None:
							self._total_bar.set_postfix(
								datasets=f"{self._done_datasets}/{self._total_datasets}",
								hits=self._total_hits,
							)
		elif event_type == "error":
			with self._lock:
				bar = self._bars.get(meta_key)
				if bar is not None:
					bar.set_postfix(error=event.get("message", "error"))
				else:
					print(f"[{meta_key}] {event.get('message', 'error')}", file=sys.stderr)

	def snapshot(self):
		"""Return a thread-safe snapshot of aggregate counters for the text printer."""
		with self._lock:
			elapsed = time.time() - self._started_at
			scanned = self._scanned_lines
			expected = self._total_expected_lines
			pct = (scanned / expected * 100) if expected else 0.0
			rate = (scanned / elapsed) if elapsed > 0 else 0.0
			eta = ((expected - scanned) / rate) if (rate > 0 and expected > scanned) else None
			return {
				"elapsed_seconds": elapsed,
				"scanned_lines": scanned,
				"expected_lines": expected,
				"progress_pct": pct,
				"rate_lines_per_sec": rate,
				"eta_seconds": eta,
				"done_datasets": self._done_datasets,
				"total_datasets": self._total_datasets,
				"done_chunks": self._done_chunks,
				"total_chunks": self._total_chunks,
				"total_hits": self._total_hits,
			}

	def close(self):
		with self._lock:
			if self._closed:
				return
			self._closed = True
			for bar in self._bars.values():
				bar.close()
			if self._total_bar is not None:
				self._total_bar.close()


def _fmt_duration(secs):
	if secs is None:
		return "?"
	s = int(secs)
	h, rem = divmod(s, 3600)
	m, s = divmod(rem, 60)
	if h:
		return f"{h}h{m:02d}m{s:02d}s"
	if m:
		return f"{m}m{s:02d}s"
	return f"{s}s"


def status_printer(reporter, stop_event):
	"""每跨过 1% 进度（按 scanned_lines / expected_lines 整数百分点变化）打一行到 stdout。
	不依赖 tqdm，后台/无 TTY 也能看到；100 个文件就是 100 行，密度刚好。"""
	last_pct = -1
	while not stop_event.is_set():
		snap = reporter.snapshot()
		cur_pct = int(snap["progress_pct"])
		if cur_pct > last_pct:
			print(
				f"[scan] {cur_pct:3d}%  "
				f"elapsed={_fmt_duration(snap['elapsed_seconds'])} "
				f"lines={snap['scanned_lines']:,}/{snap['expected_lines']:,} "
				f"datasets={snap['done_datasets']}/{snap['total_datasets']} "
				f"chunks={snap['done_chunks']}/{snap['total_chunks']} "
				f"hits={snap['total_hits']:,} "
				f"rate={snap['rate_lines_per_sec']:.0f} lines/s "
				f"eta={_fmt_duration(snap['eta_seconds'])}",
				flush=True,
			)
			last_pct = cur_pct
		time.sleep(0.5)
	# 收尾再打一行（即使没跨百分点，至少留个 FINAL 行）
	snap = reporter.snapshot()
	print(
		f"[scan] FINAL  "
		f"elapsed={_fmt_duration(snap['elapsed_seconds'])} "
		f"lines={snap['scanned_lines']:,}/{snap['expected_lines']:,} "
		f"({snap['progress_pct']:.2f}%) "
		f"datasets={snap['done_datasets']}/{snap['total_datasets']} "
		f"chunks={snap['done_chunks']}/{snap['total_chunks']} "
		f"hits={snap['total_hits']:,}",
		flush=True,
	)


def progress_consumer(progress_queue, reporter, stop_event):
	while not stop_event.is_set() or not progress_queue.empty():
		try:
			event = progress_queue.get(timeout=0.1)
		except queue.Empty:
			continue
		reporter.handle_event(event)


# =====================================================================
# 切片调度
# =====================================================================
# 旧版：ProcessPoolExecutor 每个 dataset 一个 future，单文件不可继续切分 →
#   遇到一个大文件（如 stepfun 282K 行）就会长尾，整机 N-1 个 worker 全部闲置。
# 新版：把工作单元从 "dataset" 改为 "chunk of K lines"。小文件 = 1 chunk，大文件 = ceil(N/K) chunk，
#   chunk 按目标大小降序提交，最大的先跑；多个 worker 同时啃大文件。
# chunk 各自把结果写到 per-chunk 文件 (sanitized_name.chunk0001.jsonl 等)，
# 全部完成后按 dataset 把 chunk 文件 cat 成单一输出，保留下游 (filter_clean / summarize_run) 的契约。

def build_chunks(tasks, chunk_lines):
	"""把 tasks 按 chunk_lines 切成工作单元。
	返回 list[chunk]：每个 chunk = {meta_key, annotation_path, expected_lines, start_line, end_line,
	                                 chunk_id, total_chunks, chunk_expected}
	start_line/end_line 是 1-based inclusive；end_line=None 表示读到 EOF。
	"""
	chunks = []
	for task in tasks:
		meta_key = task["meta_key"]
		annotation_path = task["annotation_path"]
		expected = task["expected_lines"] or 0

		if chunk_lines <= 0 or expected <= 0 or expected <= chunk_lines:
			# 不切：单 chunk 覆盖整个文件
			chunks.append({
				"meta_key": meta_key,
				"annotation_path": annotation_path,
				"expected_lines": expected,
				"start_line": 1,
				"end_line": None,
				"chunk_id": 0,
				"total_chunks": 1,
				"chunk_expected": expected,
			})
			continue

		n_chunks = (expected + chunk_lines - 1) // chunk_lines
		base = expected // n_chunks
		rem = expected % n_chunks
		offset = 0
		for i in range(n_chunks):
			size = base + (1 if i < rem else 0)
			start = offset + 1
			end = offset + size
			chunks.append({
				"meta_key": meta_key,
				"annotation_path": annotation_path,
				"expected_lines": expected,
				"start_line": start,
				"end_line": end,
				"chunk_id": i,
				"total_chunks": n_chunks,
				"chunk_expected": size,
			})
			offset = end

	return chunks


def chunk_output_filename(meta_key, chunk_id, total_chunks):
	safe = sanitize_filename(meta_key)
	if total_chunks <= 1:
		return f"{safe}.jsonl"
	return f"{safe}.chunk{chunk_id:04d}.jsonl"


def process_chunk(
	chunk,
	output_dir,
	indices_dir,
	discarded_indices_dir,
	total_threshold,
	single_threshold,
	tool_response_threshold,
	single_policy,
	websearch_policy,
	progress_queue,
	progress_flush_lines=1000,
):
	"""扫描一个 chunk（同 dataset 的某段行号区间）。
	写入 per-chunk 临时输出文件；返回 chunk_summary。Progress 事件聚合到 meta_key 这根 bar 上。
	"""
	meta_key = chunk["meta_key"]
	annotation_path = chunk["annotation_path"]
	start_line = chunk["start_line"]
	end_line = chunk["end_line"]
	chunk_id = chunk["chunk_id"]
	total_chunks = chunk["total_chunks"]

	started_at = time.time()
	fn = chunk_output_filename(meta_key, chunk_id, total_chunks)
	info_path = os.path.join(output_dir, fn)
	clean_index_path = os.path.join(indices_dir, "clean", fn)
	acceptable_index_path = os.path.join(indices_dir, "acceptable", fn)
	discarded_index_path = os.path.join(discarded_indices_dir, fn)

	summary = {
		"meta_key": meta_key,
		"chunk_id": chunk_id,
		"total_chunks": total_chunks,
		"annotation_path": annotation_path,
		"info_path": info_path,
		"clean_index_path": clean_index_path,
		"acceptable_index_path": acceptable_index_path,
		"discarded_index_path": discarded_index_path,
		"start_line": start_line,
		"end_line": end_line,
		"chunk_expected": chunk["chunk_expected"],
		"expected_lines": chunk["expected_lines"],
		"line_count": 0,
		"clean_count": 0,
		"dirty_kept_count": 0,
		"discarded_count": 0,
		"hit_count": 0,
		"json_error_count": 0,
		"exists": False,
		"elapsed_seconds": 0.0,
		"error": None,
	}

	if not annotation_path:
		summary["error"] = "annotation path is empty"
		progress_queue.put({"type": "error", "meta_key": meta_key, "message": summary["error"]})
		progress_queue.put({"type": "chunk_done", "meta_key": meta_key, "chunk_id": chunk_id})
		summary["elapsed_seconds"] = round(time.time() - started_at, 3)
		return summary

	if not os.path.exists(annotation_path):
		summary["error"] = "annotation file does not exist"
		progress_queue.put({"type": "error", "meta_key": meta_key, "message": summary["error"]})
		progress_queue.put({"type": "chunk_done", "meta_key": meta_key, "chunk_id": chunk_id})
		summary["elapsed_seconds"] = round(time.time() - started_at, 3)
		return summary

	summary["exists"] = True
	writer = ResultWriter(info_path, clean_index_path, acceptable_index_path, discarded_index_path)

	clean_count = 0
	dirty_kept_count = 0
	discarded_count = 0
	json_error_count = 0
	line_count = 0
	hit_since_flush = 0
	lines_since_flush = 0

	with open(annotation_path, "r", encoding="utf-8") as handle:
		for line_number, raw_line in enumerate(handle, start=1):
			if line_number < start_line:
				continue
			if end_line is not None and line_number > end_line:
				break
			line_count += 1
			lines_since_flush += 1

			stripped_line = raw_line.strip()
			if stripped_line:
				try:
					data = json.loads(stripped_line)
				except json.JSONDecodeError:
					json_error_count += 1
					data = None
				if data is not None:
					evaluation = evaluate_record(
						data,
						total_threshold,
						single_threshold,
						tool_response_threshold,
						single_policy,
						websearch_policy,
					)
					classification = evaluation["classification"]
					info_record = build_info_record(
						meta_key,
						line_number,
						evaluation,
						extract_source_preview(data),
					)

					if classification == CLASS_CLEAN:
						writer.write_clean_index(info_record)
						clean_count += 1
					else:
						writer.write_info(info_record)
						if classification == CLASS_DIRTY_KEPT:
							writer.write_acceptable_index(info_record)
							dirty_kept_count += 1
						else:
							writer.write_discarded_index(info_record)
							discarded_count += 1
						hit_since_flush += 1

			if lines_since_flush >= progress_flush_lines:
				progress_queue.put({"type": "progress", "meta_key": meta_key, "line_delta": lines_since_flush})
				if hit_since_flush:
					progress_queue.put({"type": "hit", "meta_key": meta_key, "hit_delta": hit_since_flush})
				lines_since_flush = 0
				hit_since_flush = 0

	if lines_since_flush:
		progress_queue.put({"type": "progress", "meta_key": meta_key, "line_delta": lines_since_flush})
	if hit_since_flush:
		progress_queue.put({"type": "hit", "meta_key": meta_key, "hit_delta": hit_since_flush})

	summary["line_count"] = line_count
	summary["clean_count"] = clean_count
	summary["dirty_kept_count"] = dirty_kept_count
	summary["discarded_count"] = discarded_count
	summary["hit_count"] = dirty_kept_count + discarded_count
	summary["json_error_count"] = json_error_count
	summary["elapsed_seconds"] = round(time.time() - started_at, 3)

	progress_queue.put({"type": "chunk_done", "meta_key": meta_key, "chunk_id": chunk_id})
	return summary


def merge_chunks_for_dataset(meta_key, total_chunks, output_dir, indices_dir, discarded_indices_dir):
	"""把同一 dataset 的所有 chunk 输出文件 cat 成单一 dataset 输出，并清掉 chunk 临时文件。
	若 total_chunks == 1，chunk_output_filename 直接用 dataset 文件名，不需要 merge。"""
	if total_chunks <= 1:
		return

	safe = sanitize_filename(meta_key)
	clean_dir = os.path.join(indices_dir, "clean")
	acc_dir = os.path.join(indices_dir, "acceptable")

	for dir_path in (output_dir, clean_dir, acc_dir, discarded_indices_dir):
		target = os.path.join(dir_path, f"{safe}.jsonl")
		with open(target, "w", encoding="utf-8") as out:
			for i in range(total_chunks):
				chunk_path = os.path.join(dir_path, f"{safe}.chunk{i:04d}.jsonl")
				if not os.path.exists(chunk_path):
					continue
				with open(chunk_path, "r", encoding="utf-8") as inp:
					shutil.copyfileobj(inp, out)
				os.remove(chunk_path)


def aggregate_dataset_summaries(tasks, chunk_summaries, output_dir, indices_dir, discarded_indices_dir,
                                total_threshold, single_threshold, tool_response_threshold,
                                single_policy, websearch_policy):
	"""把 per-chunk summary 按 dataset 聚合，输出与旧版 process_dataset 兼容的 dataset_summary 列表。"""
	by_key = {task["meta_key"]: task for task in tasks}
	agg = {}
	for cs in chunk_summaries:
		k = cs["meta_key"]
		if k not in agg:
			task = by_key.get(k, {})
			safe = sanitize_filename(k)
			agg[k] = {
				"dataset_name": k,
				"annotation_path": cs["annotation_path"],
				"output_file": os.path.join(output_dir, f"{safe}.jsonl"),
				"clean_index_file": os.path.join(indices_dir, "clean", f"{safe}.jsonl"),
				"acceptable_index_file": os.path.join(indices_dir, "acceptable", f"{safe}.jsonl"),
				"discarded_index_file": os.path.join(discarded_indices_dir, f"{safe}.jsonl"),
				"exists": cs.get("exists", False),
				"expected_lines": cs.get("expected_lines", task.get("expected_lines", 0)),
				"line_count": 0,
				"hit_count": 0,
				"clean_count": 0,
				"dirty_kept_count": 0,
				"discarded_count": 0,
				"json_error_count": 0,
				"total_threshold": total_threshold,
				"single_msg_threshold": single_threshold,
				"tool_response_threshold": tool_response_threshold,
				"single_msg_policy": single_policy,
				"websearch_error_policy": websearch_policy,
				"elapsed_seconds": 0.0,
				"chunk_count": 0,
				"error": None,
			}
		row = agg[k]
		row["line_count"] += cs["line_count"]
		row["clean_count"] += cs["clean_count"]
		row["dirty_kept_count"] += cs["dirty_kept_count"]
		row["discarded_count"] += cs["discarded_count"]
		row["hit_count"] += cs["hit_count"]
		row["json_error_count"] += cs["json_error_count"]
		row["elapsed_seconds"] = round(row["elapsed_seconds"] + cs["elapsed_seconds"], 3)
		row["chunk_count"] += 1
		row["exists"] = row["exists"] or cs.get("exists", False)
		if cs.get("error") and not row.get("error"):
			row["error"] = cs["error"]
	# Fill in datasets that produced zero chunks (e.g. empty meta entries; in practice always ≥1)
	for task in tasks:
		if task["meta_key"] not in agg:
			safe = sanitize_filename(task["meta_key"])
			agg[task["meta_key"]] = {
				"dataset_name": task["meta_key"],
				"annotation_path": task["annotation_path"],
				"output_file": os.path.join(output_dir, f"{safe}.jsonl"),
				"clean_index_file": os.path.join(indices_dir, "clean", f"{safe}.jsonl"),
				"acceptable_index_file": os.path.join(indices_dir, "acceptable", f"{safe}.jsonl"),
				"discarded_index_file": os.path.join(discarded_indices_dir, f"{safe}.jsonl"),
				"exists": False,
				"expected_lines": task.get("expected_lines", 0),
				"line_count": 0,
				"hit_count": 0,
				"clean_count": 0,
				"dirty_kept_count": 0,
				"discarded_count": 0,
				"json_error_count": 0,
				"total_threshold": total_threshold,
				"single_msg_threshold": single_threshold,
				"tool_response_threshold": tool_response_threshold,
				"single_msg_policy": single_policy,
				"websearch_error_policy": websearch_policy,
				"elapsed_seconds": 0.0,
				"chunk_count": 0,
				"error": "no chunks produced",
			}
	return list(agg.values())


def load_meta(meta_path):
	with open(meta_path, "r", encoding="utf-8") as handle:
		data = json.load(handle)

	tasks = []
	for meta_key, meta_item in data.items():
		if not isinstance(meta_item, dict):
			continue
		# 收集所有 annotation 字段，每个都作为独立数据集
		ann_fields = []
		for field in ("annotation", "tj_annotation"):
			path = meta_item.get(field)
			if path:
				ann_fields.append((field, path))
		for field, path in ann_fields:
			suffix = f"__{field}" if len(ann_fields) > 1 else ""
			tasks.append(
				{
					"meta_key": meta_key + suffix,
					"annotation_path": path,
					"expected_lines": meta_item.get("length", 0),
				}
			)
	return tasks


def write_summary(output_dir, meta_path, dataset_processes, chunk_lines, total_threshold, single_threshold, tool_response_threshold, single_policy, websearch_policy, summaries, started_at):
	summary_path = os.path.join(output_dir, "summary.json")
	totals = {
		"scanned_lines": sum(item["line_count"] for item in summaries),
		"clean_lines": sum(item["clean_count"] for item in summaries),
		"dirty_kept_lines": sum(item["dirty_kept_count"] for item in summaries),
		"discarded_lines": sum(item["discarded_count"] for item in summaries),
		"matched_lines": sum(item["hit_count"] for item in summaries),
		"json_error_lines": sum(item["json_error_count"] for item in summaries),
	}
	payload = {
		"meta_path": meta_path,
		"output_dir": output_dir,
		"started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
		"finished_at": datetime.now().isoformat(timespec="seconds"),
		"dataset_processes": dataset_processes,
		"chunk_lines": chunk_lines,
		"garble_threshold": total_threshold,
		"single_msg_threshold": single_threshold,
		"tool_response_threshold": tool_response_threshold,
		"single_msg_policy": single_policy,
		"websearch_error_policy": websearch_policy,
		"totals": totals,
		"datasets": [
			{
				"dataset_name": item["dataset_name"],
				"output_file": item["output_file"],
				"clean_index_file": item["clean_index_file"],
				"acceptable_index_file": item["acceptable_index_file"],
				"discarded_index_file": item["discarded_index_file"],
				"annotation_path": item["annotation_path"],
				"matched_line_count": item["hit_count"],
				"clean_line_count": item["clean_count"],
				"dirty_kept_line_count": item["dirty_kept_count"],
				"discarded_line_count": item["discarded_count"],
				"scanned_line_count": item["line_count"],
				"json_error_count": item["json_error_count"],
				"chunk_count": item.get("chunk_count", 1),
				"total_threshold": item["total_threshold"],
				"single_msg_threshold": item["single_msg_threshold"],
				"tool_response_threshold": item["tool_response_threshold"],
				"single_msg_policy": item["single_msg_policy"],
				"websearch_error_policy": item["websearch_error_policy"],
				"elapsed_seconds": item["elapsed_seconds"],
				"error": item["error"],
			}
			for item in summaries
		],
	}

	with open(summary_path, "w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2)

	return summary_path


def parse_args():
	parser = argparse.ArgumentParser(description="按 chunk（长度均衡）并发扫描 JSONL 特殊字符。大文件自动切片，长尾被多 worker 同时啃。")
	parser.add_argument("--meta-path", default=DEFAULT_META_PATH, help="meta.json 路径")
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录，默认 info_v2")
	parser.add_argument(
		"--dataset-processes",
		type=int,
		default=DEFAULT_DATASET_PROCESSES,
		help="ProcessPool worker 数量，默认 CPU 核心数",
	)
	parser.add_argument(
		"--chunk-lines",
		type=int,
		default=DEFAULT_CHUNK_LINES,
		help=f"单 chunk 目标行数，大文件按此切；设为 0 或极大值则退回 1 chunk/file 老行为。默认 {DEFAULT_CHUNK_LINES}。",
	)
	parser.add_argument(
		"--threads-per-dataset",
		type=int,
		default=DEFAULT_THREADS_PER_DATASET,
		help="（已废弃，保留以兼容旧脚本调用；新调度不使用线程）",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=DEFAULT_BATCH_SIZE,
		help="（已废弃，保留以兼容旧脚本调用；新调度按整 chunk 处理）",
	)
	parser.add_argument(
		"--garble-threshold",
		type=float,
		default=DEFAULT_GARBLE_THRESHOLD,
		help="脏字符占比阈值（0~1）。> 阈值整条丢弃，> 0 且 ≤ 阈值进 acceptable，= 0 进 clean。默认 0.05。",
	)
	parser.add_argument(
		"--single-msg-threshold",
		type=float,
		default=DEFAULT_SINGLE_MSG_THRESHOLD,
		help="单条非 tool message 的脏占比阈值（0~1）。超过则按 --single-msg-policy 处理。默认 0.20。",
	)
	parser.add_argument(
		"--tool-response-threshold",
		type=float,
		default=DEFAULT_TOOL_RESPONSE_THRESHOLD,
		help="单条 role=='tool' message（工具返回）的脏占比阈值（0~1）。超过则按 --single-msg-policy 处理（与单句共用 policy）。默认 0.95。",
	)
	parser.add_argument(
		"--single-msg-policy",
		choices=POLICY_CHOICES,
		default=DEFAULT_SINGLE_MSG_POLICY,
		help="单句 / tool 触发时：drop_all 整条丢弃；drop_tail 从该 message 起丢弃后续。默认 drop_all。",
	)
	parser.add_argument(
		"--websearch-error-policy",
		choices=POLICY_CHOICES,
		default=DEFAULT_WEBSEARCH_ERROR_POLICY,
		help="检测到 web_search 工具 status=error 时：drop_all 整条丢弃；drop_tail 从该 message 起丢弃后续。默认 drop_all。",
	)
	parser.add_argument(
		"--indices-dir",
		default=None,
		help="可用行索引输出目录。下面会建 clean/ 和 acceptable/ 两个子目录。默认放在 --output-dir 同级的 usable_indices/。",
	)
	parser.add_argument(
		"--discarded-indices-dir",
		default=None,
		help="被丢弃（脏占比超阈值）行索引输出目录。默认放在 --output-dir 同级的 discarded_indices/。",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	started_at = time.time()

	if not os.path.exists(args.meta_path):
		raise FileNotFoundError(f"meta file not found: {args.meta_path}")

	if not (0.0 <= args.garble_threshold <= 1.0):
		raise ValueError(f"--garble-threshold must be in [0, 1], got {args.garble_threshold}")
	if not (0.0 <= args.single_msg_threshold <= 1.0):
		raise ValueError(f"--single-msg-threshold must be in [0, 1], got {args.single_msg_threshold}")
	if not (0.0 <= args.tool_response_threshold <= 1.0):
		raise ValueError(f"--tool-response-threshold must be in [0, 1], got {args.tool_response_threshold}")

	tasks = load_meta(args.meta_path)
	os.makedirs(args.output_dir, exist_ok=True)

	indices_dir = args.indices_dir or os.path.join(
		os.path.dirname(os.path.abspath(args.output_dir)) or ".",
		"usable_indices",
	)
	for sub in ("clean", "acceptable"):
		os.makedirs(os.path.join(indices_dir, sub), exist_ok=True)

	discarded_indices_dir = args.discarded_indices_dir or os.path.join(
		os.path.dirname(os.path.abspath(args.output_dir)) or ".",
		"discarded_indices",
	)
	os.makedirs(discarded_indices_dir, exist_ok=True)

	if not tasks:
		summary_path = write_summary(
			output_dir=args.output_dir,
			meta_path=args.meta_path,
			dataset_processes=0,
			chunk_lines=args.chunk_lines,
			total_threshold=args.garble_threshold,
			single_threshold=args.single_msg_threshold,
			tool_response_threshold=args.tool_response_threshold,
			single_policy=args.single_msg_policy,
			websearch_policy=args.websearch_error_policy,
			summaries=[],
			started_at=started_at,
		)
		print(f"No tasks found. Empty summary written to {summary_path}")
		return 0

	# 切片，大 chunk 排前面优先调度（避免末尾再出长尾）
	chunks = build_chunks(tasks, args.chunk_lines)
	chunks.sort(key=lambda c: -(c["chunk_expected"] or 0))
	chunks_per_dataset = {}
	for c in chunks:
		chunks_per_dataset[c["meta_key"]] = c["total_chunks"]

	max_workers = max(1, min(args.dataset_processes, len(chunks)))

	manager = Manager()
	progress_queue = manager.Queue()
	reporter = ProgressReporter(tasks, chunks_per_dataset)
	stop_event = threading.Event()
	consumer_thread = threading.Thread(
		target=progress_consumer,
		args=(progress_queue, reporter, stop_event),
		daemon=True,
	)
	consumer_thread.start()
	printer_thread = threading.Thread(
		target=status_printer,
		args=(reporter, stop_event),
		daemon=True,
	)
	printer_thread.start()

	chunk_summaries = []
	try:
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			futures = [
				executor.submit(
					process_chunk,
					chunk,
					args.output_dir,
					indices_dir,
					discarded_indices_dir,
					args.garble_threshold,
					args.single_msg_threshold,
					args.tool_response_threshold,
					args.single_msg_policy,
					args.websearch_error_policy,
					progress_queue,
				)
				for chunk in chunks
			]
			for future in futures:
				chunk_summaries.append(future.result())
	finally:
		stop_event.set()
		consumer_thread.join()
		printer_thread.join()
		reporter.close()
		manager.shutdown()

	# 把多 chunk 的输出 cat 回单文件
	for meta_key, total_chunks in chunks_per_dataset.items():
		merge_chunks_for_dataset(meta_key, total_chunks, args.output_dir, indices_dir, discarded_indices_dir)

	summaries = aggregate_dataset_summaries(
		tasks, chunk_summaries, args.output_dir, indices_dir, discarded_indices_dir,
		args.garble_threshold, args.single_msg_threshold, args.tool_response_threshold,
		args.single_msg_policy, args.websearch_error_policy,
	)
	summaries.sort(key=lambda item: item["dataset_name"])
	summary_path = write_summary(
		output_dir=args.output_dir,
		meta_path=args.meta_path,
		dataset_processes=max_workers,
		chunk_lines=args.chunk_lines,
		total_threshold=args.garble_threshold,
		single_threshold=args.single_msg_threshold,
		tool_response_threshold=args.tool_response_threshold,
		single_policy=args.single_msg_policy,
		websearch_policy=args.websearch_error_policy,
		summaries=summaries,
		started_at=started_at,
	)

	scanned = sum(item["line_count"] for item in summaries)
	clean = sum(item["clean_count"] for item in summaries)
	dirty_kept = sum(item["dirty_kept_count"] for item in summaries)
	discarded = sum(item["discarded_count"] for item in summaries)
	matched = sum(item["hit_count"] for item in summaries)
	discarded_ratio = (discarded / scanned) if scanned > 0 else 0.0
	dirty_kept_ratio = (dirty_kept / scanned) if scanned > 0 else 0.0

	print(
		"Scan completed: "
		f"datasets={len(summaries)}, "
		f"chunks={len(chunks)}, "
		f"matched_lines={matched}, "
		f"clean_lines={clean}, "
		f"dirty_kept_lines={dirty_kept}, "
		f"discarded_lines={discarded}, "
		f"scanned_lines={scanned}, "
		f"summary={summary_path}"
	)
	print("| total_threshold | single_msg_threshold | tool_response_threshold | single_msg_policy | websearch_error_policy | discarded_ratio | dirty_kept_ratio |")
	print("| --- | --- | --- | --- | --- | --- | --- |")
	print(
		f"| {args.garble_threshold:g} "
		f"| {args.single_msg_threshold:g} "
		f"| {args.tool_response_threshold:g} "
		f"| {args.single_msg_policy} "
		f"| {args.websearch_error_policy} "
		f"| {discarded_ratio:.4%} "
		f"| {dirty_kept_ratio:.4%} |"
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
