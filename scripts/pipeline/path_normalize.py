r"""Windows/foreign path normalization for training queries.

数据源 (new_trajectories_labeled.jsonl) 的 seed_query 里带三类"非沙箱内"绝对路径，
沙箱是 Linux、工作目录 /home/user，`sandbox_setup._expand_sandbox_path` 只认 `~` →
这些路径原样传进去，模型被要求读/写不存在的位置 → 读不到输入/写不到指定处 →
task_done 崩、模型转而"假完成"。故在构建训练集时统一改成沙箱内相对路径。

三类路径族（2026-08-21 数据勘查 + 复检）:
  1. Windows 盘符:  E:\hermes\runtime\bigtasks\D<N>\<id>\inputs\<f>  → ./inputs/<f>
                    E:\hermes\runtime\winruns\D<N>\<id>\ws\<f>       → ./outputs/<f>
     (唯一真实盘符 E:，其余 C:/D:/S: 等是 "word:\n" 换行伪影，本模块不碰)
  2. UNC 无盘符:    \\hermes\runtime\bigtasks\...\inputs\  → ./inputs/
                    \\hermes\runtime\winruns\...\ws\       → ./outputs/
  3. 外部 Linux 绝对路径(别人的机器/家目录，沙箱内不存在):
       /root/hermes_sig_home/... , /root/<workspace>/... , /mnt/afs_toolcall/... ,
       /mnt/user-data/... , /mnt/afs_agents/... 里带 inputs/ 或 ws/ 段的 → ./inputs 或 ./outputs
     注意：/home/user/... 、/usr/... 、/opt/... 是【合法沙箱路径】，不动。

cl_agent_dataset 把 taskspecs_w3/<rid>/files 注入到沙箱 ./inputs（SANDBOX_INPUTS_DIR），
故 inputs 映射与注入点对齐；输出用独立 ./outputs 与输入隔离，judge 校验产物落此处。
"""
import re

# ── 1. Windows 盘符族（作用于 json.loads 后的单反斜杠字符串）────────────────────
_INPUTS_RE = re.compile(r"[A-Za-z]:\\hermes\\runtime\\bigtasks\\[^\\]+\\[^\\]+\\inputs\\", re.IGNORECASE)
_OUTPUTS_RE = re.compile(r"[A-Za-z]:\\hermes\\runtime\\winruns\\[^\\]+\\[^\\]+\\ws\\", re.IGNORECASE)
_INPUTS_FALLBACK = re.compile(r"[A-Za-z]:\\hermes\\runtime\\[^\\]+\\[^\\]+\\[^\\]+\\inputs\\", re.IGNORECASE)
_OUTPUTS_FALLBACK = re.compile(r"[A-Za-z]:\\hermes\\runtime\\[^\\]+\\[^\\]+\\[^\\]+\\ws\\", re.IGNORECASE)
_HERMES_ROOT = re.compile(r"[A-Za-z]:\\hermes(\\runtime)?\\?", re.IGNORECASE)

# ── 2. UNC 无盘符族  \\hermes\runtime\... ──────────────────────────────────────
_UNC_INPUTS = re.compile(r"\\\\hermes\\runtime\\bigtasks\\[^\\]+\\[^\\]+\\inputs\\", re.IGNORECASE)
_UNC_OUTPUTS = re.compile(r"\\\\hermes\\runtime\\winruns\\[^\\]+\\[^\\]+\\ws\\", re.IGNORECASE)
_UNC_INPUTS_FB = re.compile(r"\\\\hermes\\runtime\\[^\\]+\\[^\\]+\\[^\\]+\\inputs\\", re.IGNORECASE)
_UNC_OUTPUTS_FB = re.compile(r"\\\\hermes\\runtime\\[^\\]+\\[^\\]+\\[^\\]+\\ws\\", re.IGNORECASE)
_UNC_ROOT = re.compile(r"\\\\hermes(\\runtime)?\\?", re.IGNORECASE)

# ── 3. 外部 Linux 绝对路径族（沙箱内不存在的别人机器路径）──────────────────────
# 只处理已知的"外部工作区"前缀，且带 inputs/ 或 ws/ 段时映射；否则截到相对段。
# /home/user、/usr、/opt、/tmp 是合法沙箱路径，绝不碰。
_FOREIGN_PREFIX = r"/(?:root|mnt/afs_toolcall|mnt/afs_agents|mnt/user-data)/[^\s\"']*?"
_LIN_INPUTS = re.compile(_FOREIGN_PREFIX + r"/inputs/", )
_LIN_OUTPUTS = re.compile(_FOREIGN_PREFIX + r"/(?:ws|winruns)/", )
# 外部前缀但无 inputs/ws 段：把整个外部绝对前缀削成 ./（保留最后的文件名/相对尾）
_LIN_ROOT = re.compile(r"/(?:root/[^\s\"'/]+|mnt/afs_toolcall/[^\s\"'/]+(?:/[^\s\"'/]+)?|mnt/afs_agents/[^\s\"'/]+|mnt/user-data)/")
# /root/<file>.<ext> 直接写在 /root 下的产出文件（无子目录）→ ./outputs/<file>
_LIN_ROOT_FILE = re.compile(r"/root/([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,6})\b")

# bare /workspace（无外部前缀，非 /home/user/workspace）→ /home/user/workspace（沙箱可写根）。
# 数据里 8007 条指令直接引用 /workspace/app.py 等，沙箱用户在根目录 / 下无权限、且
# Hermes write_file 临时文件父目录不存在会崩(见 Bug_Fix F8)。归到默认 cwd 下的可写
# workspace。后瞻含中文标点、排除 /workspaces 复数别词；负向后视排除已是 /home/user/workspace。
_BARE_WORKSPACE = re.compile(
    r"(?<!/home/user)/workspace(?=$|[/\s.,;:\'\"`)\]}，。、；：）】！？])"
)

# 从原始（未归一化）query 里抽 generated_tasks_hermes 的任务 id：
#   ...\bigtasks\D10\D10_k982304_zh\inputs\...  →  ("D10", "D10_k982304_zh")
# 支持 E:\、\\ (UNC)、/ 三种分隔。D<N> 即 task_id 的下划线前缀，故只需 task_id。
_GEN_TASK_ID = re.compile(r"(?:bigtasks|winruns)[\\/]+D\d+[\\/]+(D\d+_[A-Za-z0-9]+_[a-z]{2})[\\/]", re.IGNORECASE)

# ── review 任务的 workspace 快照（F5-review，2026-08-22）──────────────────────────
# 采集轨迹(tongronglei)的完整 workspace 快照：
#   /mnt/afs_toolcall/tongronglei/workspace/subagent_trajectory/<branch>/<D<N>>/<taskdir>/ws/...
# 这是 agent 当时的【完整工作目录】(代码/数据/产物混在一起，本就是一个整体，不该拆
# inputs/outputs)。故这批任务的 ws 路径统一归一到 /home/user/workspace/，再把整个 ws
# 文件夹注入沙箱同一位置(cl_agent_dataset.build_agent_assets 的 review-ws 分支)。
# ★必须在通用 _LIN_OUTPUTS(把任意 .../ws/ → ./outputs/)之前处理，否则被误拆到 outputs。
_REVIEW_WS = re.compile(
    r"/mnt/afs_toolcall/tongronglei/workspace/subagent_trajectory/[^\s\"']*?/D\d+/[\w\-]+/ws/",
)
# 索引用：从 ws 路径抽 (D<N>, taskdir) → 定位真实 ws 目录。归一化【前】调用。
_WS_DIR = re.compile(
    r"/subagent_trajectory/(?P<branch>[\w./\-]+?)/(?P<d>D\d+)/(?P<task>[\w\-]+)/ws\b",
)


def extract_ws_dir(text: str) -> str | None:
    """从原始 seed_query 抽 review 任务的 ws 目录相对标识：``<branch>/<D<N>>/<taskdir>``。

    必须在 normalize_paths【之前】调用（归一化会把 /mnt/.../ws/ 削成 /home/user/workspace/
    丢掉定位信息）。返回 None = 该任务 query 不含采集 ws 路径。返回值供
    cl_agent_dataset 拼真实绝对路径 <ROOT>/<branch>/<d>/<task>/ws 注入沙箱。
    """
    if not text:
        return None
    m = _WS_DIR.search(text)
    if not m:
        return None
    return f"{m.group('branch')}/{m.group('d')}/{m.group('task')}"


def extract_gen_task_id(text: str) -> str | None:
    """从原始 query 的 Windows/UNC 路径里抽 generated_tasks_hermes 任务 id（如 D10_k982304_zh）。

    必须在 normalize_paths【之前】调用——归一化会把路径削成 ./inputs/ 丢掉 id。
    抽不到返回 None（该任务无输入文件引用或路径不含 D<N>_<id> 结构）。
    规范化大小写为 generated_tasks_hermes 目录约定：D<N> 大写 D + 数字，后缀小写
    （如 `D11_K982840_EN` → `D11_k982840_en`，实测目录用小写后缀）。
    """
    if not text:
        return None
    m = _GEN_TASK_ID.search(text)
    if not m:
        return None
    tid = m.group(1)
    prefix, _, suffix = tid.partition("_")
    return f"{prefix.upper()}_{suffix.lower()}" if suffix else tid


def _slashify(s: str) -> str:
    """把 /home/user/workspace/ 后残留的反斜杠路径段转正斜杠（多级循环处理）。"""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(/home/user/workspace/[^\s\"']*?)\\(?=[A-Za-z0-9_.])", r"\1/", s)
    return s


def normalize_paths(text: str) -> str:
    """把 query 里 Windows/UNC/外部Linux 绝对路径统一改成沙箱内 /home/user/workspace/。

    只碰三类明确的"非沙箱路径"族；不误伤 "detail:\\n"（换行伪影）、/home/user、/usr 等
    合法沙箱路径、以及普通含斜杠文本。
    输入输出统一 workspace（去掉 input/output 区分，2026-08-26）。
    """
    if not text:
        return text
    s = text
    # 1. Windows 盘符
    s = _INPUTS_RE.sub("/home/user/workspace/", s)
    s = _OUTPUTS_RE.sub("/home/user/workspace/", s)
    s = _INPUTS_FALLBACK.sub("/home/user/workspace/", s)
    s = _OUTPUTS_FALLBACK.sub("/home/user/workspace/", s)
    s = _HERMES_ROOT.sub("/home/user/workspace/", s)
    # 2. UNC
    s = _UNC_INPUTS.sub("/home/user/workspace/", s)
    s = _UNC_OUTPUTS.sub("/home/user/workspace/", s)
    s = _UNC_INPUTS_FB.sub("/home/user/workspace/", s)
    s = _UNC_OUTPUTS_FB.sub("/home/user/workspace/", s)
    s = _UNC_ROOT.sub("/home/user/workspace/", s)
    # 3. 外部 Linux 绝对路径
    # ★ 先处理 review 任务的完整 workspace 快照(tongronglei 采集 ws)：整体归一到
    #   /home/user/workspace/，不拆 inputs/outputs（ws 本就是一个完整工作目录）。
    #   必须在通用 _LIN_OUTPUTS(.../ws/ → workspace/) 之前，否则被误拆。
    s = _REVIEW_WS.sub("/home/user/workspace/", s)
    s = _LIN_INPUTS.sub("/home/user/workspace/", s)
    s = _LIN_OUTPUTS.sub("/home/user/workspace/", s)
    s = _LIN_ROOT_FILE.sub(r"/home/user/workspace/\1", s)  # /root/out.json → /home/user/workspace/out.json
    s = _LIN_ROOT.sub("/home/user/workspace/", s)
    # bare /workspace → /home/user/workspace（在 _REVIEW_WS 之后：采集 ws 完整路径已先归一）
    s = _BARE_WORKSPACE.sub("/home/user/workspace", s)
    # 反斜杠尾巴转正斜杠
    s = _slashify(s)
    return s


if __name__ == "__main__":
    tests = [
        (r"read E:\hermes\runtime\bigtasks\D1\D1_k9_en\inputs\data.csv now",
         "read /home/user/workspace/data.csv now"),
        (r"write to E:\hermes\runtime\winruns\D1\D1_k9_en\ws\out.json done",
         "write to /home/user/workspace/out.json done"),
        (r"nested E:\hermes\runtime\bigtasks\D2\x\inputs\sub\deep\file.xlsx end",
         "nested /home/user/workspace/sub/deep/file.xlsx end"),
        (r"unc \\hermes\runtime\bigtasks\D4\D4_g98_en\inputs\kb.xls here",
         "unc /home/user/workspace/kb.xls here"),
        (r"unc out \\hermes\runtime\winruns\D5\x\ws\r.json done",
         "unc out /home/user/workspace/r.json done"),
        (r"foreign /root/hermes_sig_home/audit_ws/inputs/codebase.txt read",
         "foreign /home/user/workspace/codebase.txt read"),
        (r"foreign out /mnt/afs_toolcall/tongronglei/workspace/subagent/ws/x.json",
         "foreign out /home/user/workspace/x.json"),
        (r"write to /root/fix_output.json now",
         "write to /home/user/workspace/fix_output.json now"),
        # review 任务 ws 快照 → /home/user/workspace/（整体，不拆 inputs/outputs）
        (r"review /mnt/afs_toolcall/tongronglei/workspace/subagent_trajectory/hermes_sig_v1/raw/D7/D7_s980601_en_15_13222/ws/app.py now",
         "review /home/user/workspace/app.py now"),
        (r"check /mnt/afs_toolcall/tongronglei/workspace/subagent_trajectory/hermes_sig_v1/raw/D6/D6_g80_zh_58_12917/ws/inputs/contracts.xlsx",
         "check /home/user/workspace/inputs/contracts.xlsx"),
        # must NOT touch legit sandbox paths / newline artifacts
        ("keep /home/user/inputs/file.xlsx as-is", "keep /home/user/inputs/file.xlsx as-is"),
        ("newline detail:\\n and format:\\n text", None),
        ("legit /usr/share/x and /opt/miniconda3/y", "legit /usr/share/x and /opt/miniconda3/y"),
    ]
    ok = 0
    for inp, exp in tests:
        got = normalize_paths(inp)
        good = (exp is None) or (got == exp)
        ok += good
        print(f"{'OK  ' if good else 'FAIL'} | {inp!r}\n       -> {got!r}")
    print(f"\n{ok}/{len(tests)} passed")

    # extract_ws_dir 单测
    print("\n--- extract_ws_dir ---")
    ws_tests = [
        (r"review /mnt/afs_toolcall/tongronglei/workspace/subagent_trajectory/hermes_sig_v1/raw/D7/D7_s980601_en_15_13222/ws/app.py",
         "hermes_sig_v1/raw/D7/D7_s980601_en_15_13222"),
        ("no ws path here", None),
    ]
    for inp, exp in ws_tests:
        got = extract_ws_dir(inp)
        print(f"{'OK  ' if got == exp else 'FAIL'} | {got!r} (exp {exp!r})")
