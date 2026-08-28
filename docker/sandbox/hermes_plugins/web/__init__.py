# Serper + Jina web search providers — plugins/web/.
#
# 与 bundled plugins/web/__init__.py 对齐的包标记文件（bundled 有，故保留一致）。
# 注：hermes 加载 user 插件靠 importlib.util.spec_from_file_location + submodule_
# search_locations（见 hermes_cli/plugins.py::_load_directory_module），模块名是
# hermes_plugins.web__serper，**不靠此文件的包结构**。所以：
#   · 此文件在不在都不影响加载（加载走文件路径 spec，不走 import plugins.web.*）。
#   · serper/__init__.py 必须用【相对 import】`from .provider import ...`；用绝对
#     `from plugins.web.serper.provider import ...` 会 No module named（user 插件不在
#     sys.path，绝对包名解析不到）——这才是 web_search undefined 的真根因（2026-08-28）。
