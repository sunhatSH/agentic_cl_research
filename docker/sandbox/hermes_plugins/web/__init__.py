# Serper + Jina web search providers — plugins/web/.
#
# 让 plugins.web 成为可 import 的 Python 包（照抄 bundled plugins/web/__init__.py 的
# 约定）。serper/ 子目录 follow 同 layout：
#   plugins/web/serper/{plugin.yaml, __init__.py, provider.py}
#
# kind: backend 自动加载，register 里 ctx.register_web_search_provider() 注册进
# agent.web_search_registry。缺此文件则 `from plugins.web.serper.provider import ...`
# 的包导入失败 → 插件静默被跳过（hermes plugins list 不显示、web_search 无 backend）。
