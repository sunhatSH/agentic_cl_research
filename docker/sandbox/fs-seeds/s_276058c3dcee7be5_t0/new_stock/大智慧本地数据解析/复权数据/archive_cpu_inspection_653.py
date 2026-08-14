import re

def parse_cpu_h3c(text):
    m = re.search(r'(\d+)% in last 5 seconds', text)
    return int(m.group(1)) if m else None

# 华为 USG9520/S5700 的 'CPU Usage : X%' 待新增
