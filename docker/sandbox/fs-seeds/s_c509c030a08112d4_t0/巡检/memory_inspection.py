import re

def parse_memory_ruijie(text):
    # 缺少 'Used Rate : X%' (S5750-6F) 格式正则(待修复)
    m = re.search(r'System memory\s*:\s*(\d+)% used', text)
    return int(m.group(1)) if m else None

def parse_memory_huawei(text):
    m = re.search(r'Memory Using Percentage Is:\s*(\d+)%', text)
    return int(m.group(1)) if m else None
