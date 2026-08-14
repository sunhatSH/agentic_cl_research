import re

def parse_power_h3c(text):
    # 仅支持盒式格式A, 框式格式B未支持(待修复)
    m = re.search(r'Power \d+ State:\s*(\w+)', text)
    return m.group(1) if m else None

def parse_power_maipu(text):
    # Device:0 框式前缀未兼容(待修复)
    m = re.search(r'Power \d+:\s*(\w+)', text)
    return m.group(1) if m else None

def parse_power_ruijie(text):
    # S8610E 的 NO DEVICE/NoLink 未处理(待修复)
    m = re.search(r'Power Status:\s*(\w+)', text)
    return m.group(1) if m else None

def extract_model(text):
    m = re.search(r'(H3C|Huawei|Ruijie|Maipu|Hillstone)\s+(\S+)', text)
    return m.group(0) if m else None  # 失败时应返回'采集失败,请人工核实'

# TODO: 汇总输出 xlsx, 异常红色填充
