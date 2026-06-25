import re

def parse_fan_h3c(text):
    return re.findall(r'Fan \d+ State:\s*(\w+)', text)

# 迈普 Device:0 块、锐捷 NO DEVICE 跳过逻辑待完善
