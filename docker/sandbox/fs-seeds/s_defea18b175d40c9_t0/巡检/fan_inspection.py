# 风扇巡检脚本
import os, glob
import pandas as pd

LATEST = '20260519'
BASE = os.path.join(os.path.dirname(__file__), LATEST)

def parse_fan_huawei(text):
    # 当前仅支持 CE 系列 display device fan 格式
    rows = []
    if 'display device fan' not in text:
        return rows
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit() and parts[1].startswith('FAN'):
            rows.append({'fan_slot': parts[0] + '/' + parts[1], 'status': parts[3]})
    return rows

def parse_fan_cisco(text):
    rows = []
    for line in text.splitlines():
        if line.strip().startswith('Fan '):
            name, _, st = line.partition(':')
            rows.append({'fan_slot': name.strip(), 'status': st.strip()})
    return rows

def parse_fan_h3c(text):
    rows = []
    for line in text.splitlines():
        if line.startswith('Fan-tray'):
            p = line.split('State:')
            rows.append({'fan_slot': p[0].strip(), 'status': p[1].strip()})
    return rows

PARSERS = {'huawei': parse_fan_huawei, 'cisco': parse_fan_cisco, 'h3c': parse_fan_h3c}

def process_log_file(path, vendor):
    with open(path) as f:
        text = f.read()
    rows = PARSERS[vendor](text)
    device = os.path.basename(path).replace('.log', '')
    out = []
    for r in rows:
        ok = r['status'] in ('Normal', 'Present', 'Registered')
        out.append({'device': device, 'vendor': vendor, 'fan_slot': r['fan_slot'], 'status': r['status'], 'result': 'OK' if ok else 'ALARM'})
    return out

def main():
    all_rows = []
    for vendor in os.listdir(BASE):
        vdir = os.path.join(BASE, vendor)
        if not os.path.isdir(vdir):
            continue
        for log in glob.glob(os.path.join(vdir, '*.log')):
            all_rows.extend(process_log_file(log, vendor))
    df = pd.DataFrame(all_rows)
    with pd.ExcelWriter(os.path.join(os.path.dirname(__file__), LATEST + '-巡检表.xlsx')) as w:
        df.to_excel(w, sheet_name='fan', index=False)
    print('parsed devices:', df['device'].nunique())

if __name__ == '__main__':
    main()
