# -*- coding: utf-8 -*-
# 按 pwr文件格式说明.md 实现的解析脚本（疑似与真实结构不符）
import struct, csv, sys

def parse(path):
    with open(path, 'rb') as f:
        buf = f.read()
    magic, version, count, timestamp = struct.unpack('<4sIII', buf[:16])
    print('magic=%r version=%d count=%d timestamp=%d' % (magic, version, count, timestamp))
    records = []
    pos = 16
    for i in range(count):
        if pos + 16 > len(buf):
            break
        code = buf[pos:pos+8].split(b'\x00')[0].decode('ascii', 'ignore')
        offset, length = struct.unpack('<II', buf[pos+8:pos+16])
        pos += 16
        if not code.startswith(('SH', 'SZ')):
            continue
        dp = offset
        while dp + 16 <= offset + length and dp + 16 <= len(buf):
            date, songgu, peigu, jia = struct.unpack('<iiii', buf[dp:dp+16])
            # 文档未提到 date==0 需要终止，这里照旧读取
            records.append([code, date, songgu/1000.0, peigu/1000.0, jia/1000.0])
            dp += 16
    return records

def main():
    recs = parse('full_sh.PWR')
    print('parsed stocks/records:', len(set(r[0] for r in recs)), len(recs))
    with open('full_sh_exquan.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['code', 'date', 'songgu', 'peigu', 'peigujia'])
        w.writerows(recs)

if __name__ == '__main__':
    main()
