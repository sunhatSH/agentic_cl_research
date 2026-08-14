# 深市复权数据解析（与本任务无关，勿改）
import struct

def parse(path):
    with open(path, 'rb') as f:
        return f.read()
