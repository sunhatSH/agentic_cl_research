# 电源巡检输出格式说明

## H3C 华三
常见命令: display power
格式A(盒式 S5560):
```
Slot 1:
 Input Power: 250(W)
 Power 1 State: Normal
 Power 2 State: Absent
```
格式B(框式 S7506):
```
 PowerID  State    Mode
 1        Normal   AC
 2        Fault    AC
```
(注: 当前 parse_power_h3c 仅识别格式A, 12台框式设备全部解析失败需修复)

## 华为 huawei
命令: display power
```
PowerID  Online  Mode  State
1        Present  AC    Supply
2        Present  AC    NoPower
```

## 锐捷 ruijie
S5750: `show power` 输出 `Power Status: OK`
S8610E: 输出含 `NO DEVICE` / `NoLink` 行需跳过, 当前未支持

## 迈普 maipu
盒式正常, 框式输出 `Device:0` 前缀块未兼容, 大量失败

## 山石 shanshi
命令: show environment power
```
Slot 0 power: normal
```
