# 设备风扇巡检格式说明

本文档定义各厂商设备风扇状态解析格式与巡检判定要求。

## 巡检要求
- 遍历最新日期文件夹下所有设备日志(按厂商分子目录)。
- 每个设备提取风扇槽位编号与运行状态。
- 状态为 Normal/Present/Registered 视为正常，其余(Abnormal/Absent/Fault)告警。
- 结果写入巡检表 Excel 的 fan sheet，列：device, vendor, fan_slot, status, result。

## 华为 Huawei
### 格式A: CE系列 (命令 `display device fan`)
```
Slot  FanID  Online  Status  Speed
1     FAN1   Present Normal  60%
```
### (本文档当前仅描述 CE 格式，USG9520/S5700 待补充)

## 思科 Cisco (命令 `show environment fan`)
```
Fan 1: Normal
Fan 2: Normal
```

## H3C (命令 `display fan`)
```
Fan-tray 1 State: Normal
```
