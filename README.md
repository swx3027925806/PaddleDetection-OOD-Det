# PaddleDetection-OOD-Det

该项目为ICCV2023 OOD Det赛道自监督冠军方案，有监督季军方案。

项目基于`PaddlePaddle2.4.0`设计，在使用前请安装好对应版本。同时项目基于PaddleDetection2.6 dev稍作修改


## 数据集准备

该项目有监督数据集来源为官方数据，半监督数据来源于官方不同赛道的训练集与阶段一的测试集。

## 代码运行

有监督代码请指定配置文件 `PaddleDetection/configs/ppyoloe_plus_vit/main.yml`

```
python tools/train.py -c PaddleDetection/configs/ppyoloe_plus_vit/main.yml
```

半监督代码请指定配置文件 `PaddleDetection/configs/ppyoloe_plus_vit_semi/main.yml`

```
PaddleDetection/configs/ppyoloe_plus_vit_semi/main.yml
```

## 团队介绍

西安电子科技大学 `IPIU` 实验室

团队成员：

| 姓名      | 年级 | 邮箱 |
| ----------- | ----------- | ----------- |
| 佘文轩 | 2022级硕士研究生 | swx_sxpp@qq.com |
| 刘雨   | 2022级硕士研究生 | ly2865818487@163.com |
