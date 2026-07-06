# 字体图像生成项目

## 项目简介

这是一个字体/书法图像生成项目框架。当前代码包含基础工具、模型结构、FLUX decoder 接入实验，以及一个中文纹理文字数据生成器。

## 目录结构

```text
configs/        配置文件
src/models/     模型模块
src/utils/      配置、随机种子、图像保存、checkpoint 工具
scripts/        简单测试脚本
outputs/        本地输出目录，默认不提交
assets/         本地下载素材目录，默认不提交
```

## 环境安装

推荐使用 Docker：

```bash
docker compose build
docker compose run --rm app python -m compileall src
```

也可以直接安装依赖：

```bash
pip install -r requirements.txt
```

## 快速测试

```bash
docker compose run --rm app python scripts/test_model_forward.py
```

测试会构造 dummy 输入，运行一次模型 forward，并打印输出图像 shape。

## 中文纹理文字数据生成器

`generate_chinese_texture_images.py` 可以批量生成中文文字数据图：中文风格字体叠加在 DTD/custom 纹理照片，或程序生成的水波纹、木纹、纸张、布纹背景上。

本机下载素材记录在 `ASSETS.md`。`assets/` 和 `outputs/` 默认被 Git 忽略，因为 DTD、字体包和生成数据体积较大。

示例：

```powershell
py .\generate_chinese_texture_images.py --count 1000 --width 512 --height 160
```

常用参数：

- `--background-dir`: 指定自定义背景图片目录
- `--custom-background-ratio`: 控制自定义/DTD 背景占比
- `--standard-font-ratio`: 控制标准字体占比，默认让风格化字体占多数
- `--fonts-dir`: 可重复传入多个字体目录

