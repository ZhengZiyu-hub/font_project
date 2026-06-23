# 字体/书法图像生成项目

## Project Overview

本项目用于搭建一个自己的字体与书法图像生成工程框架。当前阶段只整理工程目录和基础配置，不迁移参考实现，也不实现复杂模型。

## Directory Structure

```text
font_project/
├── configs/
│   └── base.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── pipelines/
│   ├── train/
│   └── utils/
├── outputs/
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

也可以使用 Docker 管理环境：

```bash
docker compose build
docker compose run --rm app python -m compileall src
docker compose run --rm app python scripts/test_model_forward.py
```

Docker 构建使用 `requirements-docker.txt`，默认安装 CPU 版 PyTorch，适合当前的模型结构验证。

## Usage

基础配置位于：

```bash
configs/base.yaml
```

当前还没有训练和推理入口，后续阶段会逐步补齐数据读取、模型定义、训练流程和推理流程。

## Current Status

This repository currently provides the base project structure and configuration. Full model implementation, dataset training, and inference are still under development.

## Notes

- 当前阶段只做工程整理。
- 不接入外部参考仓库作为运行依赖。
- 输出文件统一放在 `outputs/`。
