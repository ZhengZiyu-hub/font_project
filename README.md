# 字体图像生成项目

## 项目简介

这是一个字体/书法图像生成项目框架。当前代码包含基础工具、模型结构和 FLUX decoder 接入实验，用于跑通最小 forward 流程。

## 目录结构

```text
configs/        配置文件
src/models/     模型模块
src/utils/      配置、随机种子、图像保存、checkpoint 工具
scripts/        简单测试脚本
outputs/        输出目录
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

