本仓库包含两个不同的 MotrixLab 运行环境：

1. ANYmal C 学习阶段：基于 MotrixLab 主线环境；
2. VBot 越障阶段：基于 MotrixArena-S1 分支环境。

两个阶段的代码、资产和配置不能直接混用。

## 1. ANYmal C Environment

用于：

- `01_anymal_c_minimal`
- `02_anymal_c_point_navigation`

使用 MotrixLab 主线环境。

```bash
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
uv sync --all-packages --all-extras

对应环境名：

anymal_c_navigation_minimal
anymal_c_navigation_point

## 2. VBot MotrixArena-S1 Environment

用于：

- `03_vbot_section01_navigation`

该阶段需要使用官方指定的 MotrixArena-S1 分支。

git clone --branch MotrixArena-S1 https://github.com/Motphys/MotrixLab.git MotrixLab-ArenaS1
cd MotrixLab-ArenaS1
uv sync --all-packages --all-extras

对应环境名：

vbot_navigation_section01

## 3. Important Notes

ANYmal C 阶段和 VBot 阶段依赖的 MotrixLab 分支不同；
不建议把两个阶段的文件直接混在同一个 MotrixLab 工作区中；
如果需要同时保留两个环境，建议使用两个本地目录：

~/MotrixLab-main/
~/MotrixLab-ArenaS1/

本仓库只保存自定义代码、配置、命令和结果记录，不保存完整 MotrixLab 官方源码。
