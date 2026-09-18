# 训练路径与 GUI 加固说明（2026-09-18）

本文是 AudioClone Studio 当前实现的维护基线；如早期计划或设计文档与本文冲突，以本文和代码测试为准。

## 路径边界

- 设置页的“训练模块代码目录”是模块源码、公共模型资源和子进程工作目录。GPT-SoVITS 当前指向 voice-pipeline 仓库根目录。
- 训练页不再提供第二个项目目录。选中框架后，job 的 `project_root` 自动取对应模块的代码目录。
- `training_data.path` 是外部只读输入，可位于 AudioMiner 输出目录或其他任意现有绝对路径。GPT-SoVITS 自动评测按 `dataset.list` 顺序选择第一条有效的中、日或英语样本作为参考，不再要求 GUI 单独填写参考音频。
- `dataset.list` 内绝对音频路径直接使用；相对音频路径以 `dataset.list` 所在目录为基准解析。
- job、事件日志、缓存、checkpoint、评测、晋升模型与推理输出必须位于模块 `project_root` 内。
- GPT-SoVITS 的正式训练目录为 `<module_root>/runs/<project_name>`，GUI 与独立 CLI 使用同一条落盘规则。
- GUI 不复制或移动训练数据，也不向外部素材目录写入缓存。

## GUI 行为

- Windows 进程使用稳定 AppUserModelID `AudioCloneStudio.Desktop`，同时设置应用和窗口图标。
- 应用级滚轮过滤器覆盖所有 `QComboBox` 和 `QAbstractSpinBox`；滚轮用于页面滚动，不修改选项或数值。
- 训练配置使用紧凑双列布局；路径占整行，设备/精度等短字段同行。
- 高级设置默认折叠，展开后将通用、S1、S2 三组并排显示；每组及其控件使用深灰圆角矩形，无底边强调线。
- AudioClone Studio 默认窗口在屏幕允许时为 `1440 × 900`，仍按可用屏幕范围收缩；素材挖掘设置入口使用透明底绿色小齿轮。
- 训练进度默认隐藏，在训练启动、收到事件或显示训练错误时出现；进度条为绿色。
- 活动记录默认折叠，仍保留最多 500 个文本块。
- 候选未生成时保留 A/B/C 三个等宽占位卡；生成后替换为三语试听与人工晋升控件。

## 回归测试入口

AudioMiner 重点测试：

```powershell
python -m pytest tests/training_modules/test_job.py tests/gui/test_training_form.py tests/gui/test_training_page.py tests/gui/test_candidate_page.py tests/gui/test_qt_smoke.py -q
```

voice-pipeline 重点测试：

```powershell
python -m pytest tests/test_module_job.py tests/test_module_materialize.py tests/test_module_runner.py -q
```
