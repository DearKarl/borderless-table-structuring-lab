# V4 GBDT + TeleOCR — 2026-10-06 公开归档

本目录归档已完成的 **GBDT+Tele experimental-001** 实验：1651 页完整计分，1644 页完成识别、7 页超时保留空预测，Overall **97.17188999355706**。完整指标、方向、单位、TeleOCR 冻结参考比较及本地协议见 [测评.md](测评.md)。

这是原本地交付的**公开导出版**，独立 ZIP 和清单见本目录。原模型三文件与评分值未变；个人 owner、宿主路径和 GPU 标识已改为不可直接运行的占位值。公开代码的字符串差异与原始/公开 SHA256 映射见 [PUBLICATION.md](PUBLICATION.md) 和 [PUBLIC_EXPORT.json](provenance/PUBLIC_EXPORT.json)。原本地 ZIP 保持私有且不变。公开导出没有重新测评，不把历史成绩当作新环境运行保证。

## 内容与检查

| 路径 | 内容 |
|---|---|
| `model/` | 原 30 PDF / 30 组 / 90 行训练的 GBDT 模型、配置、manifest |
| `runtime/gpu/code/` | 46 个 GPU 源码文件；41 个原字节不变，5 个仅部署元数据字符串脱敏 |
| `runtime/cpu/` | 48 个 CPU 源码文件；43 个原字节不变，5 个同样脱敏；独立公开文件锁 |
| `runtime/legacy/` | 7 个未修改的容器、租约和清理模块 |
| `dependencies/` | 原生版本、外部资产哈希、输入 roster 元数据及依赖配方 |
| `results/` | 未改字节的 REPORT.json、METRICS.json 与来源哈希 |
| `provenance/` | 模型来源、原始锁、公开源码锁及逐文件导出映射 |
| `examples/` | 明确未授权、路径待绑定的请求模板 |

从本目录运行：

```bash
python -B verify_delivery.py
```

校验器只读取文件，不加载 pickle、不执行推理、不联网。`DELIVERY_MANIFEST.json` 是本公开目录的清单；`ORIGINAL_GPU_PACKAGE_MANIFEST.json`、`ORIGINAL_SOURCE_LOCKS.json` 仅作原版本来源记录。ZIP 与目录内容一致，SHA256 见 `SHA256SUMS.txt`。

## 模型与策略

原 `model/config.json` 保留 `learned_enabled=false` 默认值。实际测评通过 `hybrid.v4_selected_eval.experimental_host` / `prepare_experimental` 在内存中启用同一 GBDT，margin=0.0、fallback=B；模型磁盘字节不变。本次没有重训。直接按原配置加载默认策略，不能代表本次实验策略。原记录中 estimator predict 调用 1476 次，A/B 选择为 241/1410 页，这不是因果收益结论。

本次冻结输入是 raster，候选 A/B，每页最多一次原生识别；原输入、尺寸、取整、预算、失败与输出规则仍由所附逻辑定义。原模型来源见 `provenance/MODEL_ORIGIN.json`。分数来自本地官方协议完整测评；与 TeleOCR 公开参考的 +0.2619 分仅作背景比较。

## 环境和复现边界

这是版本与源码归档，**不是可直接启动的通用安装包**。公开 owner/GPU/lease 值是拒绝误用的占位符，未实现新平台适配。新的部署需要独立审查真实环境绑定并生成新的锁、授权和运行目录；不得删掉校验或复用原运行状态。本次发布没有启动这些流程。

`dependencies/EXTERNAL_ASSETS.json` 用占位路径列出 TeleOCR 模型/源代码/辅助模型、原生环境、驱动和官方评测器的哈希。大型权重、数据集、GT 正文、私有 native binding / rounding / overlay / dependency receipts / 授权前置证据未附带，缺失时冻结流程应拒绝启动。输入文件必须匹配 `INPUT_ROSTER.json`；该文件仅包含页面标识和哈希。

原生 Python 3.12.14 与镜像固定；GBDT overlay 固定 scikit-learn 1.9.1、SciPy 1.18.1、joblib 1.6.0、threadpoolctl 3.7.0、narwhals 2.26.0、cloudpickle 3.1.2，保留原生 NumPy 2.5.3。`dependencies/recipe/` 仅提供原始构建/探测配方，不代表当前环境已安装或通过。具体包版本以清单为准。

正式流程仍是：真实绑定 → 原实验入口单次推理 → 全部页面终态且分配正向释放 → 冻结单组预测 → CPU 官方评分。评分使用冻结 OmniDocBench v1.6 源码/配置/GT，8 CPU、32 GiB、禁网、无 GPU、原 24 小时总界限。GT 仅供评分。任何公开版新运行均需重新验证环境；原报告不能作为它已运行的证明。

本次公开检查验证目录/ZIP 哈希、原模型及未修改源码一致性、报告数值、相对链接、元数据脱敏和静态语法。没有重新读取或评分完整预测正文；原始证据仍由项目保留。外部资产与许可边界见 [PUBLICATION.md](PUBLICATION.md)。
