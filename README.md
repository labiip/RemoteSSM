# RemoteSSM: Physiology-Informed Lightweight Complex State-Space Model for Remote Heart-Rate Estimation

本仓库是论文 *Physiology-Informed Lightweight Complex State-Space Model for Remote Heart-Rate Estimation*
（投稿 *Biomedical Signal Processing and Control*）的实验归档：**全部训练与评测代码、超参数配置，
以及论文中全部数字对应的评测日志**。

模型规模 **0.52 M** 参数（515,068），在 subject-independent 协议下 UBFC / PURE / MMPD 三个测试集上的
帧级 MAE 分别为 **0.87 / 0.31 / 13.69 bpm**。

---

## 1. 目录结构

```text
.
├── src/                               全部源码（76 个顶层 .py + core/4 + 人脸检测级联 xml）
│   ├── main.py                        主脚本：MMPD face_roi-only 训练 + 评测入口
│   ├── model.py                       模型定义 RemoteSSM（复数 SSM + COB + 门控融合 + 频域 FFN + 残差）
│   ├── dataset.py                     数据加载：UBFC/PURE/MMPD 读取与离线缓存构建（含 5-ROI 信号抽取）
│   ├── train.py                       训练循环（AdamW / warmup+cosine / 梯度累积 / 梯度裁剪）
│   ├── evaluate.py                    评测（FFT 峰值拾取 → MAE/RMSE/Pearson/Acc<3,5,8）
│   ├── config.py                      全部超参数与 subject 划分
│   ├── face_tracker.py                人脸检测与跟踪（OpenCV Haar + 时域平滑）
│   ├── preprocess_mmpd.py             原始 MMPD (.mat) → 统一数据格式
│   ├── preprocess_pure.py             原始 PURE (zip)  → 统一数据格式
│   ├── utils.py                       工具函数（生理信号提取等）
│   ├── core/                          模型核心算子: frame_stem / temporal_stem / parallel_scan
│   ├── run_*.py                       各实验的训练入口（主实验 / 消融 / 跨域 / 基线）
│   ├── eval*_step30.py / _eval_*.py   论文表格口径（step=30）的评测脚本
│   ├── *_model.py                     基线模型（PhysNet / PhysMamba / RhythmMamba）
│   ├── _*.py / *_probe.py             诊断、校准与一次性核查脚本（非论文结果主链路）
│   └── haarcascade_frontalface_alt2.xml   人脸检测级联模型（face_tracker.py 加载）
├── logs/                              80 个日志（53 个评测/训练 *.out + 27 个训练 *_train.log），论文全部数字的原始出处
├── requirements.txt
└── README.md
```


---

## 2. 运行环境

| 项 | 值 |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.5.1（CUDA 12.1） |
| OpenCV | 4.10.0 |
| GPU | 单张 NVIDIA TITAN Xp（12 GB） |
| 训练耗时 | 联合 UBFC+PURE 训练 ≈ 11 h；MMPD 微调 ≈ 32 h（均 40 epoch） |
| 混合精度 | 关闭（全程 fp32，避免 FFT/exp 下溢导致 NaN） |

```bash
pip install -r requirements.txt
# requirements.txt 末尾另列了复现 Mamba 系基线（PhysMamba / RhythmMamba）所需的可选依赖
```

---

## 3. 数据准备

使用三个**公开**数据集（本仓库不包含原始数据，请自行获取）：

| 数据集 | 获取方式 |
| --- | --- |
| UBFC-RPPG | https://sites.google.com/view/ybenezeth/ubfcrppg |
| PURE | https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/pure/ |
| MMPD | https://github.com/McJackTang/MMPD_rPPG_dataset |

预处理流程（生成 `auto_face_roi_{32,64}x{32,64}.npy` + `auto_bgr_feat_5roi.npy` + `ground_truth.txt`）：

```bash
export RPPG_ROOT=$HOME/rppg         # 数据/缓存的根目录（默认 ~/rppg，可自定义）

python src/preprocess_pure.py       # PURE  zip → $RPPG_ROOT/datasets/combined
python src/preprocess_mmpd.py       # MMPD .mat → $RPPG_ROOT/datasets/combined
# UBFC 的 5-ROI 特征在首次构建离线缓存时由 dataset.py 自动抽取, 无需单独脚本
```

首次训练会自动构建离线缓存（`$RPPG_ROOT/output/cache*`，约 3 GB），之后训练直接复用。

**全局 subject id 命名空间**（`config.py`）：UBFC = 1–49，MMPD = 51–83（官方 id + 50），PURE = 101–110（官方 id + 100）。
划分均为 **subject-independent**：

| 数据集 | train | val | test |
| --- | --- | --- | --- |
| UBFC | 1–34 | 35–39 | 40–49 |
| PURE | 101–106 | 107–108 | 109–110 |
| MMPD | 51–72 | 73–77 | 78–83 |

---

## 4. 复现实验

所有实验均为 300 帧窗口（30 fps → 10 s）。**评测统一使用 step-30 重叠窗口**（UBFC 481 窗、PURE 696 窗、MMPD test 4674 窗）。

```bash
# ① 主实验: UBFC+PURE 联合训练（对应论文 UBFC 0.87 / PURE 0.31）
CUDA_VISIBLE_DEVICES=0 python src/run_ubfc_pure.py

# ② 对照: 仅用 UBFC 训练（0.87 → 1.04）
CUDA_VISIBLE_DEVICES=0 python src/run_ubfc_only.py

# ③ MMPD 主行: 从①的模型微调（论文表 1/2/4 的 13.69）
CUDA_VISIBLE_DEVICES=0 python src/run_mmpd_dual_pt_v30.py

# ④ UBFC 消融（config.py 里切开关）
CUDA_VISIBLE_DEVICES=0 python src/run_ubfc_ablate.py no_cob         # w/o COB
CUDA_VISIBLE_DEVICES=0 python src/run_ubfc_ablate.py real_ssm       # 复数 → 实数 SSM
CUDA_VISIBLE_DEVICES=0 python src/run_ubfc_ablate.py no_outband     # w/o 带外能量约束
CUDA_VISIBLE_DEVICES=0 python src/run_ubfc_ablate.py bpm_supervision # 标量 HR 监督

# ⑤ MMPD 消融（均从①的模型微调）
CUDA_VISIBLE_DEVICES=0 python src/run_mmpd_dual_real_pt.py          # 实数 SSM
CUDA_VISIBLE_DEVICES=0 python src/run_mmpd_dual_nores_pt.py         # w/o 残差
CUDA_VISIBLE_DEVICES=0 python src/run_mmpd_dual_nocob_pt.py         # w/o COB
CUDA_VISIBLE_DEVICES=0 python src/run_mmpd_faceonly_pt.py           # 仅面部支路

# ⑥ 基线复现
CUDA_VISIBLE_DEVICES=0 python src/run_physnet_ubfc_pure_r2.py       # PhysNet (ours)

# ⑦ 主脚本: MMPD face_roi-only（64×64）+ 自带评测
FACE_ONLY=1 CUDA_VISIBLE_DEVICES=0 python src/main.py
```

### 关键超参数

| 项 | 联合 UBFC+PURE | MMPD 微调 |
| --- | --- | --- |
| 优化器 | AdamW | AdamW |
| 初始学习率 | 1e-4 | 5e-5 |
| 权重衰减 | 5e-3 | 5e-3 |
| 批大小 | 16 | 微批 4 × 梯度累积 4（等效 16） |
| epoch | 40 | 40 |
| LR 调度 | warmup 5 epoch + cosine 衰减 | 同左 |
| 梯度裁剪 | 0.5（范数） | 0.5 |
| dropout | 0.40 | 0.40 |
| 随机种子 | 42 | 42 |
| 训练窗口步长 | 10 帧 | 30 帧 |
| 人脸 ROI | 32×32 | 64×64 |

### 损失与模型要点

- 三个信号域损失，权重 **2.0 / 3.0 / 0.3**：负 Pearson 波形损失、心脏带频谱 L1、带外能量惩罚；**不使用标量 HR 回归头**（`BPM_LOSS_W=0`、`HR_HEAD_LOSS_W=0`）。
- 复数状态空间：`d_model=256`、2 个 SSM block、`seq_len=300`；A 矩阵由 **COB**（cardiac oscillator bank）初始化为 0.7–3.0 Hz 的阻尼振荡器，特征频率可训练。
- 门控融合：面部特征与色度特征按 `g⊙v + (1−g)⊙c` 融合（real 支路用 CHROM、imag 支路用 POS）。
- 频域 FFN：对时间维 rFFT → 逐频点可学习复增益 → irFFT → 残差 + LayerNorm（共 3 处，2.4 k 参数）。
- 输出为 CHROM 残差形式：`p̂ = CHROM + r`。
- 心率由输出频谱在 0.7–3.0 Hz 带内峰值拾取得到；HR 标签由 BVP 波形导出。

---

## 5. 论文数字 ↔ 评测脚本 ↔ 日志

下表把论文中每个数字对应到 `logs/` 里的哪份日志。一份日志可能包含多次评测，每次评测前都会先打印本次加载的权重和测试集，再给出 MAE 等指标，例如 `eval_ubfc_step30.out` 中的一段：

```text
===== w/o COB (RANDOM_A) =====
[INFO] Loaded: remotessm_ubfc_ablate_no_cob_best.pth | Params: 0.52M
[INFO] Test: 481 samples | subjects [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
MAE: 0.59  RMSE: 0.90  StdErr: 0.68  Pearson: 0.9957
```

因此即使没有权重文件，也能从日志判断每个数值出自哪个实验。

| 论文位置 | 数值 | 评测脚本 | 日志 |
| --- | --- | --- | --- |
| 表 1/2：UBFC | 0.87 | `_eval_step30.py` | `eval_step30.out` |
| 表 1/2：PURE | 0.31 | `_eval_step30.py` | `eval_step30.out` |
| 表 1/2/4：MMPD | **13.69** | `eval_new_ablations_step30.py` | `eval_new_ablations_step30.out` |
| §4.4 UBFC-only 对照 | 1.04 | `_eval_step30.py` | `eval_step30.out` |
| 表 3：full | 0.87 | `eval_ubfc_step30.py` | `eval_ubfc_step30.out` |
| 表 3：实数 SSM | 1.30 | `eval_new_ablations_step30.py` | `eval_new_ablations_step30.out` |
| 表 3：w/o 带外约束 | 0.97 | `eval_ubfc_step30.py` | `eval_ubfc_step30.out` |
| 表 3：w/o COB | 0.59 | `eval_ubfc_step30.py` | `eval_ubfc_step30.out` |
| 表 3：标量 HR 监督 | 23.28 | `eval_ubfc_step30.py` | `eval_ubfc_step30.out` |
| 表 4：仅面部, 复数 | 25.33 | `_eval_faceonly30.py` | `evalfc_step30.out` |
| 表 4：仅面部, 实数 | 28.23 | `_eval_real30.py` | `evalfc_step30.out` |
| 表 4：双路, 实数 | 13.74 | `run_mmpd_dual_real_pt.py`（自带评测） | `mmpd_dual_real_pt.out` |
| 表 4：双路, w/o 残差 | 13.31 | `run_mmpd_dual_nores_pt.py`（自带评测） | `mmpd_dual_nores_pt.out` |
| 表 4：双路, w/o COB | 17.60 | `eval_new_ablations_step30.py` | `eval_new_ablations_step30.out` |
| 跨域：UBFC-only → MMPD | 19.72 | `_verify_step30.py` | `verify_step30.out` |
| 跨域：联合 → MMPD | 20.76 | `_eval_step30.py` | `eval_step30.out` |
| 基线 PhysNet (ours) | 1.63 / 1.13 | `run_physnet_ubfc_pure_r2.py` | `physnet_ubfc_pure_r2.out` |
| 基线 PhysMamba | 2.90 / 7.04 | （临时脚本，未归档） | `physmamba_ubfc_pure.out` |
| 基线 RhythmMamba | 7.03 / 3.38 | （临时脚本，未归档） | `rhythmmamba_ubfc_pure.out` |

补充说明：

- MMPD 主行还有一版**旧选点**结果 13.55（见 `eval_step30.out` 末段，训练时按 val_step=300 选点），已被 13.69（val_step=30 统一口径）取代，保留仅供追溯；其训练日志 `mmpd_dual_pt.out` 末尾的 step=300 自评测为 14.43。
- UBFC 消融另有 `w/o 残差`（0.70）与 OOD 组（`eval_leave_step30.out`），未进正文表格。
- `ubfc_quick.out` / `ubfc_pure_quick.out` 为调试用的快速短跑，非论文结果。
- PhysMamba / RhythmMamba 的模型定义在 `src/`，但其训练入口当时为临时脚本、未归档；PhysNet 的复现脚本已归档。

---

## 6. 说明

- **文件命名**：源码统一使用项目命名——`main.py`（主脚本）、`model.py`（模型定义 `RemoteSSM`）、`dataset.py`、`train.py`、`evaluate.py`；`logs/` 中日志的文件名即实验名，与第 5 节表格一一对应。
- **路径可移植性**：数据/缓存根目录由环境变量 `RPPG_ROOT` 驱动（默认 `~/rppg`），见第 3 节；`config.py`、`preprocess_mmpd.py`、`preprocess_pure.py` 中所有路径均基于该变量，不含任何机器相关写死路径。
- **未随仓库分发的内容**：**实验权重**（各实验训练产出 `remotessm_<实验名>_best.pth`，因体积原因未上传，可按第 4 节自行训练得到）、原始数据集（公开、体积大）、离线缓存（可由第 3 节重建）、论文图源与绘图脚本/绘图数据、第三方基线（PhysNet / PhysMamba / RhythmMamba）的原始仓库与预训练权重、训练过程中的 `*_final.pth` / `*_resume.pth`。
- **基线的第三方代码**：`physnet_model.py`、`physmamba_model.py`、`rhythmmamba_model.py`、`vendor_bimamba.py` 为对应方法作者公开实现的移植/改写，版权归原作者，仅供学术复现使用。
