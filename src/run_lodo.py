"""leave-one-dataset-out 跨域实验 runner.

用法 (每轮一个独立进程, 通过 CUDA_VISIBLE_DEVICES 分配 GPU):
  CUDA_VISIBLE_DEVICES=0 python run_lodo.py leave_ubfc
  CUDA_VISIBLE_DEVICES=1 python run_lodo.py leave_pure
  CUDA_VISIBLE_DEVICES=2 python run_lodo.py leave_mmpd

每个配置:
  - train/val 来自另外两个数据集 (subject-independent 划分)
  - 整块留出的数据集作为 test (完全 unseen)
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from train import train_model
from evaluate import evaluate

# 数据集全局 subject id 命名空间
SPLIT = {
    "ubfc": {"train": list(range(1, 35)), "val": list(range(35, 40)), "test": list(range(40, 50))},
    "mmpd": {"train": list(range(51, 73)), "val": list(range(73, 78)), "test": list(range(78, 84))},
    "pure": {"train": list(range(101, 107)), "val": list(range(107, 109)), "test": list(range(109, 111))},
}
# 完整数据集 subject id (test = 整块留出)
ALL = {
    "ubfc": list(range(1, 50)),
    "mmpd": list(range(51, 84)),
    "pure": list(range(101, 111)),
}

CONFIGS = {}
for _held in ["ubfc", "mmpd", "pure"]:
    _others = [k for k in SPLIT if k != _held]
    CONFIGS[f"leave_{_held}"] = {
        "train": sum((SPLIT[k]["train"] for k in _others), []),
        "val": sum((SPLIT[k]["val"] for k in _others), []),
        "test": ALL[_held],
    }


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIGS:
        print("usage: python run_lodo.py {" + ",".join(CONFIGS) + "}")
        sys.exit(1)
    name = sys.argv[1]
    cfg = CONFIGS[name]
    out_dir = os.path.join(PROJ, "outputs", name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[LODO] {name}: train={len(cfg['train'])} val={len(cfg['val'])} test={len(cfg['test'])}")
    print(f"[LODO] visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    # 训练 (内部会写 remotessm_{tag}_*.pth/.log 到 out_dir)
    train_model(train_subjects=cfg["train"], val_subjects=cfg["val"],
                out_dir=out_dir, tag=name)

    # 评估 held-out 测试集
    best_path = os.path.join(out_dir, f"remotessm_{name}_best.pth")
    evaluate(model_path=best_path, test_subjects=cfg["test"])


if __name__ == "__main__":
    main()