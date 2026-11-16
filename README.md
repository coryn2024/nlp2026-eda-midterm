# EDA 在 SST-2 上的课程实验

中期进展、结果与误判分析见 [docs/midterm.md](docs/midterm.md)。

## 固定资源与实验设置

- 数据为 SST-2。`data/sst2_train_500.txt` 是作者仓库中的 500 条训练样本。按 seed 42 划分为 `data/train_400.txt` 与 `data/val_100.txt`。增强只作用于这 400 条，结果为 `data/train_eda_final.txt`，6800 行。评测集是 GLUE SST-2 的 dev，`data/sst2_dev_872.tsv`，872 条。作者论文中的 1752 条测试集不在公开仓库中，因此本组数字不能与表 2 直接比较。
- 增强使用作者的 `code/A/augment.py` 与 `code/A/eda.py`。四种操作的 α 都是 0.05，每条原句生成 16 条增强句，并保留清洗后的原句。`code/A/eda.py` 中 `random.seed(1)` 是固定的。
- 词向量为 GloVe 6B 300 维，冻结。文件约 1GB，未放入本仓库。下载地址：https://nlp.stanford.edu/data/glove.6B.zip ，使用其中的 `glove.6B.300d.txt`。重训时设置环境变量 `GLOVE_PATH`。
- CNN 按论文第 9.1 节实现：Conv1D 128 个滤波器、核宽 5、全局最大池化、20 维 ReLU、二分类。RNN 为 BiLSTM 64、Dropout 0.5、BiLSTM 32、Dropout 0.5、20 维 ReLU；输出是单值 Sigmoid 加二元交叉熵，与论文的 softmax 加交叉熵不同。两者都用 Adam，学习率 0.001，batch size 32，初始化 N(0, 0.05)，早停 patience 为 3，种子 42 到 46。
- 模型权重未放入本仓库。准确率以 `results` 中的逐条预测为准。

## 复核

在仓库根目录执行：

```
python code/D/verify_predictions.py
```

脚本读取 20 份逐条预测，重新计算四组准确率和样本标准差，并写入 `results/verification.json`。不需要 GloVe，也不需要模型权重。

## 已有结果的位置

- `results/cnn`：十份逐条预测、按种子汇总和训练日志
- `results/rnn`：十份逐条预测、汇总、每次实验的配置和训练曲线

预测文件字段：id, text, gold_label, pred_label, prob_0, prob_1。每份 872 行。
