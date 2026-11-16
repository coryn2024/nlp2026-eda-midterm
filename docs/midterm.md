# 中期进展记录

论文：Wei 与 Zou，2019，EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks。[论文](https://arxiv.org/abs/1901.11196) · [作者代码](https://github.com/jasonwei20/eda_nlp)

## 原文目标位置

第 4.1 节表 2，训练样本数为 500 的那一列。要检验的是：训练集较小时，EDA 是否提高文本分类准确率。该列是五个数据集上 CNN 与 RNN、五个种子的平均准确率，无 EDA 为 76.9%，有 EDA 为 79.9%。模型结构见第 9.1 节。500 条的增强参数见表 3：四种操作的 α 均为 0.05，每条原句生成 16 条，标签不变。

## 本组设置差异

| 项目 | 原文 | 本组 |
|---|---|---|
| 数据集 | 五个数据集平均 | 只做 SST-2 |
| 训练样本 | 500 条 | 400 条训练，100 条早停验证 |
| EDA 数据 | 原句加增强句 | 仅上述 400 条，共 6800 行 |
| 评测集 | SST-2 测试集 1752 条 | GLUE dev，872 条 |
| 增强参数 | α = 0.05，每条 16 句 | 相同 |
| 词向量 | GloVe 300 维 | GloVe 6B 300 维，冻结 |
| 框架 | Keras | PyTorch 2.7 |
| CNN | 第 9.1 节 | 按第 9.1 节实现 |
| RNN 输出 | softmax，交叉熵 | Sigmoid，二元交叉熵 |
| 优化与种子 | 早停 patience = 3，五个种子 | Adam，学习率 0.001，batch size 32，初始化 N(0, 0.05)，种子 42 到 46 |

因此本组准确率不能与表 2 的 76.9% 和 79.9% 直接比较。数据划分、词向量和复核命令见 [README](../README.md)。

## 已有并列结果

已完成种子 42、43、44、45、46。评测集 872 条。下表为五个种子的平均准确率 ± 样本标准差，由本组逐条预测重新算出。分数为百分数。

| 实验 | 准确率 | 五个种子的情况 |
|---|---|---|
| 原文表 2，五个数据集 | 76.9% / 79.9% | 论文报告有 EDA 更高 |
| CNN，无增强 | 72.02% ± 0.49% | 作为对照 |
| CNN，有增强 | 72.20% ± 0.87% | 三个上升，两个下降，平均高 0.18 个百分点 |
| RNN，无增强 | 71.54% ± 1.34% | 作为对照 |
| RNN，有增强 | 68.85% ± 1.69% | 五个种子全部下降，平均低 2.69 个百分点 |

我们发现，按文本规范化后，872 条里有 46 条与训练或验证重合。去掉这些句子后，CNN 无增强为 70.77% ± 0.51%，有增强为 70.94% ± 0.91%。方向不变，所以上表仍用全部 872 条。

CNN 汇总：[summary.json](../results/cnn/summary.json) · [results_by_seed.csv](../results/cnn/results_by_seed.csv)。RNN 汇总：[rnn_mean_std_summary.csv](../results/rnn/rnn_mean_std_summary.csv) · [rnn_all_seeds_summary.csv](../results/rnn/rnn_all_seeds_summary.csv)。

## 本组运行生成的逐条输出链接

每份 872 行，字段为 id、text、gold_label、pred_label、prob_0、prob_1。标签 0 为负面，标签 1 为正面。下面二十个链接都是本组运行生成的逐条预测。

| 种子 | CNN 无增强 | CNN 有增强 | RNN 无增强 | RNN 有增强 |
|---|---|---|---|---|
| 42 | [预测](../results/cnn/predictions_cnn_baseline_seed42.csv) | [预测](../results/cnn/predictions_cnn_eda_seed42.csv) | [预测](../results/rnn/predictions_rnn_baseline_seed42.csv) | [预测](../results/rnn/predictions_rnn_eda_seed42.csv) |
| 43 | [预测](../results/cnn/predictions_cnn_baseline_seed43.csv) | [预测](../results/cnn/predictions_cnn_eda_seed43.csv) | [预测](../results/rnn/predictions_rnn_baseline_seed43.csv) | [预测](../results/rnn/predictions_rnn_eda_seed43.csv) |
| 44 | [预测](../results/cnn/predictions_cnn_baseline_seed44.csv) | [预测](../results/cnn/predictions_cnn_eda_seed44.csv) | [预测](../results/rnn/predictions_rnn_baseline_seed44.csv) | [预测](../results/rnn/predictions_rnn_eda_seed44.csv) |
| 45 | [预测](../results/cnn/predictions_cnn_baseline_seed45.csv) | [预测](../results/cnn/predictions_cnn_eda_seed45.csv) | [预测](../results/rnn/predictions_rnn_baseline_seed45.csv) | [预测](../results/rnn/predictions_rnn_eda_seed45.csv) |
| 46 | [预测](../results/cnn/predictions_cnn_baseline_seed46.csv) | [预测](../results/cnn/predictions_cnn_eda_seed46.csv) | [预测](../results/rnn/predictions_rnn_baseline_seed46.csv) | [预测](../results/rnn/predictions_rnn_eda_seed46.csv) |

可用 [verify_predictions.py](../code/D/verify_predictions.py) 按这些文件重算上表准确率。

## 两例失败或边界

两例都来自种子 42 的评测预测，金标为 0。

编号 735，句末否定。原句：the experience of going to a film festival is a rewarding one ; the experiencing of sampling one through this movie is not . 句末 not 把评价转成负面。我们发现，RNN 在无增强时预测为 0，正面概率 0.336；有增强时预测为 1，正面概率 0.522。CNN 两组都预测为 1，正面概率由 0.503 升到 0.674。增强没有帮助模型抓住句末否定。[CNN 无增强第 737 行](../results/cnn/predictions_cnn_baseline_seed42.csv#L737) · [CNN 有增强第 737 行](../results/cnn/predictions_cnn_eda_seed42.csv#L737) · [RNN 无增强第 737 行](../results/rnn/predictions_rnn_baseline_seed42.csv#L737) · [RNN 有增强第 737 行](../results/rnn/predictions_rnn_eda_seed42.csv#L737)

编号 544，anything but。原句：although huppert 's intensity and focus has a raw exhilaration about it , the piano teacher is anything but fun . anything but 表示否定。我们发现，无增强时两种模型都预测为 0，正面概率分别是 0.452 和 0.381；有增强时都预测为 1，正面概率分别是 0.886 和 0.560。[CNN 无增强第 546 行](../results/cnn/predictions_cnn_baseline_seed42.csv#L546) · [CNN 有增强第 546 行](../results/cnn/predictions_cnn_eda_seed42.csv#L546) · [RNN 无增强第 546 行](../results/rnn/predictions_rnn_baseline_seed42.csv#L546) · [RNN 有增强第 546 行](../results/rnn/predictions_rnn_eda_seed42.csv#L546)

## 下一项单因素实验

数据划分、两种模型、种子、学习率，以及同义替换、随机插入、随机交换的 α = 0.05 和每条 16 句都保持不变，只把随机删除的 α 从 0.05 改为 0.5。对应第 4.4 节：删除比例过高时句子难以保持原意。此项尚未运行。
