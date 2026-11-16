EDA 中期提交包

中期进展记录.docx 和 中期进展记录.pdf 与本说明放在同一层。
本包同时包含 SST-2 数据划分、EDA 增强结果、两种模型的代码，以及 20 份逐条预测。
解压后运行 code 文件夹中的 verify_predictions.py，会重新计算四组准确率，并写入 results 文件夹中的 verification.json。

目录
- 中期进展记录.docx、中期进展记录.pdf：进展记录
- data：500 条原训练、400 条训练和 100 条验证、6800 行增强训练集、872 条评测集
- code 中的 augment.py 与 data_utils.py：数据与增强
- code 中的 train_cnn.py：卷积神经网络
- code 中的 train_rnn_torch.py：循环神经网络
- code 中的 verify_predictions.py：根据逐条预测重算准确率
- results 下的 cnn：卷积神经网络的十份逐条预测和汇总
- results 下的 rnn：循环神经网络的十份逐条预测、训练记录和曲线

未放入本包
- GloVe 6B 300 维，约 1GB。下载地址：https://nlp.stanford.edu/data/glove.6B.zip
  解压后使用 glove.6B.300d.txt。重训时设置环境变量 GLOVE_PATH。
- 模型权重 best_model.pt。准确率以 results 里的逐条预测为准，不需要权重即可复核。

预测文件字段：id, text, gold_label, pred_label, prob_0, prob_1。每份 872 行。
