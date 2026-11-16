# A 模块交付说明：数据准备、EDA 增强与 GloVe 预处理

**编写人：** A 模块负责人  
**交付对象：** B（CNN 实验）、C（RNN 实验）、D（统一评测与中期记录）  
**项目：** EDA 论文复现（SST-2，500 条训练样本，CNN/RNN 有无 EDA 四组对比）

---

## 1. 我做了什么

A 模块负责为 B/C 提供统一、干净、可直接训练的数据和预处理接口。具体完成了以下工作：

1. 从官方仓库 `jasonwei20/eda_nlp` 获取 `data/sst2_train_500.txt`，共 500 条 SST-2 训练样本，格式为 `标签\t句子`，标签为 0/1。
2. 使用官方 `code/augment.py` 对 500 条原句做 EDA 增强，参数为四种操作 α 均为 0.05、`num_aug=16`，输出 `data/sst2_train_500_eda.txt`，共 8500 行（500 × 17，每条原句保留 1 条原句 + 16 条增强）。
3. 从 GLUE 官方下载 SST-2 dev 集（872 条带标签样本），保存为 `data/sst2_dev_872.tsv`，作为最终评测集，全程只评一次，不参与训练、早停或调参。
4. 对原始 500 条按 80/20 分层划分（`random_state=42`），得到 400 条训练池和 100 条早停验证集。训练池写入 `data/train_400.txt`，验证集写入 `data/val_100.txt`（纯原句，不含任何增强样本）。
5. 按训练池索引从全量增强文件中抽取 400 × 17 = 6800 行，写入 `data/train_eda_final.txt`，作为 +EDA 组的训练集。抽取逻辑保证验证集对应的原句及其增强样本不会进入训练集。
6. 下载 GloVe 300d 词向量，文件位于 `C:\Users\11348\Desktop\glove\glove.6B.300d.txt`，约 1.04 GB。
7. 编写 `data_utils.py`，提供统一的 `load_splits()` 接口，完成分词、词表构建、GloVe 加载、OOV 处理、padding，并同时返回 Baseline 组（400 条）和 EDA 组（6800 条）的训练数据，两组共用同一个词表和 embedding。
8. 编写并运行 `split_and_extract.py` 和 `make_train400.py`，生成上述划分文件。这两个脚本已运行完毕，保留在仓库中备查，B/C 不需要重新运行。

所有数据文件、代码和本说明均位于仓库根目录 `C:\Users\11348\Desktop\eda_nlp-master\eda_nlp-master` 下，GloVe 文件单独放在 `C:\Users\11348\Desktop\glove` 下。

---

## 2. 交付文件清单：位置、来源、格式与用途

仓库根目录：`C:\Users\11348\Desktop\eda_nlp-master\eda_nlp-master`。以下相对路径均相对该目录。

| 文件 | 行数 | 格式 | 来源 | 用途 |
|---|---|---|---|---|
| `data/sst2_train_500.txt` | 500 | `标签\t句子` | 官方仓库 `jasonwei20/eda_nlp` 的 `data/` 目录 | 原始训练集，仅作来源留存 |
| `data/sst2_train_500_eda.txt` | 8500 | `标签\t句子` | 由 `augment.py` 对 500 条原句增强生成 | 全量增强集，**不要直接用于训练** |
| `data/train_eda_final.txt` | 6800 | `标签\t句子` | 从全量增强集中按训练池索引抽取 400×17 行 | **+EDA 组训练集** |
| `data/train_400.txt` | 400 | `标签\t句子` | 从原始 500 条中按 `random_state=42` 分层抽取 | **Baseline 组训练集** |
| `data/val_100.txt` | 100 | `标签\t句子` | 从原始 500 条中分层划出的验证集，纯原句 | **早停验证集，两组共用** |
| `data/sst2_dev_872.tsv` | 873（含 1 行表头） | `sentence\tlabel` | GLUE 官方 SST-2.zip 中的 `dev.tsv` | **最终评测集，只评一次** |
| `data/SST-2/` 与 `data/SST-2.zip` | — | — | 下载/解压残留 | 可删除，不影响数据 |
| `data_utils.py` | — | Python 模块 | A 编写 | **统一预处理接口，B/C 必须使用** |
| `split_and_extract.py` | — | Python 脚本 | A 编写并已运行 | 保留备查，B/C 不需要重跑 |
| `make_train400.py` | — | Python 脚本 | A 编写并已运行 | 保留备查，B/C 不需要重跑 |
| `A模块交付说明.md` | — | Markdown | 本文件 | 交付说明 |

GloVe 文件：

| 文件 | 路径 | 大小 | 说明 |
|---|---|---|---|
| `glove.6B.300d.txt` | `C:\Users\11348\Desktop\glove\glove.6B.300d.txt` | 约 1.04 GB | 300 维 GloVe 词向量 |

标签取值均为 `0`（负面）/ `1`（正面）。SST-2 文本已按空格预分词，不需要再调用 NLTK 分词器。

---

## 3. 数据来源与生成方式（怎么来的）

### 3.1 原始训练集

从官方仓库 `https://github.com/jasonwei20/eda_nlp` 获取 `data/sst2_train_500.txt`。该文件为论文作者提供的 500 条 SST-2 训练样本，每行 `标签\t句子`，标签为 0/1。

### 3.2 EDA 增强

在仓库根目录执行以下命令（已执行，B/C 不需要重跑）：

```powershell
python code/augment.py --input=data/sst2_train_500.txt --output=data/sst2_train_500_eda.txt --num_aug=16 --alpha_sr=0.05 --alpha_ri=0.05 --alpha_rs=0.05 --alpha_rd=0.05

说明：

augment.py 没有统一的 --alpha 参数，只有四个分开的参数，默认值都是 0.1。必须四个都显式传 0.05，否则会按 0.1 跑，与论文 Table 3 不一致。

num_aug=16 时，官方 eda() 函数会把原句也追加到输出中，因此每条原句生成 17 行（16 条增强 + 1 条原句）。500 × 17 = 8500 行。

输出保留标签，格式为 标签\t句子，不需要额外补标签。

同义词来源为 NLTK WordNet。本机已下载 wordnet 和 omw-1.4

### 3.3 最终评测集
从 GLUE 官方地址下载：

text
https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
解压后使用 SST-2/dev.tsv，共 872 条带标签样本，复制为 data/sst2_dev_872.tsv。该文件包含 1 行表头，实际数据 872 行，总行数 873。该文件全程只用于最终评测一次，未参与训练、早停、调参或词表构建。

3.4 训练/验证划分
对原始 500 条按 80/20 分层划分，random_state=42：

训练池 400 条 → data/train_400.txt

验证集 100 条 → data/val_100.txt

全量增强文件 8500 行的排列为“第 i 条原句对应第 17*i 到 17*i+16 行”。按训练池索引从增强文件中抽取 400 × 17 = 6800 行，写入 data/train_eda_final.txt。验证集只使用原始 100 条原句，不含任何增强样本。

实际标签分布：

train_400.txt：{0: 195, 1: 205}
val_100.txt：{1: 51, 0: 49}

两组分布接近平衡，但并非严格 50/50。这是 500 条上分层抽样的正常结果，不要为凑 50/50 重新划分。

3.5 GloVe 词向量
从 Stanford 下载：

text
https://nlp.stanford.edu/data/glove.6B.zip
解压后使用 glove.6B.300d.txt，放在 C:\Users\11348\Desktop\glove\glove.6B.300d.txt。约 1.04 GB，确认文件完整。

4. 环境配置
4.1 Python 与依赖
A 模块使用的环境为 Python 虚拟环境（venv），已安装：

powershell
pip install nltk numpy scikit-learn
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
如果 NLTK 下载报 Security Violation（代理环境常见），先执行：

powershell
$env:NLTK_ALLOW_PROXIED_URLOPEN=1
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
验证 WordNet 可用：

powershell
python -c "from nltk.corpus import wordnet as wn; print(wn.synsets('good')[:3])"
能打印 synset 列表即可。B/C 训练阶段不需要 NLTK，只有复现 EDA 时才需要。

4.2 不要直接拷贝 venv 文件夹
A 的 venv 文件夹中的路径是写死的，换电脑后 python.exe 和 site-packages 路径全部失效。请在你自己电脑上重新建环境：

powershell
cd C:\path\to\eda_nlp-master
python -m venv venv
venv\Scripts\activate
pip install numpy scikit-learn nltk
B/C 训练模型所需的深度学习框架（TensorFlow 或 PyTorch）自行选择并安装。CNN 和 RNN 两组必须使用同一个框架、同一版本，否则对比不公平。

4.3 GloVe 路径配置
data_utils.py 中 GloVe 路径写法为：

python
GLOVE_PATH = os.environ.get(
    "GLOVE_PATH",
    r"C:\Users\11348\Desktop\glove\glove.6B.300d.txt",
)
环境变量优先，没设则使用默认值。换电脑后默认值一定失效直接修改 data_utils.py 第 7 行的默认路径为实际路径。

如果路径错误，load_glove 会抛出 FileNotFoundError，按提示修改即可。

5. 统一预处理接口 data_utils.py
B/C 必须通过 data_utils.py 取数据，不要各自重写分词、词表或 padding。否则 CNN 和 RNN 输入不一致，四组对比不公平。

使用方式：

python
from data_utils import load_splits

d = load_splits()   # 默认读取 data/ 下五个文件，GloVe 路径见第 4.3 节

d["vocab"]                  # dict，PAD=0，UNK=1
d["embedding"]              # np.ndarray, shape (3748, 300)

# Baseline 组（CNN Baseline / RNN Baseline）
d["baseline_train_x"]       # (400, 50)
d["baseline_train_y"]       # (400,)

# EDA 组（CNN + EDA / RNN + EDA）
d["train_x"]                # (6800, 50)
d["train_y"]                # (6800,)

# 早停验证集（两组共用）
d["val_x"]                  # (100, 50)
d["val_y"]                  # (100,)

# 最终评测集（两组共用，只评一次）
d["test_x"]                 # (872, 50)
d["test_y"]                 # (872,)

# 原始句子列表，用于写逐条预测文件
d["baseline_train_texts"]   # list[str], 长度 400
d["train_texts"]            # list[str], 长度 6800
d["val_texts"]              # list[str], 长度 100
d["test_texts"]             # list[str], 长度 872
实际运行输出（已在本机验证）：

text
baseline (400, 50)
eda (6800, 50)
val (100, 50)
test (872, 50)
emb (3748, 300)
预处理规则已固定，不要修改：

分词：按空格切分，SST-2 已预分词。

MAX_LEN = 50：超长截断，短句用 <PAD> 补齐。

<PAD> = 0：向量为零向量。

<UNK> = 1：GloVe 中不存在的词映射为 <UNK>，向量为小随机初始化（np.random.normal(0, 0.1)）。

词表：仅从 train_eda_final.txt 构建，不包含验证集或评测集词汇，避免信息泄漏。

两组共用词表和 embedding：Baseline 组和 EDA 组使用同一个 vocab 和 embedding，保证唯一变量是训练样本数量与是否增强。这是刻意选择，不要给两组分别建词表。

6. 四组实验对应关系与使用方式
实验	模型	训练集	早停	评测
CNN Baseline	论文 9.1 节 CNN	baseline_train_x/y（400 条）	val_x/y	test_x/y
CNN + EDA	同上	train_x/y（6800 条）	val_x/y	test_x/y
RNN Baseline	论文 9.1 节 RNN	baseline_train_x/y（400 条）	val_x/y	test_x/y
RNN + EDA	同上	train_x/y（6800 条）	val_x/y	test_x/y
铁律：

早停只能用 val_x / val_y。禁止用 test_x / test_y 早停或调参。

test_x / test_y 只在训练结束后评一次，报告准确率并保存逐条预测文件。

Baseline 和 EDA 组除训练集外，模型结构、初始化、优化器、学习率、batch size、早停 patience、随机种子、训练轮数上限必须完全一致。

第一轮统一用随机种子 42。时间允许再补 5 个种子取平均，每个种子都保存预测文件。

每组实验保存完整训练日志（loss/accuracy 曲线、早停触发 epoch、最终 epoch 数），D 整理中期记录时需要。

模型结构参考论文第 9.1 节：

CNN：

text
输入 (50,) 词索引
→ Embedding (3748, 300)，用 d["embedding"] 初始化
→ Conv1D：128 个滤波器，kernel_size=5
→ GlobalMaxPooling1D
→ Dense(20, ReLU)
→ Dense(1, Sigmoid)
损失用二元交叉熵，优化器 Adam，正态分布随机初始化，EarlyStopping patience=3，监控 val_loss。Embedding 冻结还是微调，两组必须一致并在日志中写明。

RNN：

text
输入 (50,) 词索引
→ Embedding (3748, 300)，用 d["embedding"] 初始化
→ Bidirectional(LSTM(64))
→ Dropout(0.5)
→ Bidirectional(LSTM(32))
→ Dropout(0.5)
→ Dense(20, ReLU)
→ Dense(1, Sigmoid)
损失、优化器、初始化、EarlyStopping 同上。Embedding 策略与 CNN 组保持一致。

论文未指定 batch size 和学习率，B/C 自行选择并在日志中写清楚，两组必须一致。建议 batch_size=32 或 64，lr=1e-3 作为起点。

7. 逐条预测文件格式
四组实验各保存一份 CSV：

csv
id,text,gold_label,pred_label,prob_0,prob_1
0,"it 's a charming and often affecting journey .",1,1,0.12,0.88
1,"unflinchingly bleak and desperate",0,0,0.91,0.09
...
id：0–871，对应 test_texts 下标。

text：原始评测句子，CSV 标准转义。

gold_label：test_y 中的真实标签。

pred_label：模型预测标签（0/1）。

prob_0 / prob_1：sigmoid 输出的两类概率，保留 4–6 位小数。

建议命名：

text
predictions_cnn_baseline_seed42.csv
predictions_cnn_eda_seed42.csv
predictions_rnn_baseline_seed42.csv
predictions_rnn_eda_seed42.csv
提交给 D 时附上可访问链接、准确率及训练日志。

8. 怎么查看与验证
B/C 拿到文件夹后，在仓库根目录激活环境，先运行以下命令确认数据接口正常：

powershell
python -c "from data_utils import load_splits; d=load_splits(); print('baseline', d['baseline_train_x'].shape); print('eda', d['train_x'].shape); print('val', d['val_x'].shape); print('test', d['test_x'].shape); print('emb', d['embedding'].shape)"
预期输出：

text
baseline (400, 50)
eda (6800, 50)
val (100, 50)
test (872, 50)
emb (3748, 300)
如果报 FileNotFoundError 且指向 GloVe，按第 4.3 节修改 GLOVE_PATH。

再确认文件行数：

powershell
(Get-Content data\train_400.txt).Count
(Get-Content data\train_eda_final.txt).Count
(Get-Content data\val_100.txt).Count
(Get-Content data\sst2_dev_872.tsv).Count
预期分别为 400、6800、100、873。

确认验证集分布：

powershell
python -c "from collections import Counter; print(Counter(int(l.split('\t')[0]) for l in open('data/val_100.txt',encoding='utf-8') if l.strip()))"
预期 Counter({1: 51, 0: 49})。

9. 注意事项（全部必读）
GLOVE_PATH 是绝对路径，换电脑必改。 见第 4.3 节。找不到文件会直接报 FileNotFoundError。

不要用 sst2_train_500_eda.txt 直接训练。 它包含验证集 100 条原句对应的增强样本，会造成验证集信息泄漏。B/C 训练只用 train_eda_final.txt（6800 条）或 train_400.txt（400 条）。

不要用 test_x/test_y 早停或调参。 最终评测集只评一次，早停一律用 val_x/val_y。

不要给 Baseline 和 EDA 组分别建词表。 两组共用 d["vocab"] 和 d["embedding"]，否则对比不公平。data_utils.py 已这样设计，不要改。

split_and_extract.py 中的 val label distribution 打印语句有 bug，会输出 {0: 1, 1: 1}。这是打印逻辑写错（val_idx.count(i) 统计索引出现次数，每个索引只出现一次）。数据本身没问题，实际分布是 {1: 51, 0: 49}，以第 8 节命令输出为准，不要被那行输出误导。

data/SST-2/ 和 data/SST-2.zip 是下载/解压残留，可删除，不影响数据。data/sst2_dev_872.tsv 已复制出来，是评测集的唯一来源。删残留前确认 sst2_dev_872.tsv 是 873 行。

num_aug=16 时每条原句输出 17 行（16 增强 + 1 原句），所以 500×17=8500、400×17=6800。如果看到 8000 或 6400，说明脚本版本不同、没有保留原句，但本仓库已确认是 8500/6800。不要手动删原句或改行数，否则索引对齐会错。

两组训练条件必须一致。 除了训练集不同，模型结构、初始化、优化器、学习率、batch size、早停 patience、随机种子、训练轮数上限都要相同，并在日志中记录。这是四组对比有意义的前提。

augment.py 没有统一的 --alpha 参数，只有四个分开的参数，默认 0.1。必须四个都显式传 0.05。本仓库已按 0.05 跑过，B/C 不需要重跑 EDA。

不要直接拷贝 A 的 venv 文件夹到另一台电脑。 虚拟环境路径写死，换电脑后全部失效。按第 4.2 节重新建环境。

PowerShell 续行符是反引号 `，不是反斜杠 \。 粘贴多行命令时容易出错，建议直接写成一行。

NLTK 代理环境会报 Security Violation。 设置 $env:NLTK_ALLOW_PROXIED_URLOPEN=1 后再下载。B/C 训练不需要 NLTK，只有复现 EDA 时才需要。

评测集 dev.tsv 解压后可能多套一层目录（data/SST-2/SST-2/dev.tsv）。本仓库已从正确位置复制出 data/sst2_dev_872.tsv，B/C 直接用这个文件，不要再去找原始 dev.tsv。

MAX_LEN=50 是固定值，不要在 B/C 各自改动。 CNN 和 RNN 必须用相同的序列长度，否则输入维度不一致，对比不公平。如果确实需要改，两组一起改并记录。

标签是 0/1 二分类，不是 -1/1。 如果模型输出层用 softmax 二分类，注意 prob_0 对应标签 0、prob_1 对应标签 1；如果用 sigmoid，输出是标签 1 的概率，prob_0 = 1 - prob_1。写预测文件时不要弄反。

验证集和 Baseline 训练集的标签分布不是严格 50/50（{1: 51, 0: 49} 和 {0: 195, 1: 205}），这是 500 条上分层抽样的正常结果，不影响实验。不要为凑 50/50 重新划分，重新划分会改变 random_state=42 下的一致性。

data_utils.py 中 load_splits() 同时返回 Baseline 和 EDA 两组训练数据，这是刻意设计。 B/C 不要自己从文件重新读取数据，直接用这个接口，保证四组实验数据完全同源。

10. 交接说明
A 模块的数据准备、EDA 增强、GloVe 预处理已全部完成。B/C 不需要再碰数据准备，直接使用 data_utils.load_splits() 即可开始写 CNN / RNN 训练脚本。D 可以直接使用 data/sst2_dev_872.tsv 和 B/C 提交的预测文件做统一评测。

如果发现数据文件行数、shape 与第 8 节不符，先检查是否误改了文件，再联系 A。如果 GloVe 路径问题无法解决，联系 A 确认原始文件是否完整。本说明中所有预期输出均来自 A 模块本机实际运行，不是估算值。任何数字对不上都说明文件被改动或路径配置有误，先排查再继续。