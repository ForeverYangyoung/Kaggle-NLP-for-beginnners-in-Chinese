# Kaggle-NLP-for-beginnners-in-Chinese
中文翻译与学习笔记：Kaggle经典项目《面向绝对初学者的NLP入门》。旨在帮助中文学习者更好地理解和实践NLP基础知识。
大家好呀，我在kaggle上面看到了一个NLP自然语言处理的入门指南，现在做汉化分享处理。
原文的链接在这里：https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners
开始：

# 引言

在过去几年里，深度学习有一个领域取得了显著的进步，那就是自然语言处理（NLP）。如今，计算机已经能够生成文本、自动进行语言翻译、分析评论、标注句子中的单词等等。

在所有NLP的应用中，**文本分类**可能是最实用、最广泛的一种。简单来说，就是将一份文档自动归入某个类别。这个技术可以用于：

* **情感分析** (例如，判断人们对你的产品是持**积极**还是**消极**评价)
* **作者识别** (判断某份文档最可能是由哪位作者撰写的)
* **法律文件检索** (在庭审中，筛选出哪些文件是与案件相关的)
* **按主题组织文档**
* **筛选收件箱邮件**
* ...以及更多！

分类模型同样可以用来解决一些初看起来并不像分类问题的问题。例如，Kaggle上的 [美国专利术语匹配](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) 比赛。在这个任务中，我们需要比较两个单词或短语，并根据它们在专利分类中的上下文，判断它们的相似度。相似度得分为 `1` 表示两者意思完全相同，`0` 表示意思完全不同。例如，“abatement”（减少）和“eliminating process”（消除过程）的得分为 `0.5`，意味着它们有些相似，但并不完全相同。

事实证明，这个问题可以被转化成一个分类问题。怎么做呢？通过像下面这样重新组织问题：

> 对于以下文本...: "TEXT1: abatement; TEXT2: eliminating process" ...请选择一个相似度类别: "不同; 相似; 相同"。

在这个Notebook中，我们将通过上述类似的方式，把专利术语匹配问题当作一个分类任务来解决。

---

## 关于 Kaggle

Kaggle 对于有志于成为数据科学家或任何希望提升机器学习技能的人来说，是一个绝佳的资源。没有什么比亲身实践并获得实时反馈更能帮助你提升的了。它提供了：

* 有趣的数据集
* 关于你模型表现的实时反馈
* 一个排行榜，让你能看到优秀水平、无限可能和顶尖技术
* 获胜选手的Notebook和博客文章，分享了许多有用的技巧和技术

我们将要使用的数据集只能从Kaggle获取。因此，你需要先在网站上注册，然后进入**[比赛页面](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching)**。在该页面上，点击“Rules”，然后点击“I Understand and Accept”。（虽然比赛已经结束，你不会真的参赛，但你仍然需要同意规则才能下载数据。）

你有两种方式来使用这些数据：

1.  **最简单**: 直接在Kaggle上运行这个Notebook。
2.  **最灵活**: 将数据下载到本地，在你的个人电脑或GPU服务器上运行。

如果你是在 Kaggle.com 上运行，可以跳过下一个部分。只需确保你在Kaggle的会话中已经选择了使用GPU，方法是点击右上角的菜单（三个点），然后点击“Accelerator”，它应该看起来是这样的：
<img width="679" height="580" alt="image" src="https://github.com/user-attachments/assets/f0130d8a-8ed6-408e-a634-86739257d5ca" />


### 在你自己的电脑上使用Kaggle数据

我们需要根据当前运行环境是在Kaggle上还是在本地，来写一些稍有不同的代码。因此，我们用下面这个变量来追踪运行环境：

```python
import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
```

Kaggle 每周会限制你使用GPU的时间。这个限制很宽松，但你可能还是会觉得不够用！在这种情况下，你会想要用自己的GPU服务器，或者像 Colab、Paperspace Gradient、SageMaker Studio Lab 这样的云服务器（它们都有免费方案）。要做到这一点，你就需要能从Kaggle上下载数据集。

下载Kaggle数据集最简单的方法是使用 **Kaggle API**。你可以在 notebook 的单元格里运行下面的命令来用 `pip` 安装它：

```python
!pip install kaggle
```

你需要一个 **API 密钥**才能使用 Kaggle API。要获取密钥，请在Kaggle网站上点击你的头像，选择 “My Account”，然后点击 “Create New API Token”。这会下载一个名为 `kaggle.json` 的文件到你的电脑上。你需要把这个密钥复制到你的GPU服务器。具体操作是：打开你下载的文件，复制里面的内容，然后粘贴到下面这个单元格里（例如，`creds = '{"username":"xxx","key":"xxx"}'`）：

```python
creds = ''
```

然后执行这个单元格（只需要运行一次）：

```python
# 推荐使用 pathlib.Path 来处理Python中的路径
from pathlib import Path

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)
```

现在你就可以从Kaggle下载数据集了。

```python
path = Path('us-patent-phrase-to-phrase-matching')
```

然后使用Kaggle API将数据集下载到指定路径并解压：

```python
if not iskaggle and not path.exists():
    import zipfile,kaggle
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)
```

> **提示**：你可以很方便地从Kaggle下载notebook，然后上传到其他的云服务上。所以，如果你的Kaggle GPU额度快用完了，可以试试这个方法！
>

````markdown
---
## 导入数据与探索性数据分析 (EDA)

```python
# 如果在Kaggle环境中，则设置数据路径并安装datasets库
if iskaggle:
    path = Path('../input/us-patent-phrase-to-phrase-matching')
    ! pip install -q datasets
````

**小结一下哈**：上面两种方法其实就是要不你就在kaggle官网上完成整个测试，要不就在新的环境里面把kaggle上面的数据自动获取。
我自己尝试的时候发现没办法直接获取到，总是显示比赛的网址unreachable,所以我就手动下载了整个文件夹，然后如下图放在一个文件夹里面在本地的jupyter进行整个测试。
<img width="550" height="180" alt="image" src="https://github.com/user-attachments/assets/38537b55-e69a-4882-862f-239fabfe0fda" />
我也会把附件发出来供想采用我这种方式的小伙伴使用。
注意：采用这种方式要自己设置文件的路径哦，最简单的那种设置方式。



在NLP数据集中，文档通常以两种主要形式存在：

  * **较长的文档**: 每个文档一个文本文件，通常按类别分在不同的文件夹中。
  * **较短的文档**: 在一个CSV文件中，每一行代表一个文档（或文档对，以及可选的元数据）。


我们来看看我们的数据是什么样的。在Jupyter Notebook中，你可以在一行的开头使用 `!` 来执行任何bash/shell命令，并用 `{}` 来包含Python变量，就像这样：

```python
!ls {path}
```

```text
sample_submission.csv  test.csv  train.csv
```

看来这个比赛用的是CSV文件。要打开、操作和查看CSV文件，通常最好使用 **Pandas** 库。Pandas的首席开发者在这本书中对它有精彩的讲解（这本书也是matplotlib和numpy的绝佳入门读物，我在这个notebook中都用到了这两个库）。通常，我们会将Pandas导入并简写为 `pd`。

```python
import pandas as pd
```

我们来设置一下数据的路径：

```python
df = pd.read_csv(path/'train.csv')
```

这会创建一个 `DataFrame`，它是一个带有命名列的表格，有点像数据库的表。要查看DataFrame的首尾几行以及总行数，只需输入它的名字：

```python
df
```

```text
        id                  anchor                        target context  score
0      37d61fd2272659b1       abatement        abatement of pollution     A47   0.50
1      7b9652b17b68b7a4       abatement               act of abating     A47   0.75
2      36d72442aefd8232       abatement              active catalyst     A47   0.25
3      5296b0c19e1ce60e       abatement          eliminating process     A47   0.50
4      54c1e3b9184cb5b6       abatement                forest region     A47   0.00
...                 ...             ...                           ...     ...    ...
36468  8e1386cbefd7f245  wood article                wooden article     B44   1.00
36469  42d9e032d1cd3242  wood article                   wooden box     B44   0.50
36470  208654ccb9e14fa3  wood article                wooden handle     B44   0.50
36471  756ec035e694722b  wood article              wooden material     B44   0.75
36472  8d135da0b55b8c88  wood article             wooden substrate     B44   0.50

36473 rows × 5 columns
```

仔细阅读**数据集描述**对于理解每一列的用途非常重要。`DataFrame`最有用的功能之一是 `describe()` 方法：

```python
df.describe(include='object')
```

```text
                  id         anchor                target context
count          36473          36473                 36473   36473
unique         36473            733                 29340     106
top     37d61fd2272659b1  component composite coating  composition     H01
freq                 1            152                    24     186
```

我们可以看到，在36473行数据中，有733个唯一的`anchor`，106个`context`，以及近30000个`target`。有些`anchor`非常常见，例如 "component composite coating" 出现了152次。

早些时候，我建议我们可以把模型的输入表示成类似 "TEXT1: abatement; TEXT2: eliminating process" 的形式。我们还需要把`context`也加进去。在Pandas中，我们直接用 `+` 来拼接字符串：

```python
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
```

我们可以用标准的Python“点”符号来引用一个列（也称为`series`），或者像访问字典一样访问它。要获取前几行，使用 `head()` 方法：

```python
df.input.head()
```

```text
0    TEXT1: A47; TEXT2: abatement of pollution; ANC...
1    TEXT1: A47; TEXT2: act of abating; ANC1: abate...
2    TEXT1: A47; TEXT2: active catalyst; ANC1: abat...
3    TEXT1: A47; TEXT2: eliminating process; ANC1: ...
4    TEXT1: A47; TEXT2: forest region; ANC1: abatement
Name: input, dtype: object
```

### 文本分词 (Tokenization)

Transformers库使用一个 `Dataset` 对象来存储数据集。我们可以这样创建一个：

```python
from datasets import Dataset,DatasetDict
ds = Dataset.from_pandas(df)
```

在notebook里它会这样显示：

```text
Dataset({
    features: ['id', 'anchor', 'target', 'context', 'score', 'input'],
    num_rows: 36473
})
```

但是，我们不能直接把文本传给模型。一个深度学习模型期望的输入是数字，而不是英文句子！所以我们需要做两件事：

1.  **分词 (Tokenization)**: 把每段文本切分成单词（或者，我们后面会看到，切分成 *tokens*）。
2.  **数值化 (Numericalization)**: 把每个单词（或token）转换成一个数字。

具体如何操作取决于我们使用的特定模型。所以我们首先需要选一个模型。有成千上万的模型可供选择，但对于几乎所有的NLP问题，一个合理的起点是使用下面这个（在你完成探索后，可以把 "small" 换成 "large" 来获得一个更慢但更精确的模型）：

```python
model_nm = 'microsoft/deberta-v3-small'
```

`AutoTokenizer` 会为给定的模型创建一个合适的分词器：

```python
from transformers import AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)
```

下面是一个分词器如何将文本切分成“tokens”的例子（tokens类似单词，但可以是更小的“子词”片段）：

```python
tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!")
```

```text
[' G', "'", 'day', ' folks', ',', ' I', "'", 'm', ' Jeremy', ' from', ' fast', '.', 'ai', '!']
```

不常见的词会被切分成几部分。一个新词的开始由 `     ` 符号表示：

```python
tokz.tokenize("A platypus is an ornithorhynchus anatinus.")
```

```text
[' A', ' platypus', ' is', ' an', ' or', 'ni', 'tho', 'rhynch', 'us', ' an', 'at', 'inus', '.']
```

下面是一个简单的函数，用来对我们的输入进行分词：

```python
def tok_func(x): return tokz(x["input"])
```

为了在数据集的每一行上快速并行地运行这个函数，我们使用 `map`：

```python
tok_ds = ds.map(tok_func, batched=True)
```

这会在我们的数据集中增加一个名为 `input_ids` 的新项目。例如，这是我们数据第一行的输入文本和对应的ID：

```python
row = tok_ds[0]
row['input'], row['input_ids']
```

```text
('TEXT1: A47; TEXT2: abatement of pollution; ANC1: abatement',
 [1, 54453, 435, 294, 336, 5753, 346, 54453, 445, 294, 47284, 265, 6435, 346, 23702, 435, 294, 47284, 2])
```

那么，这些ID是什么，它们从哪里来？秘密在于分词器里有一个叫做 `vocab` 的列表，它为每个可能的token字符串包含了一个唯一的整数。我们可以这样查询它，比如查找单词 "of" 对应的token：

```python
tokz.vocab[' of']
```

```text
265
```

看看我们上面的 `input_ids`，我们确实看到了 `265` 如期出现。
最后，我们需要准备我们的标签。Transformers总是假设你的标签列名叫 `labels`，但在我们的数据集中，它目前是 `score`。因此，我们需要重命名它：

```python
tok_ds = tok_ds.rename_columns({'score':'labels'})
```

### 测试集和验证集

你可能已经注意到我们的目录里还有另一个文件：

```python
eval_df = pd.read_csv(path/'test.csv')
eval_df.describe()
```

这是**测试集**。机器学习中最重要的思想之一可能就是拥有独立的训练集、验证集和测试集。

#### 验证集

为了解释其动机，我们从一个简单的例子开始，想象我们正在拟合一个模型，其真实关系是下面这个二次函数：

```python
def f(x): return -3*x**2 + 2*x + 20
```

一个模型可能**欠拟合**（under-fit）或**过拟合**（over-fit）。

  * **欠拟合**：模型过于简单，无法捕捉数据的基本结构。就像试图用一条直线去拟合一条曲线。
  * **过拟合**：模型过于复杂，它不仅学习了数据的基本结构，还学习了数据中的噪声。这导致它在见过的数据上表现很好，但在未见过的新数据上表现很差。

那么，我们如何判断我们的模型是欠拟合、过拟合，还是“刚刚好”呢？我们使用**验证集**。这是一部分我们从训练中“保留”出来的数据——我们完全不让模型在训练时看到它。
验证集**只**被用来评估我们的模型表现如何，它**永远不会**被用作训练模型的输入。

Transformers使用 `DatasetDict` 来存放你的训练集和验证集。要创建一个包含25%数据作为验证集，75%作为训练集的数据集，使用 `train_test_split`：

```python
dds = tok_ds.train_test_split(0.25, seed=42)
dds
```

```text
DatasetDict({
    train: Dataset({
        features: ['id', 'anchor', 'target', 'context', 'labels', 'input', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 27354
    })
    test: Dataset({
        features: ['id', 'anchor', 'target', 'context', 'labels', 'input', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 9119
    })
})
```

如你所见，这里的验证集被命名为 `test` 而不是 `validate`，所以要小心！

#### 测试集

那么验证集解释完了，也创建好了。那“测试集”又是做什么的呢？
**测试集**是另一份从训练中保留出来的数据集。但它也从模型评估中被保留出来！你的模型在测试集上的准确率，只有在你完成了整个训练过程（包括尝试不同的模型、训练方法、数据处理等）之后，才会去检查一次。

因为当你在验证集上尝试各种方法时，你可能会偶然发现一些能提升验证集指标的东西，但这些提升在实际中并没有普适性。你实际上是在“过拟合”你的验证集！

这就是为什么我们保留一个测试集。Kaggle的公开排行榜就像一个你可以偶尔查看的测试集。但不要查得太频繁，否则你甚至会过拟合测试集！Kaggle还有第二个测试集，叫做“私有排行榜”，仅在比赛结束时用于最终排名。

我们用 `eval` 作为测试集的名称，以避免与上面创建的名为 `test` 的验证集混淆。

```python
eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)
```

### 评估指标与相关性

在Kaggle中，我们使用什么指标非常明确：Kaggle会告诉你！根据本次比赛的评估页面，“提交结果将根据预测的相似度分数与实际分数之间的**皮尔逊相关系数**进行评估。”

这个系数通常用 `r` 表示，取值范围在-1（完全负相关）到+1（完全正相关）之间。

Transformers期望指标以 `dict` 形式返回，这样训练器就知道该用什么标签。我们创建一个函数来完成这个任务：

```python
import numpy as np
def corr(x,y): return np.corrcoef(x,y)[0][1]
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}
```

### 训练模型

要在Transformers中训练模型，我们需要以下组件：

```python
from transformers import TrainingArguments,Trainer
```

我们选择一个适合我们GPU的批量大小（batch size），以及一个较小的训练轮数（epochs），这样可以快速进行实验：

```python
bs = 128
epochs = 4
```

最重要的超参数是学习率。你需要通过反复试验来找到一个最大但又不会导致训练失败的值。

```python
lr = 8e-5
```

Transformers使用 `TrainingArguments` 类来设置参数。

```python
args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')
```

现在我们可以创建我们的模型和 `Trainer`，这是一个将数据和模型结合在一起的类：

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  tokenizer=tokz, compute_metrics=corr_d)
```

现在，开始训练我们的模型！

```python
trainer.train();
```

```text
Epoch | Training Loss | Validation Loss | Pearson
--- | --- | --- | ---
1 | No log | 0.024492 | 0.800443
2 | No log | 0.022003 | 0.826113
3 | 0.041600 | 0.021423 | 0.834453
4 | 0.041600 | 0.022275 | 0.834767
```

关键要看上表中的 "Pearson" 值。如你所见，它在不断增加，并且已经超过了0.8。这是个好消息！我们现在可以获取在测试集上的预测结果了：

```python
preds = trainer.predict(eval_ds).predictions.astype(float)
```

注意 - 我们的一些预测值小于0或大于1！这再次显示了检查数据的重要性。我们来修正这些越界的预测：

```python
preds = np.clip(preds, 0, 1)
```

好了，现在我们准备好创建我们的提交文件了。

```python
import datasets
submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds.flatten() # 使用flatten()来确保维度正确
})

submission.to_csv('submission.csv', index=False)
```

### 尾声

感谢阅读！这对我是个小实验——我从未在Kaggle上写过“绝对初学者”的指南。希望你喜欢！如果你喜欢，请不吝点赞。有任何问题或想法，也请随时发表评论。

```
```









