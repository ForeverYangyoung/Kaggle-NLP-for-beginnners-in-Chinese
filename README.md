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









