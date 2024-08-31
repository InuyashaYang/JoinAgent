

# JoinAgent 🚀

<p align="center">
  <img src="./JoiningAI.png" alt="JoinAgent Logo" width="200"/>
</p>

<p align="center">
  <em>将LLM交互提升到前所未有的高度</em>
</p>

<p align="center">
  <a href="README_en.md">English</a> •
  <a href="README.md">中文</a> •
  <a href="README_fr.md">Français</a>
</p>


<p align="center">
  <a href="#主要特性">主要特性</a> •
  <a href="#安装">安装</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#文档">文档</a> •
  <a href="#贡献">贡献</a> •
  <a href="#许可证">许可证</a>
</p>

---

JoinAgent 是一个为高性能、并发LLM（大型语言模型）交互而设计的最先进框架。它为大规模AI任务提供了一个强大的解决方案，具有高级解析、验证和错误纠正功能。

## 主要特性

- 🚄 **高并发**：高效管理多个LLM调用
- 🧠 **智能解析**：无缝解释和结构化LLM输出
- 🛡️ **验证和错误纠正**：通过内置检查和纠正确保数据完整性
- 🔄 **检查点系统**：通过我们的高级检查点机制永不丢失进度
- 🔀 **灵活提示**：支持单个和多个提示模板
- ⏱️ **时间管理**：内置超时处理无响应任务

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from JoinAgent import MultiProcessor, MultiLLM, LLMParser, TextDivider

# 设置环境和文件路径
file_path='your_file_path'

# 初始化组件
llm = MultiLLM()
parser = LLMParser()
divider = TextDivider(threshold=4096, overlap=128)

# 定义模板和提示
data_template = '''
{"pos1":['数学对象1','数学对象2',...]}
'''

prompt_template = '''
你是一个工作细致的助手。我将给你一段数学类资料上的文本，请你帮我抽取出文本中所有的实体数学对象，并统一放入一个列表。
在工作期间，你将全程关闭搜索功能以及与外部的连接，仅凭文本本身内容来完成这项工作，不要擅自添加新的数学对象。
如果你提取出了一个数学对象，但你无法在文本中找到它的定义，请你不要输出这个数学对象。
请你除了输出这个列表外，不要在你的输出开头和结尾添加其他的东西。
提取的格式是：
{data_template}

特别注意：请你遇到文本中的示例，例题，习题等题目时直接跳过，不要解析其中的内容！！
数字、数学算式、代数式、字母等不含汉字的内容不被视作数学对象，请你删去！
请你不要输出类似于"函数f"，"矩阵B"这样并没有定义普适性、只是上下文中定义的指代性数学对象。

以下是我给你的文本：{pos1}，请你帮我提取出数学对象，并放入一个列表。
'''

correction_prompt = '''
你是一个严谨的校对员。我将给你一个由大模型生成的数据结构，请你根据规定格式内容进行校对和修正。

校对的格式是：
{data_template}

以下是待校验的文本：{answer}，请你帮我校对和修正这个列表。
'''

def validation(text):
    return True

# 创建 MultiProcessor 实例
processor = MultiProcessor(llm=llm, 
                           parse_method=parser.parse_dict, 
                           data_template=data_template, 
                           prompt_template=prompt_template, 
                           correction_template=correction_prompt, 
                           validator=validation)

# 处理文本
text_list = divider.divide(file_path)
text_dict = {index: {"pos1": value} for index, value in enumerate(text_list)}

# 执行多任务处理
results = processor.multitask_perform(text_dict, num_threads=5)

# 打印结果
for index, result in results.items():
    print(f"Chunk {index}: {result}")

```


## 许可证

本项目采用Apache License 2.0许可证 - 有关详细信息，请参阅[LICENSE](LICENSE)文件。

---

<p align="center">
  由 <a href="https://github.com/InuyashaYang">JoiningAI</a> 用❤️精心打造
</p>



