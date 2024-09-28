

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


# 模型池使用手册

## 1. 如何调用DeepseekLLM类或MultiLLM类进行对话(非流式)

这部分没有变化，仍然可以直接使用各自的 `ask` 方法：

```python
# DeepseekLLM
deepseek = DeepseekLLM(version='coder', api_key='your_api_key')
response = deepseek.ask("你的问题")

# MultiLLM
multi_llm = MultiLLM(model='deepseek-coder', api_key='your_api_key')
response = multi_llm.ask("你的问题")
```

## 2. 如何调用MultiLLM类进行视觉识别、嵌入

这部分也没有变化：

```python
# 视觉识别
multi_llm = MultiLLM(vision_model='gpt-4o-mini', api_key='your_api_key')
response = multi_llm.look("image_path.jpg", "描述这张图片")

# 嵌入
multi_llm = MultiLLM(embed_model='text-embedding-3-large', api_key='your_api_key')
embedding = multi_llm.embed_text("要嵌入的文本")
```

## 3. 如何初始化一个模型池，并用模型池进行对话、视觉识别、嵌入

```python
deepseek = DeepseekLLM(api_key='deepseek_api_key')
multi_llm = MultiLLM(api_key='multi_llm_api_key')

model_pool = ModelPool([deepseek, multi_llm])

# 对话
response = model_pool.ask("你的问题")

# 视觉识别
response = model_pool.look("image_path.jpg", "描述这张图片")

# 嵌入
embedding = model_pool.embed_text("要嵌入的文本")
```

## 4. 如何向模型池中增加或者删除模型

```python
# 增加模型
new_model = MultiLLM(api_key='new_api_key')
model_pool.add_model(new_model)

# 删除模型
model_pool.remove_model(existing_model)
```

## 5. 如何提高某个池中模型的访问量权重

修改后的 `set_weight` 方法现在使用权重调整因子（倍率）：

```python
model_pool.set_weight(existing_model, weight_factor)
```

例如，要将某个模型的权重提高为原来的两倍：

```python
model_pool.set_weight(existing_model, 2.0)
```

注意：调整后，所有模型的权重会自动重新归一化。

## 6. 如何同时修改所有模型的权重

使用 `update_weights` 方法可以一次性更新所有模型的权重：

```python
new_weights = [1.0, 2.0, 0.5]  # 假设有3个模型
model_pool.update_weights(new_weights)
```

如果提供的权重数量少于模型数量，会自动使用平均值补齐；如果多于模型数量，会截断多余的权重。更新后，权重会自动归一化。

## 7. 如何查看当前池中模型

使用 `get_models` 方法可以获取当前池中的所有模型及其权重：

```python
models_and_weights = model_pool.get_models()
for model, weight in models_and_weights:
    print(f"Model: {type(model).__name__}, Weight: {weight}")
```

这将返回一个包含模型实例和对应权重的列表。

另外，你还可以使用以下方法获取更多信息：

- `get_pool_size()`: 获取模型池的大小
- `get_available_operations()`: 获取池中所有可用的操作


# 1.如何设计并启动一个多线程操作

## 1. 设计提示词和数据模板

首先，设计您的数据模板和提示词模板：

## 提示词模板示例

```python
prompt_template = '''
给定输入：
{pos1}

请执行以下操作并输出三个结果：
1. [操作说明1]
2. [操作说明2]
3. [操作说明3]

同时简要描述您的操作行为。

请按照以下格式输出：
{data_template}
'''
```

## 输入数据示例

```python
index_dict = {
    "1": {"pos1": "输入数据1"},
    "2": {"pos1": "输入数据2"},
    "3": {"pos1": "输入数据3"},
    # ... 更多数据
}
```

## 关系解释

1. **变量匹配**：
   - 提示词模板中的 `{pos1}` 对应输入数据字典中的 `"pos1"` 键。
   - 当处理 "1" 号任务时，`{pos1}` 将被替换为 "输入数据1"。

2. **动态替换**：
   - 对于每个任务，系统会用相应的输入数据替换模板中的变量。
   - 例如，处理 "2" 号任务时，生成的实际提示词会包含 "输入数据2"。

3. **多个变量**：
   - 如果您的任务需要多个输入，可以在模板中使用多个变量，如 `{pos1}`, `{pos2}` 等。
   - 相应地，输入数据字典中也应该有匹配的键：`{"pos1": "...", "pos2": "..."}`。

4. **数据模板**：
   - `{data_template}` 是一个特殊变量，它会被替换为您定义的 `data_template` 字符串。
   - 这告诉模型如何构造其输出。

```python

correction_template = '''
您是一位细心的校对者。我将给您一个由大型语言模型生成的数据结构。请根据指定的格式和内容对其进行校对和纠正。

校对的格式为：
{data_template}

这是需要验证的文本：{answer}。请帮我校对并纠正这个列表。
'''

def validator(data):
    # 实现您的验证逻辑
    return True
```

## 2. 准备输入数据

创建一个包含所有任务的字典：

```python
index_dict = {
    "1": {"pos1": "输入数据1"},
    "2": {"pos1": "输入数据2"},
    "3": {"pos1": "输入数据3"},
    # ... 更多数据
}
```

## 3. 初始化 MultiProcessor

```python
from your_llm_module import YourLLM  # 导入您的LLM类

llm = YourLLM(api_key="your_api_key")
parse_method = parse_dict  # 或其他解析方法：parse_list, parse_code, parse_pads

processor = MultiProcessor(
    llm=llm,
    parse_method=parse_method,
    data_template=data_template,
    prompt_template=prompt_template,
    correction_template=correction_template,
    validator=validator,
    time_limit=120,  # 单个任务的时间限制（秒）
    temperature=0.7
)
```

## 4. 启动多线程操作

使用 `multitask_perform` 方法启动多线程操作：

```python
results = processor.multitask_perform(
    index_dict=index_dict,
    num_threads=10,  # 设置线程数
    checkpoint=20,   # 每处理20个任务保存一次检查点
    Active_Reload=False,  # 是否从检查点重新加载
    Active_Transform=False,  # 是否转换结果格式
    checkpoint_dir="my_checkpoints"  # 检查点保存目录
)
```

## 5. 处理结果

处理返回的结果：

```python
for index, result in results.items():
    if result:
        print(f"Task {index} completed:")
        print(result)
    else:
        print(f"Task {index} failed.")
```

# 2. 如何管理任务完成率、重试次数和最大处理时间

要管理任务完成率、重试次数和最大处理时间，您可以使用 `multitask_manage` 方法：

```python
results = processor.multitask_manage(
    index_dict,
    num_threads=10,
    checkpoint=20,
    Active_Reload=False,
    Active_Transform=False,
    checkpoint_dir="my_checkpoints",
    threshold=0.95,  # 设置任务完成率阈值为95%
    max_multitask_retries=3,  # 最大重试次数为3次
    max_time=3600  # 最大处理时间为1小时（3600秒）
)
```

这个方法会：

1. 重复执行 `multitask_perform` 直到满足以下条件之一：
   - 达到或超过设定的完成率阈值（threshold）
   - 达到最大重试次数（max_multitask_retries）
   - 达到最大处理时间（max_time）

2. 在每次重试时，它会：
   - 从检查点重新加载已完成的任务（Active_Reload=True）
   - 只处理未完成的任务
   - 更新完成率并检查是否达到阈值

3. 输出处理进度和完成率信息


## 许可证

本项目采用Apache License 2.0许可证 - 有关详细信息，请参阅[LICENSE](LICENSE)文件。

---

<p align="center">
  由 <a href="https://github.com/InuyashaYang">JoiningAI</a> 用❤️精心打造
</p>



