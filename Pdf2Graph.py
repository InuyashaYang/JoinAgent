import os
import re
from JoinAgent import *
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time
import shutil
import traceback
# 初始化API

# 采用Divider类方法切割文本
divider=TextDivider(threshold=4096,overlap=128)

parser=LLMParser()

def pdf_to_png(pdf_path, output_folder, dpi=300):
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # 清除目标文件夹中的所有内容
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    # 使用多线程处理PDF页面
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for page_num in range(len(doc)):
            futures.append(executor.submit(process_page, doc, page_num, output_folder, dpi))
        
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDF"):
            future.result()

def process_page(doc, page_num, output_folder, dpi):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    
    # 将图像保存为PNG文件
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
    img.save(img_path)
    
    # 二值化处理
    img = img.convert("L")
    img = img.point(lambda p: 255 if p > 72 else 0)
    img.save(img_path)

def split_image(img_path, output_folder, num_parts):
    # 打开图像
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    # 计算每个部分的宽度
    part_height = img_height // num_parts
    
    # 提取页面编号
    page_num = os.path.basename(img_path).split('_')[1].split('.')[0]
    
    for i in range(num_parts):
        # 计算切割区域的坐标
        top = i * part_height
        bottom = (i + 1) * part_height
        
        # 切割图像
        part_img = img.crop((0, top, img_width, bottom))
        
        # 保存切割后的图像
        part_img_path = os.path.join(output_folder, f"page_{page_num}_part_{i + 1}.png")
        part_img.save(part_img_path)

def process_pdf(pdf_path, output_folder, dpi=300, num_parts=3):
    # 将PDF转换为PNG图像
    pdf_to_png(pdf_path, output_folder, dpi)
    
    # 遍历所有生成的PNG图像
    for img_file in os.listdir(output_folder):
        if img_file.endswith(".png"):
            img_path = os.path.join(output_folder, img_file)
            split_image(img_path, output_folder, num_parts)
            os.remove(img_path)  # 删除原始的未切割图像

def process_png_to_markdown(pdf_folder, markdown_folder, llm, parser, completion_rate_threshold=0.99, max_attempts=5, max_workers=500):
    if not os.path.exists(markdown_folder):
        os.makedirs(markdown_folder)

    def check_completed_files():
        completed_files = set()
        for filename in os.listdir(markdown_folder):
            if filename.endswith('.md'):
                completed_files.add(filename.replace('.md', '.png'))
        return completed_files

    def process_file(filename, completed_files):
        input_path = os.path.join(pdf_folder, filename)
        output_path = os.path.join(markdown_folder, filename.replace('.png', '.md'))

        if filename in completed_files:
            return True

        prompt = '请使用markdown格式为我忠实地整理这一页上的语言论述和数学公式，我要求你尽量准确无误地整理数学公式，并使用$符号包裹，输出遵循如下格式=start_pad=...main_md_content...=end_pad=,若你无法识别输入图，在main_md_content部分留空'
        correct_prompt = '以下是大语言模型所生成的回复，我们期待它遵循如下格式=start_pad=...main_md_content...=end_pad=，请你协助我将其纠正'

        try:
            # 第一次尝试
            output = llm.look(input_path, prompt)
            parsed_output = parser.parse_pads(output)

            # 检查是否需要矫正
            if not parsed_output.strip() or "=start_pad=" not in output or "=end_pad=" not in output:
                print(f"文件 {filename} 的输出可能需要矫正，尝试使用矫正提示词。")
                # 使用矫正提示词进行第二次尝试
                corrected_output = llm.look(input_path, correct_prompt + "\n" + output)
                parsed_output = parser.parse_pads(corrected_output)

            # 无论内容是否为空，都写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(parsed_output)

            # 如果最终输出仍然不符合预期格式，则认为处理失败
            if "=start_pad=" not in parsed_output or "=end_pad=" not in parsed_output:
                print(f"文件 {filename} 处理后的输出不符合预期格式。")
                return False
            return True

        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {str(e)}")
            traceback.print_exc()
            return False

    def multi_thread_process(remaining_files):
        completed_files = check_completed_files()
        successful_files = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, filename, completed_files): filename for filename in remaining_files}
            
            with tqdm(total=len(remaining_files), desc="处理进度") as progress_bar:
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        if future.result():
                            successful_files += 1
                    except Exception as exc:
                        print(f'{filename} 生成异常: {exc}')
                        traceback.print_exc()
                    finally:
                        progress_bar.update(1)

        return check_completed_files(), successful_files

    try:
        all_files = [f for f in os.listdir(pdf_folder) if f.endswith('.png')]
        total_files = len(all_files)
        
        for attempt in range(max_attempts):
            print(f"开始第 {attempt + 1} 轮处理")
            completed_files, successful_files = multi_thread_process(all_files)
            completion_rate = len(completed_files) / total_files

            print(f"当前完成率: {completion_rate:.2%}")

            if completion_rate >= completion_rate_threshold:
                print(f"成功完成解析，{completion_rate:.2%} 的文件已被成功处理。")
                return

            all_files = [f for f in all_files if f not in completed_files]

        print(f"警告：在 {max_attempts} 次尝试后，成功率仍低于 {completion_rate_threshold:.2%}，请检查未成功解析的文件。")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        traceback.print_exc()

def sort_key(filename):
    # 从文件名中提取页码(i)和部分(j)
    match = re.match(r'page_(\d+)_part_(\d+)\.md', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (float('inf'), float('inf'))  # 对于不匹配的文件名,放到最后

def merge_files(directory):
    # 获取目录中所有的.md文件
    files = [f for f in os.listdir(directory) if f.endswith('.md')]
    
    # 按照排序规则对文件进行排序
    sorted_files = sorted(files, key=sort_key)
    
    # 合并文件内容
    merged_content = []
    for filename in sorted_files:
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            merged_content.append(content)
            
            # 如果文件内容不以换行符结束,添加一个换行符
            if not content.endswith('\n'):
                merged_content.append('\n')
    
    # 写入合并后的内容到merged.md
    with open(os.path.join(directory, 'merged.md'), 'w', encoding='utf-8') as outfile:
        outfile.write(''.join(merged_content))

def graph_process(file_path, llm, parser, num_threads=500, checkpoint=100):
    global entity_pairs_explanation, extracted_entity_result
    def contains_chinese(text):
        return re.search(r'[\u4e00-\u9fff]', text) is not None

    def filter_dict_of_lists(data):
        filtered_data = {}
        for key, value in data.items():
            filtered_pos1 = [item for item in value['pos1'] if contains_chinese(item)]
            filtered_data[key] = {'pos1': filtered_pos1}
        return filtered_data

    def process_dict_of_lists(data):
        flat_list = [item for sublist in data.values() for item in sublist['pos1']]
        embeddings_dict = llm.embed_list(flat_list)
        similarity_dict = llm.partition_by_similarity(embeddings_dict, threshold=0.9)

        name_mapping = {}
        for key in flat_list:
            if key in similarity_dict:
                if similarity_dict[key]['Similar_keys']:
                    name_mapping[key] = similarity_dict[key]['Similar_keys'][0]
                else:
                    name_mapping[key] = key
            else:
                name_mapping[key] = key

        new_data = {}
        for key, value in data.items():
            new_pos1 = [name_mapping[item] for item in value['pos1']]
            new_data[key] = {'pos1': list(set(new_pos1))}  # 使用 set 去重

        return new_data

    def create_entity_pairs(processed_data):
        entity_pairs = {}
        new_index = 1

        for index, data in processed_data.items():
            entities = data['pos1']
            n = len(entities)
            
            for i in range(n):
                for j in range(i+1, n):
                    entity1 = entities[i]
                    entity2 = entities[j]
                    
                    # 确保实体对的顺序一致
                    if entity1 > entity2:
                        entity1, entity2 = entity2, entity1

                    pair = (entity1, entity2)
                    
                    if pair in entity_pairs:
                        # 如果实体对已存在，更新共同出现的索引列表
                        entity_pairs[pair]['pos3'].append(index)
                    else:
                        # 如果是新的实体对，创建新条目
                        entity_pairs[pair] = {
                            'pos1': entity1,
                            'pos2': entity2,
                            'pos3': [index]
                        }

        # 将结果转换为所需的格式
        result = {}
        for pair, data in entity_pairs.items():
            result[new_index] = data
            new_index += 1

        return result

    def extract_entities(processed_data):
        entity_dict = {}
        new_index = 1

        for original_index, data in processed_data.items():
            entities = data['pos1']
            
            for entity in entities:
                if entity in entity_dict:
                    # 如果实体已存在，更新它出现的索引列表
                    entity_dict[entity]['pos2'].append(original_index)
                else:
                    # 如果是新的实体，创建新条目
                    entity_dict[entity] = {
                        'pos1': entity,
                        'pos2': [original_index]
                    }

        # 将结果转换为所需的格式
        result = {}
        for entity, data in entity_dict.items():
            result[new_index] = data
            new_index += 1

        return result

    def replace_indices_with_text(data_dict, text_list):
        result = {}
        
        for new_index, item in data_dict.items():
            # 复制原始项
            new_item = item.copy()
            
            # 动态检查哪个键对应着列表
            index_key = None
            for key, value in item.items():
                if isinstance(value, list) and all(isinstance(i, int) for i in value):
                    index_key = key
                    break
            
            # 如果没有找到对应列表的键，跳过这一项
            if index_key is None:
                result[new_index] = new_item
                continue
            
            # 获取索引列表
            indices = item[index_key]
            
            # 获取对应的文本段落并聚合
            texts = [text_list[i-1] for i in indices if 1 <= i <= len(text_list)]  # 确保索引在有效范围内
            aggregated_text = ' '.join(texts)
            
            # 用聚合的文本替换索引列表
            new_item[index_key] = aggregated_text
            
            # 将新项添加到结果字典
            result[new_index] = new_item
        
        return result

    def merge_results(results):
        merged_dict = {}
        for item in results:
            key = item['算法对象']  # 获取数学对象名称
            if key in merged_dict:
                for k, v in item.items():
                    if k == '性质':
                        if k in merged_dict[key]:
                            merged_dict[key][k] = list(set(merged_dict[key][k] + v))  # 性质存储在列表中，合并并去重
                        else:
                            merged_dict[key][k] = v
                    elif k != '算法对象':
                        if k in merged_dict[key]:
                            merged_dict[key][k] += f";{v}"  # 不是性质的项，用分号连接
                        else:
                            merged_dict[key][k] = v
            else:
                merged_dict[key] = item
        return list(merged_dict.values())  # 返回合并后的字典值列表

    def filter_and_print_report(data_list, strength_key='关系强度', threshold=7.5):
        def is_valid_strength(strength):
            try:
                return float(strength) >= threshold
            except ValueError:
                return False

        original_count = len(data_list)
        filtered_list = [item for item in data_list if is_valid_strength(item[strength_key])]
        filtered_count = len(filtered_list)

        removed_count = original_count - filtered_count
        removed_percentage = (removed_count / original_count) * 100 if original_count > 0 else 0

        # 打印报告
        print(f"原始条目数: {original_count}")
        print(f"筛选后条目数: {filtered_count}")
        print(f"被筛掉的条目数: {removed_count}")
        print(f"被筛掉的百分比: {removed_percentage:.2f}%")

        return filtered_list

    def shift_properties(f1, f2):
        # 从f1中提取唯一的性质
        unique_list_of_properties = []
        for item in f1:
            if '性质' in item and item['性质']:
                for prop in item['性质']:
                    if prop not in unique_list_of_properties:
                        unique_list_of_properties.append(prop)

        # 将性质也作为节点，添加到f1
        for prop in unique_list_of_properties:
            f1.append({'算法对象': prop, '定义': '一种性质'})

        # 向f2中添加对象及对应属性间的关系（此处设置为*）
        for item in f1:
            if '性质' in item and item['性质']:
                for prop in item['性质']:
                    f2.append({
                        '出发节点': item['数学对象'],
                        '到达节点': prop,
                        '关系名称': '*',
                        '解释': f'{prop}是{item["算法对象"]}的性质',
                        '关系强度': '10'
                    })

        # 从f1中移除'性质'键值对
        for item in f1:
            if '性质' in item:
                del item['性质']

        return f1, f2

    data_template01='''
    {"pos1":['算法1','数据结构1',...]}
    '''

    prompt_template01 ='''
    你是一个工作细致的助手。我将给你一段算法教材上的文本，请你帮我抽取出文本中所有的算法和数据结构，并统一放入一个列表。
    在工作期间，你将全程关闭搜索功能以及与外部的连接，仅凭文本本身内容来完成这项工作，不要擅自添加新的算法或数据结构。
    如果你提取出了一个算法或数据结构，但你无法在文本中找到它的定义或描述，请你不要输出它。
    请你除了输出这个列表外，不要在你的输出开头和结尾添加其他的东西。
    提取的格式是：
    {data_template}

    特别注意：请你遇到文本中的示例，例题，习题等题目时直接跳过，不要解析其中的内容！！
    变量名、函数名、代码片段等不含算法或数据结构名称的内容不被视作有效内容，请你删去！
    请你不要输出类似于"函数f"，"数组arr"这样并没有定义普适性、只是上下文中定义的指代性对象。

    示例：
    Input:本章介绍了几种基本的排序算法，包括冒泡排序、选择排序和插入排序。这些算法都属于比较排序，其时间复杂度为O(n^2)。接下来我们将讨论更高效的排序算法，如快速排序和归并排序，它们的平均时间复杂度为O(nlogn)。

    Output:{{"pos1":['冒泡排序','选择排序','插入排序','快速排序','归并排序']}}
    以上是示例，请你不要在正文中输出
    ############################################
    以下是我给你的文本：{pos1}，请你帮我提取出算法和数据结构，并放入一个列表。
    '''

    def validation(text):
        return True

    correction_prompt= '''
        你是一个严谨的校对员。我将给你一个由大模型生成的数据结构，请你根据规定格式内容进行校对和修正。

        校对的格式是：
        {data_template}

        以下是待校验的文本：{answer}，请你帮我校对和修正这个列表。
        '''

    data_template02='''
        {
        "算法对象":"算法名称",
        "定义":"...",
        "时间复杂度":"...",
        "空间复杂度":"...",
        "优点":["...",...],
        "缺点":["...",...],
        "适用场景":"...",
        "实现步骤":["...",...],
        ...
            }
    '''

    prompt_template02 ='''
        这里有一段文本:{pos2}.
        我想要研究其中算法{pos1}的定义、复杂度、优缺点、应用等属性，你来负责帮我从文本中帮我抽取相关内容，并按给定格式输出。
        这里有很多算法概念，但是我只要{pos1}这一个算法的解释
        提取的格式是一个单独的字典：
        {data_template}
        

        示例：
            {{
                "算法对象": "快速排序",
                "定义": "一种分治策略的排序算法",
                "时间复杂度": "平均情况O(nlogn)，最坏情况O(n^2)",
                "空间复杂度": "O(logn)",
                "优点": ["平均情况下效率高", "原地排序"],
                "缺点": ["不稳定", "最坏情况下效率低"],
                "适用场景": "大规模数据排序",
                "实现步骤": ["选择基准元素", "划分数组", "递归排序子数组"]
            }}

        以上是示例，请你不要在正文中输出
        ############################################
        字典中，第一个键是"算法对象"，值必须是{pos1}，即列表中字典的个数和所给算法对象的个数应相同。
        注意：字典中，键"优点"和"缺点"所对应的值都要是一个列表。如果无法概括为列表形式，则不输出这个键值对。
        字典中只有算法对象是必要的，如定义、时间复杂度、空间复杂度等键，如果在文本中找不到对应的值，则直接删除这个键。
        每个键的值不要超过50字。

        如果你觉得有其他键可以描述算法对象的属性，也可以添加，但注意：
        1.不要有"相关算法"，"变体"这种包含算法对象间关系的键出现；
        2.不要有跟已规定的键相似度较高的键，如"特点"键和规定的"优点"或"缺点"键内容相似度高，则不要输出这样的键；
        3.如果添加新键，要保证该键名称书面化，规范，简洁，且与其他键的值没有较高关联性。

        ""中不能出现字母，但如果字母未在该行第一个""中出现，则保留。
        不要解析文本中所有的示例。
        如果""中含顿号，则拆开成两个""分别储存内容。
        ""中如果为算法对象或一句完整的句子，则不做修改；如果是一个状语或不完整的句子，则删减为算法对象，但前提是保证修改内容最少。
        请你除了输出算法对象和它们的属性之外，不要在你的回答最前面或最后面输入解释的语句。
        '''

    data_template03='''
        {
        "出发节点":"算法1",
        "到达节点":"算法2",
        "关系名称":"具体名称",
        "关系解释"："根据具体文本内容确定",
        "关系强度":用1-10间的整数进行评分
        }
    '''

    prompt_template03left ='''
        这里有一段文本:{pos3}
        你是一个工作细致的助手，负责帮我解释{pos1}和{pos2}之间的关系。
        '关系名称'的值的格式总应该是'...关系'，包括但不限于改进关系，衍生关系，并列关系...如果无法概括成这种形式，也尽量用书面化语言表述。
        特别注意的是，关系是有向的，根据出发节点和到达节点的不同而不同。
        提取的格式是：{data_template}

        
        示例：
        {{
        "出发节点":"冒泡排序",
        "到达节点":"快速排序",
        "关系名称":"改进关系",
        "关系解释":"快速排序是对冒泡排序的一种改进，通过分治策略提高了排序效率",
        "关系强度":"8"
        }}

        特别注意：出发节点是{pos1}，到达节点是{pos2}
        ""中不能出现字母，但如果字母未在该行第一个""中出现，则保留。
        不要解析文本中所有的示例。
        '''

    prompt_template03right ='''
        这里有一段文本:{pos3}
        你是一个工作细致的助手，负责帮我解释{pos2}和{pos1}之间的关系。
        '关系名称'的值的格式总应该是'...关系'，包括但不限于改进关系，衍生关系，并列关系...如果无法概括成这种形式，也尽量用书面化语言表述。
        特别注意的是，关系是有向的，根据出发节点和到达节点的不同而不同。
        提取的格式是：{data_template}

        
        示例：
        {{
        "出发节点":"快速排序",
        "到达节点":"冒泡排序",
        "关系名称":"优化关系",
        "关系解释":"快速排序是对冒泡排序的一种优化，通过分治策略提高了排序效率",
        "关系强度":"8"
        }}

        特别注意：出发节点是{pos2}，到达节点是{pos1}
        ""中不能出现字母，但如果字母未在该行第一个""中出现，则保留。
        不要解析文本中所有的示例。
        '''

    # 初始化处理器
    entity_extractor = MultiProcessor(llm=llm, parse_method=parser.parse_dict, data_template=data_template01, 
                                      prompt_template=prompt_template01, correction_template=correction_prompt, 
                                      validator=validation, back_up_llm=None)
    entity_explainer = MultiProcessor(llm=llm, parse_method=parser.parse_dict, data_template=data_template02, 
                                      prompt_template=prompt_template02, correction_template=correction_prompt, 
                                      validator=validation, back_up_llm=None)
    left_relation_extractor = MultiProcessor(llm=llm, parse_method=parser.parse_dict, data_template=data_template03, 
                                             prompt_template=prompt_template03left, correction_template=correction_prompt, 
                                             validator=validation)
    right_relation_extractor = MultiProcessor(llm=llm, parse_method=parser.parse_dict, data_template=data_template03, 
                                              prompt_template=prompt_template03right, correction_template=correction_prompt, 
                                              validator=validation)

    # 处理文本
    text_list = divider.divide(file_path)
    text_dict = {index: {"pos1": value} for index, value in enumerate(text_list)}
    
    # 提取实体
    entity_dict = entity_extractor.multitask_manage(text_dict, num_threads=num_threads, checkpoint=checkpoint, 
                                                     initial_reload=False, Active_Transform=False,max_diff_ratio=0.02,max_retries=4)
    filtered_data = filter_dict_of_lists(entity_dict)
    processed_data = process_dict_of_lists(filtered_data)
    entity_pairs = create_entity_pairs(processed_data)
    extracted_entities = extract_entities(processed_data)
    
    # 处理提取的实体
    processed_extracted_entities = replace_indices_with_text(extracted_entities, text_list)
    processed_entity_pairs = replace_indices_with_text(entity_pairs, text_list)
    
    # 解释实体
    extracted_entity_result = entity_explainer.multitask_manage(processed_extracted_entities, num_threads=num_threads, initial_reload=False,
                                                                 checkpoint=checkpoint, Active_Transform=False,max_diff_ratio=0.02,max_retries=4)
    extracted_entity_result = list(extracted_entity_result.values())
    merged_result = merge_results(extracted_entity_result)
    
    # 提取关系
    left_entity_pairs_explanation = left_relation_extractor.multitask_manage(processed_entity_pairs, num_threads=num_threads,  initial_reload=False,
                                                                              checkpoint=checkpoint, Active_Transform=False,max_diff_ratio=0.02,max_retries=4)
    right_entity_pairs_explanation = right_relation_extractor.multitask_manage(processed_entity_pairs, num_threads=num_threads, initial_reload=False,
                                                                                checkpoint=checkpoint, Active_Transform=False,max_diff_ratio=0.02,max_retries=4)
    left_entity_pairs_explanation = list(left_entity_pairs_explanation.values())
    right_entity_pairs_explanation = list(right_entity_pairs_explanation.values())
    entity_pairs_explanation = left_entity_pairs_explanation + right_entity_pairs_explanation
    
    # 过滤和报告
    filtered_entity_pairs = filter_and_print_report(entity_pairs_explanation)
    entity_pairs_explanation = filtered_entity_pairs
    
    # 移动属性
    extracted_entity_result, entity_pairs_explanation = shift_properties(extracted_entity_result, entity_pairs_explanation)

    return extracted_entity_result,entity_pairs_explanation

def pdf2graph(pdf_path,model_type='gpt-4o-2024-08-06'):
    llm=MultiLLM(model_type)
    pdf_folder = r'pdf_folder'
    markdown_folder=r'markdown_folder'
    process_pdf(pdf_path, pdf_folder, dpi=300, num_parts=3)
    process_png_to_markdown(pdf_folder, markdown_folder, llm, parser)
    merge_files(markdown_folder)
    file_path=os.path.join(markdown_folder,'merged.md')
    return graph_process(file_path, llm, parser)
    
