import threading
import time
import random
from queue import Queue
from tqdm import tqdm
import json
import os
import random

class MultiProcessor:
    def __init__(self, llm, parse_method, data_template, prompt_template, correction_template, validator, time_limit=120, back_up_llm=None, temperature=0.7, IsPromptList=False):
        self.llm = llm
        self.back_up_llm = back_up_llm
        self.parse_method = parse_method
        self.data_template = data_template
        self.prompt_template = prompt_template
        self.correction_template = correction_template
        self.validator = validator
        self.time_limit = time_limit
        self.checkpoint_dir = "checkpoint"
        self.temperature = temperature
        self.checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.json')
        self.IsPromptList = IsPromptList  # 新添加的属性

    def generate_prompt(self, **kwargs):
        kwargs['data_template'] = self.data_template
        if self.IsPromptList:
            # 如果 IsPromptList 为 True，随机选择一个 prompt 模板
            selected_template = random.choice(self.prompt_template)
            prompt = selected_template.format(**kwargs)
        else:
            # 如果 IsPromptList 为 False，使用单一的 prompt 模板
            prompt = self.prompt_template.format(**kwargs)
        return prompt

    def generate_correction_prompt(self, answer):
        return self.correction_template.format(answer=answer, data_template=self.data_template)

    def task_perform(self, llm, **kwargs):
        try:
            prompt = self.generate_prompt(**kwargs)
            answer = llm.ask(prompt,self.temperature)
            structured_data = self.parse_method(answer)
            return structured_data
        except Exception as e:
            print(f"Error in task_perform: {str(e)}")
            raise e

    def correct_data(self, llm, answer):
        correction_prompt = self.generate_correction_prompt(answer)
        correction = llm.ask(correction_prompt)
        return self.parse_method(correction)

    def process_task(self, index, key_dict, Active_Transform):
        try:
            attempts = 0
            base_wait_time = 1
            use_backup = False

            while attempts < 2:
                try:
                    current_llm = self.back_up_llm if (use_backup and self.back_up_llm is not None) else self.llm
                    structured_data = self.task_perform(current_llm, **key_dict)
                    if self.validator(structured_data):
                        return self.map_answer_to_pos(structured_data) if Active_Transform else structured_data
                    corrected_answer = self.correct_data(current_llm, structured_data)
                    if corrected_answer and self.validator(corrected_answer):
                        return self.map_answer_to_pos(corrected_answer) if Active_Transform else corrected_answer
                    break
                except Exception as e:
                    if 'Throttling.RateQuota' in str(e):
                        wait_time = base_wait_time * (2 ** attempts) + random.uniform(0, 1)
                        print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds. Attempt {attempts + 1}/2")
                        time.sleep(wait_time)
                    else:
                        print(f"An error occurred: {str(e)}. Attempt {attempts + 1}/2")
                    
                    attempts += 1
                    if not use_backup and self.back_up_llm is not None:
                        use_backup = True
                        print(f"Switching to backup LLM to process task {index}")

            return None  # 如果所有尝试都失败，返回 None

        except Exception as final_error:
            print(f"Error occurred during process_task for index {index}: {str(final_error)}")
            return None  # 返回 None 表示跳过这个任务

    def map_answer_to_pos(self, answer_dict):
        pos_dict = {}
        for i, (key, value) in enumerate(answer_dict.items(), start=1):
            pos_key = f"pos{i}"
            pos_dict[pos_key] = value
        return pos_dict

    def save_checkpoint(self, results):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # 加载现有的检查点
        existing_results = self.load_checkpoint()
        
        # 更新现有结果，只添加或更新新的结果
        for k, v in results.items():
            if v is not None:  # 只更新非空结果
                existing_results[str(k)] = v

        # 保存更新后的结果
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=4)
        print(f"Checkpoint updated and saved at {self.checkpoint_path}.")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def multitask_perform(self, index_dict, num_threads, checkpoint=10, Active_Reload=False, Active_Transform=False):
        if Active_Reload:
            previous_results = self.load_checkpoint()
            
            # 确保所有的键都是整数类型
            results = {int(k): v for k, v in previous_results.items()}
            index_dict = {int(k): v for k, v in index_dict.items()}
            
            # 计算已完成的任务
            completed_tasks = {k: v for k, v in results.items() if v is not None}
            
            # 计算剩余的任务
            remaining_tasks = {k: v for k, v in index_dict.items() if k not in completed_tasks}
            
            print(f"原始数据数量: {len(index_dict)}")
            print(f"已完成任务数量: {len(completed_tasks)}")
            print(f"剩余待处理数据数量: {len(remaining_tasks)}")
            
            # 用剩余的任务更新 index_dict
        else:
            results = {}
            remaining_tasks = index_dict

        queue = Queue()
        checkpoint_counter = 0

        for index, key_dict in remaining_tasks.items():
            queue.put((index, key_dict))

        def worker(pbar):
            nonlocal checkpoint_counter
            while not queue.empty():
                index, key_dict = queue.get()
                if index in results and results[index] is not None:
                    queue.task_done()
                    pbar.update(1)
                    continue

                result = None
                thread_result_queue = Queue()

                thread = threading.Thread(target=lambda q, arg1, arg2, arg3: q.put(self.process_task(arg1, arg2, arg3)), 
                                          args=(thread_result_queue, index, key_dict, Active_Transform))
                thread.start()
                thread.join(timeout=self.time_limit)

                if thread.is_alive():
                    print(f"Thread processing task {index} timed out.")
                    thread.join()
                else:
                    if not thread_result_queue.empty():
                        result = thread_result_queue.get()
                    else:
                        print(f"No result obtained for task {index}")

                if result is not None:
                    results[index] = result

                queue.task_done()
                pbar.update(1)

                checkpoint_counter += 1
                if checkpoint_counter % checkpoint == 0:
                    print(f"Saving checkpoint at counter {checkpoint_counter}")
                    self.save_checkpoint(results)

        with tqdm(total=len(remaining_tasks)) as pbar:
            threads = []
            for _ in range(min(num_threads, len(remaining_tasks))):
                thread = threading.Thread(target=worker, args=(pbar,))
                threads.append(thread)
                thread.start()

            queue.join()

            for thread in threads:
                thread.join()

        print("Final save_checkpoint call")
        self.save_checkpoint(results)

        completed_tasks = sum(1 for r in results.values() if r is not None)
        print(f"Final results - Total tasks: {len(results)}, Completed tasks: {completed_tasks}")

        return results
