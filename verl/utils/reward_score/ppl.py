# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from generator_llms.local import compute_retrieval_utility_score



def extract_titles(solution_str):
    # Find all sections between <information> and </information>
    info_sections = re.findall(r'<information>(.*?)</information>', solution_str, re.DOTALL)

    titles = []
    for section in info_sections:
        # In each section, find all titles using the pattern
        found_titles = re.findall(r'Doc\s+\d+\(Title:\s*(.*?)\)', section)
        titles.extend(found_titles)
    
    return titles

def extract_texts(solution_str):
    # Find all sections between <information> and </information>
    info_sections = re.findall(r'<information>(.*?)</information>', solution_str, re.DOTALL)

    texts = []
    for section in info_sections:
        # Extract the {text} part after each Doc (Title: xxx)
        found_texts = re.findall(r'Doc\s+\d+\(Title:\s*.*?\)\s*(.*)', section)
        texts.extend(found_texts)
    
    return texts


def compute_score_ppl(solution_str, ground_truth):
    """
    compute perplexity score, and return as the reward
    """
    highest_ppl_score = 0
    
    question = ground_truth['question']
    golden_answers = ground_truth['target'].tolist()

    searched_texts = extract_texts(solution_str=solution_str)
    titles = extract_titles(solution_str=solution_str)
    
    context_with_info = ""
    for i, text in enumerate(searched_texts):
        context_with_info += f"Doc {i+1} (Title: {titles[i]})\n{text}\n\n"
    
    for answer in golden_answers:
        retrieval_score = compute_retrieval_utility_score(
            prompt=question,
            answer=answer,
            context=context_with_info,
            num_runs=6
        )
        if retrieval_score > highest_ppl_score:
            highest_ppl_score = retrieval_score
            
        do_print = random.randint(1, 16) == 1
        
    if do_print:
        print(f"--------------------------------")
        print(f"Question: {question}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Solution string: {solution_str}")
            
    if highest_ppl_score >= 0.9:
        return 5
    if highest_ppl_score >= 0.8:
        return 4
    elif highest_ppl_score >= 0.6:
        return 3
    elif highest_ppl_score >= 0.5:
        return 1
    elif highest_ppl_score >= 0.4:
        return 0
    elif highest_ppl_score >= 0.3:
        return -1
    else:
        return -3
    
    