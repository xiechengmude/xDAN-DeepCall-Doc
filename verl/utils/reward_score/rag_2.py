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
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from generator_llms.local_inst import *

_tokenizer = SimpleTokenizer()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def answer_span_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    normalized_golden_answers = [normalize_answer(golden_answer) for golden_answer in golden_answers]
    if has_answers(normalized_prediction, normalized_golden_answers, _tokenizer, regex=False):
        score = 1
    return score


def extract_titles_and_texts(solution_str):
    """
    Extract titles and texts from information blocks, handling important documents
    and preserving their order. Properly handles duplicated documents.
    
    <important_info> tags only apply to the most recent <information> block before them,
    even if there are other tags or content in between them.
    
    If an <information> block doesn't have a corresponding <important_info> tag,
    all documents in that block are included.
    
    Returns a list of (title, text) tuples for each document.
    """
    try:
        # Extract all information blocks and important_info tags with their positions
        info_blocks = []
        important_infos = []
        
        # Find all information blocks with their positions
        info_pattern = re.compile(r'<information>(.*?)</information>', re.DOTALL)
        for match in info_pattern.finditer(solution_str):
            info_blocks.append({
                'position': match.start(),
                'content': match.group(1),
                'important_ids': None  # Will be filled if there's a matching important_info
            })
        
        # Find all important_info tags with their positions
        important_pattern = re.compile(r'<important_info>\s*\[(.*?)\]\s*</important_info>', re.DOTALL)
        for match in important_pattern.finditer(solution_str):
            # Parse important doc IDs
            important_ids = []
            try:
                for id_str in match.group(1).strip().split(','):
                    id_str = id_str.strip()
                    if id_str:  # Only process non-empty strings
                        try:
                            important_ids.append(int(id_str))
                        except ValueError:
                            print(f"Warning: Invalid document ID format: {id_str}")
                            continue
            except Exception as e:
                print(f"Warning: Error parsing important document IDs: {str(e)}")
                important_ids = []
            
            important_infos.append({
                'position': match.start(),
                'important_ids': important_ids
            })
        
        # Match each important_info with the closest preceding information block
        for imp_info in important_infos:
            # Find the closest information block that appears before this important_info
            closest_info = None
            min_distance = float('inf')
            
            for info_block in info_blocks:
                if info_block['position'] < imp_info['position']:
                    distance = imp_info['position'] - info_block['position']
                    if distance < min_distance:
                        min_distance = distance
                        closest_info = info_block
            
            # Associate this important_info with the closest preceding information block
            if closest_info:
                closest_info['important_ids'] = imp_info['important_ids']
        
        # Process each information block to extract documents
        all_docs = []
        seen_docs = set()  # To track unique documents
        
        for info_block in info_blocks:            
            info_content = info_block['content']
            important_ids = info_block['important_ids']
            
            docs_in_block = []
            try:
                # Extract individual documents from the info block
                doc_pattern = re.compile(r'Doc\s+(\d+)\(Title:\s*(.*?)\)\s*(.*?)(?=Doc\s+\d+\(Title:|$)', re.DOTALL)
                
                for match in doc_pattern.finditer(info_content):
                    try:
                        doc_id = int(match.group(1))
                        title = match.group(2).strip().replace('"', '')  # Clean up quotes
                        text = match.group(3).strip()
                        docs_in_block.append((doc_id, title, text))
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Error parsing document: {str(e)}")
                        continue
            except Exception as e:
                print(f"Warning: Error extracting documents from info block: {str(e)}")
                continue
            
            # Filter by important_ids if available
            if important_ids:
                filtered_docs = [(title, text) for doc_id, title, text in docs_in_block 
                                if doc_id in important_ids]
            else:
                # If no important_ids, include all docs
                filtered_docs = [(title, text) for _, title, text in docs_in_block]
            
            # Add unique documents to the result
            for title, text in filtered_docs:
                doc_key = (title, text)
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    all_docs.append((title, text))
        
        return all_docs
        
    except Exception as e:
        print(f"Warning: Error in extract_titles_and_texts: {str(e)}")
        return []  # Return empty list if any unexpected error occurs


def check_answer_correct(answer, golden_answers):
    answer_context_score = answer_span_check(
        prediction=answer,
        golden_answers=golden_answers
    )
    if answer_context_score == 0:
        answer_context_score = 1 if check_if_response_is_correct_llm(
            response=answer,
            gold_answers=golden_answers
        ) else 0
    return answer_context_score


def compute_score_rag(solution_str, ground_truth, zeroshot_answers, use_generation_score=True):
    """
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        zeroshot_answers: dictionary containing cached zeroshot answers
    """
    
    utility_score = 0
    generation_score = 0
    
    question = ground_truth['question']
    golden_answers = ground_truth['target'].tolist()
    
    # Get documents with titles, handling important documents
    response_str = solution_str.split("Now, start the loop with the following question:")[1]
    docs = extract_titles_and_texts(solution_str=response_str)
    
    # Build context with unique documents
    seen_docs = set()
    doc_id = 1
    context_with_info = ""
    for title, text in docs:
        doc_key = (title, text)
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            context_with_info += f"Doc {doc_id} (Title: {title})\n{text}\n\n"
            doc_id += 1
        
    answer_context = generate_answer(prompt=question, context=context_with_info)
    answer_context_score = check_answer_correct(answer=answer_context, golden_answers=golden_answers)
        
    if question in zeroshot_answers:
        answer_zeroshot = zeroshot_answers[question]['answer']
        # answer_zeroshot_score = zeroshot_answers[question]['score']
        answer_zeroshot_score = check_answer_correct(answer=answer_zeroshot, golden_answers=golden_answers)
    else:
        answer_zeroshot = generate_answer_zero_shot(prompt=question)
        answer_zeroshot_score = check_answer_correct(answer=answer_zeroshot, golden_answers=golden_answers)
        
    utility_score = answer_context_score - answer_zeroshot_score
    generation_score = answer_context_score
    
    if use_generation_score:
        score = utility_score + generation_score
    else:
        score = utility_score
    
    do_print = random.randint(1, 16) == 1
        
    if do_print:
        print(f"--------------------------------")
        print(f"Question: {question}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer_context}")
        print(f"Extracted zeroshot answer: {answer_zeroshot}")
        print(f"Answer context score: {answer_context_score}")
        print(f"Answer zeroshot score: {answer_zeroshot_score}")
        print(f"Utility score: {utility_score}")
        print(f"Generation score: {generation_score}")
        print(f"Extracted doc_info: {context_with_info}")
        print(f"Solution string: {solution_str}")
    
    return score, answer_zeroshot, answer_zeroshot_score
    