import json
import os
import pandas as pd
import datasets
from typing import Dict, List

def process_benchmark_data(file_path: str) -> List[Dict]:
    """Process benchmark.json into the required format."""
    with open(file_path, 'r') as f:
        benchmark_data = json.load(f)
    
    processed_data = []
    
    for data_source, questions in benchmark_data.items():
        for q_id, q_data in questions.items():
            # Get the question and options
            question = q_data['question']
            options = q_data['options']
            
            # Create the full question with options
            full_question = f"{question}\nOptions:\n"
            for opt_key, opt_text in options.items():
                full_question += f"{opt_key}: {opt_text}\n"
            
            # Get the correct answer text based on the answer key
            answer_key = q_data['answer']
            golden_answer = [f"{answer_key}: {options[answer_key]}"]
            
            # Create the data point in the required format
            data_point = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": full_question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "question": full_question,
                        "target": golden_answer,
                        "gt_docs": []  # Empty list as we don't have supporting facts
                    }
                },
                "extra_info": {
                    'split': 'test',
                    'index': q_id,
                }
            }
            
            processed_data.append(data_point)
    
    return processed_data

def main():
    # Input and output paths
    input_file = "data/mirage/benchmark.json"
    output_file = "data/mirage/mirage_test.parquet"
    
    # Process the data
    processed_data = process_benchmark_data(input_file)
    
    # Convert to dataset and save as parquet
    dataset = datasets.Dataset.from_list(processed_data)
    dataset.to_parquet(output_file)
    
    # Print statistics
    print(f"Total number of questions: {len(dataset)}")
    for data_source in dataset.unique('data_source'):
        count = len(dataset.filter(lambda x: x['data_source'] == data_source))
        print(f"{data_source}: {count}")

if __name__ == "__main__":
    main() 