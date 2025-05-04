import os
import random
import pandas as pd
import json
from verl.utils.reward_score.rag_2 import generate_answer, check_answer_correct
from verl.utils.hdfs_io import copy, makedirs
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import datetime

def load_previous_results(result_file):
    """Load previous results if they exist"""
    print(f"Checking for previous results at: {result_file}")
    if os.path.exists(result_file):
        print(f"Loading previous results from {result_file}")
        with open(result_file, 'r') as f:
            return json.load(f)
    print("No previous results found")
    return {}

def save_results(answers, result_file, stats_file, total_questions, data_source_stats):
    """Save results and statistics"""
    try:
        # Ensure directories exist
        cache_dir = os.path.dirname(result_file)
        stats_dir = os.path.dirname(stats_file)
        
        print(f"Creating directories if needed:")
        print(f"Cache directory: {cache_dir}")
        print(f"Stats directory: {stats_dir}")
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save answers
        print(f"Saving answers to: {result_file}")
        with open(result_file, 'w') as f:
            json.dump(answers, f)
        
        # Save statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': total_questions,
            'processed_questions': sum(len(answers) for answers in answers.values()),
            'data_source_stats': data_source_stats,
            'remaining_questions': total_questions - sum(len(answers) for answers in answers.values())
        }
        
        print(f"Saving statistics to: {stats_file}")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print("Save completed successfully")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

def load_context(question, data_source, context_dir):
    """Load context for a question from the output sequences file"""
    context_file = os.path.join(context_dir, f"{data_source}_output_sequences.json")
    if not os.path.exists(context_file):
        print(f"Warning: Context file not found: {context_file}")
        return -1
        
    with open(context_file, 'r') as f:
        sequences = json.load(f)
        if question in sequences:
            return sequences[question].get('context_with_info')
    return -1

def process_question(row, lock, context_dir):
    try:
        # Extract question and golden answers from ground_truth
        question = row['reward_model']['ground_truth']['question']
        golden_answers = row['reward_model']['ground_truth']['target']
        data_source = row['data_source']
        
        # Load context for the question
        context = load_context(question, data_source, context_dir)
        
        if context == -1:
            print(f"Warning: No context found for question: {question}")
            return None, None, None, None
            
        # Generate answer with context
        answer = generate_answer(prompt=question, context=context)
        
        # Check if answer is correct
        is_correct = check_answer_correct(answer=answer, golden_answers=golden_answers)
        
        return question, answer, is_correct, data_source
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return None, None, None, None

def process_dataset(input_file, result_file, context_dir, num_workers=16):
    # Load the dataset
    print(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    
    # Initialize counters and shared data structures
    total_questions = len(df)
    lock = threading.Lock()
    
    # Load previous results
    answers = load_previous_results(result_file)
    if not answers:
        answers = {}
    
    # Initialize data source statistics
    data_source_stats = {}
    for data_source in df['data_source'].unique():
        data_source_stats[data_source] = {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'no_context': 0
        }
    
    # Filter out already processed questions
    processed_questions_set = set()
    for data_source, questions in answers.items():
        processed_questions_set.update(questions.keys())
        # Update data source stats from previous results
        correct_count = sum(1 for info in questions.values() if info['score'] == 1)
        data_source_stats[data_source]['total'] = len(questions)
        data_source_stats[data_source]['correct'] = correct_count
        data_source_stats[data_source]['accuracy'] = correct_count / len(questions) if questions else 0
    
    remaining_df = df[~df['reward_model'].apply(lambda x: x['ground_truth']['question']).isin(processed_questions_set)]
    
    print(f"Found {sum(len(answers) for answers in answers.values())} previously processed questions")
    print(f"Remaining questions to process: {len(remaining_df)}")
    
    # Create stats file path
    stats_file = result_file.replace('.json', '_stats.json')
    print(f"Stats file will be saved to: {stats_file}")
    
    # Process remaining questions in parallel
    print(f"Processing remaining questions with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_question, row, lock, context_dir): idx for idx, row in remaining_df.iterrows()}
        
        # Process results as they complete
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                question, answer, is_correct, data_source = future.result()
                
                with lock:
                    if question is not None:  # Valid context found
                        # Initialize data source if not exists
                        if data_source not in answers:
                            answers[data_source] = {}
                        
                        # Store results
                        answers[data_source][question] = {
                            'answer': answer,
                            'score': 1 if is_correct else 0
                        }
                        
                        # Update data source statistics
                        data_source_stats[data_source]['total'] += 1
                        if is_correct:
                            data_source_stats[data_source]['correct'] += 1
                        data_source_stats[data_source]['accuracy'] = (
                            data_source_stats[data_source]['correct'] / 
                            data_source_stats[data_source]['total']
                        )
                    else:  # No context found
                        data_source_stats[data_source]['no_context'] += 1
                        
                    # Print statistics every 100 steps
                    if pbar.n % 100 == 0:
                        print("\nCurrent statistics per data source:")
                        for source, stats in data_source_stats.items():
                            print(f"{source}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")
                
                pbar.update(1)
    
    # Save final results
    print("\nSaving final results")
    save_results(answers, result_file, stats_file, total_questions, data_source_stats)
    
    # Print final statistics
    print("\nFinal Statistics per Data Source:")
    for source, stats in data_source_stats.items():
        print(f"{source}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")
    print(f"\nResults saved to: {result_file}")
    print(f"Statistics saved to: {stats_file}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="data/nq_hotpotqa_train/test_e5_ug.parquet", help='Path to input parquet file')
    parser.add_argument('--result_file', default="results/u1_step150.json", help='Path to save answers JSON file')
    parser.add_argument('--context_dir', default="data/output_sequences_150", help='Directory containing context files')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker threads to use')
    
    args = parser.parse_args()
    process_dataset(args.input_file, args.result_file, args.context_dir, args.num_workers) 