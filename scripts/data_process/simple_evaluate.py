import json
import pandas as pd
import time
from generator_llms.claude_api import get_claude_response
import concurrent.futures
from tqdm import tqdm

def evaluate_answer(question, rag_answer, golden_answers):
    # Convert golden_answers to string if it's a list
    if isinstance(golden_answers, list):
        golden_answers = "\n".join([f"- {ans}" for ans in golden_answers])
    
    prompt = f"""You are an evaluator for question-answering systems. Your task is to determine if the given answer aligns with the golden answers.

Question: {question}

RAG System's Answer: {rag_answer}

Golden Answers (reference):
{golden_answers}

Please evaluate if the RAG system's answer aligns with the golden answers. The answer should be considered correct if it:
- Contains the same key information as the golden answers
- Expresses the same meaning, even if using different words
- Is factually consistent with the golden answers

Respond with ONLY "yes" if the answer aligns with the golden answers, or "no" if it does not. Do not include any other text or explanation."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = get_claude_response(prompt, llm="sonnet", temperature=0)
            response = response.strip().lower()
            
            # Check if response is valid
            if response in ['yes', 'no']:
                return 1 if response == 'yes' else 0
            else:
                print(f"Invalid response: {response}. Retrying...")
                time.sleep(1)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    
    print(f"Failed to get valid response after {max_retries} attempts")
    return None

def process_row(row):
    # Convert golden_answer string to list if it's in string format
    golden_answers = row['golden_answer']
    if isinstance(golden_answers, str):
        try:
            golden_answers = eval(golden_answers)  # Convert string representation of list to actual list
        except:
            golden_answers = [golden_answers]  # If conversion fails, use as single answer
    
    # Get evaluation
    judgement = evaluate_answer(row['question'], row['rag_answer'], golden_answers)
    return judgement

def main():
    # Read the CSV file
    df = pd.read_csv('sampled_rag_results.csv')
    
    # Add human_judgement column
    df['human_judgement'] = None
    
    # Process rows in parallel
    max_workers = 5  # Adjust this based on your API rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = []
        for idx, row in df.iterrows():
            futures.append((idx, executor.submit(process_row, row)))
        
        # Process results as they complete
        for idx, future in tqdm(futures, desc="Evaluating answers"):
            try:
                judgement = future.result()
                df.at[idx, 'human_judgement'] = judgement
                
                # Save progress every 10 samples
                if (idx + 1) % 10 == 0:
                    df.to_csv('sampled_rag_results_with_judgement.csv', index=False)
                    print(f"\nProgress saved at {idx + 1} samples")
            except Exception as e:
                print(f"\nError processing row {idx}: {str(e)}")
    
    # Save final results
    df.to_csv('sampled_rag_results_with_judgement.csv', index=False)
    
    # Print statistics
    print("\nEvaluation Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Correct answers: {df['human_judgement'].sum()}")
    print(f"Accuracy: {(df['human_judgement'].sum() / len(df)) * 100:.2f}%")
    
    # Compare with RAG scores
    print("\nComparison with RAG scores:")
    print("RAG Score Statistics:")
    print(df['rag_score'].value_counts(normalize=True))
    print("\nHuman Judgement Statistics:")
    print(df['human_judgement'].value_counts(normalize=True))

if __name__ == '__main__':
    main() 