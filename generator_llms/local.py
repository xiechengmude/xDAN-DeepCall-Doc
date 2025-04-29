import requests
import math
from transformers import AutoTokenizer
from generator_llms.test_cases import test_cases
# Load matching tokenizer locally
MODEL = "Qwen/Qwen3-14B"  
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def compute_alternative_perplexity(prompt: str, answer: str, num_runs: int = 10) -> float:
    """
    Calculate perplexity using the sliding window approach.
    This makes multiple API calls for each token in the answer.
    Runs multiple times and returns the average perplexity after filtering outliers.
    Uses GPT-2 Small model for more stable perplexity computation.
    """
    # print(f"Computing perplexity for prompt: '{prompt}'")
    # print(f"Will run {num_runs} times and take average after filtering outliers")
    # print(f"Using model: {MODEL}")
    
    all_perplexities = []
    
    for run in range(num_runs):
        # print(f"\nRun {run + 1}/{num_runs}")
        headers = {"Content-Type": "application/json"}
        
        # Prepare the base prompt
        base_prompt = prompt.strip() + "\nThe answer is "
        
        # Tokenize the answer to get its length
        answer_tokens = tokenizer.encode(answer.strip(), add_special_tokens=False)
        answer_token_strings = tokenizer.convert_ids_to_tokens(answer_tokens)
        # print(f"Answer has {len(answer_tokens)} tokens: {answer_token_strings}")
        
        # Go through the answer token by token
        cumulative_answer = ""
        total_logprob = 0
        token_logprobs = []
        
        # For efficiency, we'll process in small batches
        batch_size = 1  # Process one token at a time for accuracy
        
        for i in range(0, len(answer_tokens), batch_size):
            # Get the next batch of tokens
            batch_end = min(i + batch_size, len(answer_tokens))
            next_tokens = answer_tokens[i:batch_end]
            next_text = tokenizer.decode(next_tokens)
            
            # Current context is the base prompt + answer so far
            current_prompt = base_prompt + cumulative_answer
            
            # Make API call to get probability of the next token(s)
            payload = {
                "model": MODEL,
                "prompt": current_prompt,
                "max_tokens": 1,
                "logprobs": True,
                "temperature": 0.7,  # Use deterministic sampling
                "top_p": 1.0,  # No top-p sampling
                "frequency_penalty": 0.0,  # No frequency penalty
                "presence_penalty": 0.0  # No presence penalty
            }
            
            try:
                response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=payload)
                response.raise_for_status()
                res = response.json()
                
                if "choices" not in res or not res["choices"] or "logprobs" not in res["choices"][0]:
                    # print(f"Warning: Invalid response at position {i}: {res}")
                    continue
                    
                # Get the logprob for the most likely next token
                top_logprobs = res["choices"][0]["logprobs"].get("top_logprobs", [{}])
                
                if not top_logprobs or not isinstance(top_logprobs[0], dict):
                    # print(f"Warning: No top logprobs at position {i}")
                    continue
                    
                # Find the most likely next token and its logprob
                next_token_prob = None
                token_matched = False
                
                for token, logprob in top_logprobs[0].items():
                    if token.strip() in next_text or next_text.strip() in token:
                        next_token_prob = logprob
                        token_matched = True
                        # print(f"Token {i+1}/{len(answer_tokens)}: '{next_text}' - logprob: {logprob:.4f}")
                        break
                        
                if not token_matched and top_logprobs[0]:
                    # If no match, use the most likely token's probability
                    most_likely_token = max(top_logprobs[0].items(), key=lambda x: x[1])
                    next_token_prob = most_likely_token[1]
                    # print(f"Token {i+1}/{len(answer_tokens)}: '{next_text}' - No exact match, using most likely token '{most_likely_token[0]}' with logprob: {next_token_prob:.4f}")
                    
                if next_token_prob is not None:
                    total_logprob += next_token_prob
                    token_logprobs.append(next_token_prob)
                    
                # Update the cumulative answer
                cumulative_answer += next_text
                
            except Exception as e:
                print(f"Error at position {i}: {e}")
                # Continue with the next token
        
        # Calculate perplexity for this run
        if total_logprob == 0 or len(answer_tokens) == 0:
            # print("Warning: Could not calculate logprobs for this run, skipping")
            continue
            
        avg_logprob = total_logprob / len(answer_tokens)
        perplexity = math.exp(-avg_logprob)
        all_perplexities.append(perplexity)
        
        # Show token-by-token breakdown for this run
        # print("\nToken-by-token breakdown:")
        # for i, (token, logprob) in enumerate(zip(answer_token_strings, token_logprobs)):
            # print(f"{i+1}. '{token}': logprob = {logprob:.4f}, perplexity = {math.exp(-logprob):.4f}")
        
        # print(f"\nRun {run + 1} results:")
        # print(f"Total logprob: {total_logprob:.4f}")
        # print(f"Average logprob: {avg_logprob:.4f}")
        # print(f"Perplexity: {perplexity:.4f}")
    
    if not all_perplexities:
        # print("Warning: No valid perplexity calculations, returning default value")
        return 1.0
    
    def filter_outliers(values, max_iterations=5):
        """Recursively filter outliers until values stabilize or max iterations reached"""
        if len(values) < 3 or max_iterations == 0:
            return values
            
        sorted_values = sorted(values)
        q1 = sorted_values[len(sorted_values)//4]
        q3 = sorted_values[3*len(sorted_values)//4]
        iqr = q3 - q1
        
        # Use a more aggressive threshold for multiple outliers
        lower_bound = q1 - 2.0 * iqr
        upper_bound = q3 + 2.0 * iqr
        
        filtered = [v for v in values if lower_bound <= v <= upper_bound]
        
        # If no values were filtered or we've reached the minimum allowed values, return
        if len(filtered) == len(values) or len(filtered) < 3:
            return filtered
            
        # Recursively filter remaining values
        return filter_outliers(filtered, max_iterations - 1)
    
    # Apply recursive outlier filtering
    filtered_perplexities = filter_outliers(all_perplexities)
    
    if not filtered_perplexities:
        # print("Warning: All values were filtered as outliers, using original values")
        filtered_perplexities = all_perplexities
    
    # Calculate and return the average perplexity after filtering outliers
    avg_perplexity = sum(filtered_perplexities) / len(filtered_perplexities)
    # print(f"\nFinal Results (after filtering outliers):")
    # print(f"Original values: {all_perplexities}")
    # print(f"Filtered values: {filtered_perplexities}")
    # print(f"Number of outliers removed: {len(all_perplexities) - len(filtered_perplexities)}")
    # print(f"Average perplexity: {avg_perplexity:.4f}")
    # print(f"Standard deviation: {math.sqrt(sum((x - avg_perplexity) ** 2 for x in filtered_perplexities) / len(filtered_perplexities)):.4f}")
    
    return avg_perplexity


def compare_hotpotqa_style_examples():
    """
    Compare perplexity of short, factual answers with and without context,
    similar to HotpotQA examples.
    """
    # Define several test cases more similar to HotpotQA
    
    results = []
    
    for case in test_cases:
        print("\n" + "=" * 80)
        print(f"TEST CASE: {case['name']}")
        print("=" * 80)
        
        # Case 1: Question only
        print("\nCASE 1: QUESTION ONLY")
        print("-" * 40)
        question_only_ppl = compute_alternative_perplexity(case['question'], case['answer'])
        
        # Case 2: Context + Question
        print("\nCASE 2: CONTEXT + QUESTION")
        print("-" * 40)
        context_question_ppl = compute_alternative_perplexity(case['context'] + "\n\n" + case['question'], case['answer'])
        
        # Store results
        result = {
            "name": case['name'],
            "question": case['question'],
            "answer": case['answer'],
            "q_only_ppl": question_only_ppl,
            "context_q_ppl": context_question_ppl,
            "ratio": question_only_ppl / context_question_ppl if context_question_ppl > 0 else float('inf')
        }
        results.append(result)
    
    # Summary of all results
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL RESULTS")
    print("=" * 80)
    print(f"{'Test Case':<15} {'Q-Only PPL':<15} {'Context+Q PPL':<15} {'Ratio (Q/C+Q)':<15} {'Effect':<20}")
    print("-" * 80)
    
    for result in results:
        effect = "HELPS ✓" if result['q_only_ppl'] > result['context_q_ppl'] else "HURTS ✗"
        print(f"{result['name']:<15} {result['q_only_ppl']:<15.4f} {result['context_q_ppl']:<15.4f} {result['ratio']:<15.4f} {effect:<20}")


def compute_retrieval_utility_score(prompt: str, answer: str, context: str, num_runs: int = 10) -> float:
    """
    Calculate retrieval utility score by comparing context + question vs question-only scenarios.
    Runs multiple times and returns the ratio of times context + question wins.
    
    Args:
        prompt: The question to be answered
        answer: The expected answer
        context: The context to be added to the question
        num_runs: Number of times to run the comparison
        
    Returns:
        float: The ratio of times context + question had lower perplexity than question-only
    """
    # print(f"Computing retrieval utility score for question: '{prompt}'")
    # print(f"Will run {num_runs} comparisons")
    
    context_wins = 0
    prompt = prompt.lower()
    answer = answer.lower()
    context = context.lower()
    
    for run in range(num_runs):
        # print(f"\nRun {run + 1}/{num_runs}")
        
        # Case 1: Question only
        # print("\nCASE 1: QUESTION ONLY")
        # print("-" * 40)
        question_only_ppl = compute_alternative_perplexity(prompt, answer, num_runs=1)
        
        # Case 2: Context + Question
        # print("\nCASE 2: CONTEXT + QUESTION")
        # print("-" * 40)
        context_question_ppl = compute_alternative_perplexity(context + "\n\n" + prompt, answer, num_runs=1)
        
        # Compare perplexities
        if context_question_ppl < question_only_ppl:
            context_wins += 1
        #     print("✓ Context + Question wins this round")
        # else:
        #     print("✗ Question only wins this round")
    
    # Calculate and return the win ratio
    win_ratio = context_wins / num_runs
    # print(f"\nFinal Results:")
    # print(f"Context + Question wins: {context_wins}/{num_runs}")
    print(f"Retrieval Utility Score: {win_ratio:.4f}")
    
    return win_ratio


if __name__ == "__main__":
    # Example test case
    
    for test_case in test_cases:
    
        print("\n" + "=" * 80)
        print(f"TEST CASE: {test_case['name']}")
        print("=" * 80)
        
        # Calculate retrieval utility score
        retrieval_score = compute_retrieval_utility_score(
            prompt=test_case['question'],
            answer=test_case['answer'],
            context=test_case['context'],
            num_runs=10
        )
        
        print("\n" + "=" * 80)
        print("FINAL RETRIEVAL UTILITY SCORE")
        print("=" * 80)
        print(f"Score: {retrieval_score:.4f}")
        print(f"Interpretation: Context + Question was better {retrieval_score*100:.1f}% of the time")