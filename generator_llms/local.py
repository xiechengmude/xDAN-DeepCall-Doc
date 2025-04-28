import requests
import math
from transformers import AutoTokenizer

# Load matching tokenizer locally
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def compute_alternative_perplexity(prompt: str, answer: str) -> float:
    """
    Calculate perplexity using the sliding window approach.
    This makes multiple API calls for each token in the answer.
    """
    print(f"Computing perplexity for prompt: '{prompt}'")
    
    headers = {"Content-Type": "application/json"}
    
    # Prepare the base prompt
    base_prompt = prompt.strip() + "\nAnswer: "
    
    # Tokenize the answer to get its length
    answer_tokens = tokenizer.encode(answer.strip(), add_special_tokens=False)
    answer_token_strings = tokenizer.convert_ids_to_tokens(answer_tokens)
    print(f"Answer has {len(answer_tokens)} tokens: {answer_token_strings}")
    
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
            "model": "mistralai/Mistral-7B-v0.1",
            "prompt": current_prompt,
            "max_tokens": 1,
            "logprobs": True
        }
        
        try:
            response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=payload)
            response.raise_for_status()
            res = response.json()
            
            if "choices" not in res or not res["choices"] or "logprobs" not in res["choices"][0]:
                print(f"Warning: Invalid response at position {i}: {res}")
                continue
                
            # Get the logprob for the most likely next token
            top_logprobs = res["choices"][0]["logprobs"].get("top_logprobs", [{}])
            
            if not top_logprobs or not isinstance(top_logprobs[0], dict):
                print(f"Warning: No top logprobs at position {i}")
                continue
                
            # Find the most likely next token and its logprob
            next_token_prob = None
            token_matched = False
            
            for token, logprob in top_logprobs[0].items():
                if token.strip() in next_text or next_text.strip() in token:
                    next_token_prob = logprob
                    token_matched = True
                    print(f"Token {i+1}/{len(answer_tokens)}: '{next_text}' - logprob: {logprob:.4f}")
                    break
                    
            if not token_matched and top_logprobs[0]:
                # If no match, use the most likely token's probability
                most_likely_token = max(top_logprobs[0].items(), key=lambda x: x[1])
                next_token_prob = most_likely_token[1]
                print(f"Token {i+1}/{len(answer_tokens)}: '{next_text}' - No exact match, using most likely token '{most_likely_token[0]}' with logprob: {next_token_prob:.4f}")
                
            if next_token_prob is not None:
                total_logprob += next_token_prob
                token_logprobs.append(next_token_prob)
                
            # Update the cumulative answer
            cumulative_answer += next_text
            
        except Exception as e:
            print(f"Error at position {i}: {e}")
            # Continue with the next token
    
    # Calculate perplexity
    if total_logprob == 0 or len(answer_tokens) == 0:
        print("Warning: Could not calculate logprobs, returning default value")
        return 1.0
        
    avg_logprob = total_logprob / len(answer_tokens)
    perplexity = math.exp(-avg_logprob)
    
    # Show token-by-token breakdown
    print("\nToken-by-token breakdown:")
    for i, (token, logprob) in enumerate(zip(answer_token_strings, token_logprobs)):
        print(f"{i+1}. '{token}': logprob = {logprob:.4f}, perplexity = {math.exp(-logprob):.4f}")
    
    print(f"\nTotal logprob: {total_logprob:.4f}")
    print(f"Average logprob: {avg_logprob:.4f}")
    print(f"Overall perplexity: {perplexity:.4f}")
    
    return perplexity


def compare_hotpotqa_style_examples():
    """
    Compare perplexity of short, factual answers with and without context,
    similar to HotpotQA examples.
    """
    # Define several test cases more similar to HotpotQA
    test_cases = [
        {
            "name": "Birth Year",
            "question": "When was Christopher Columbus born?",
            "context": "Christopher Columbus was an Italian explorer and navigator born in 1451 in the Republic of Genoa.",
            "answer": "1451"
        },
        {
            "name": "Nationality",
            "question": "What nationality was Marie Curie?",
            "context": "Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity.",
            "answer": "Polish"
        },
        {
            "name": "Capital City",
            "question": "What is the capital of Denmark?",
            "context": "Denmark is a Nordic country with a population of around 5.8 million. Its capital and largest city is Copenhagen.",
            "answer": "Copenhagen"
        }
    ]
    
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


if __name__ == "__main__":
    compare_hotpotqa_style_examples()