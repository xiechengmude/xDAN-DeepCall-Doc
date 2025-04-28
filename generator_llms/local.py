import requests
import math
from transformers import AutoTokenizer

# Load matching tokenizer locally
MODEL = "Qwen/Qwen2.5-7B"  
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def compute_alternative_perplexity(prompt: str, answer: str, num_runs: int = 1) -> float:
    """
    Calculate perplexity using the sliding window approach.
    This makes multiple API calls for each token in the answer.
    Runs multiple times and returns the average perplexity after filtering outliers.
    Uses GPT-2 Small model for more stable perplexity computation.
    """
    print(f"Computing perplexity for prompt: '{prompt}'")
    print(f"Will run {num_runs} times and take average after filtering outliers")
    print(f"Using model: {MODEL}")
    
    all_perplexities = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        headers = {"Content-Type": "application/json"}
        
        # Prepare the base prompt
        base_prompt = prompt.strip() + "\nThe answer is "
        
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
                "model": MODEL,
                "prompt": current_prompt,
                "max_tokens": 1,
                "logprobs": True,
                "temperature": 0,  # Use deterministic sampling
                "top_p": 1.0,  # No top-p sampling
                "frequency_penalty": 0.0,  # No frequency penalty
                "presence_penalty": 0.0  # No presence penalty
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
        
        # Calculate perplexity for this run
        if total_logprob == 0 or len(answer_tokens) == 0:
            print("Warning: Could not calculate logprobs for this run, skipping")
            continue
            
        avg_logprob = total_logprob / len(answer_tokens)
        perplexity = math.exp(-avg_logprob)
        all_perplexities.append(perplexity)
        
        # Show token-by-token breakdown for this run
        print("\nToken-by-token breakdown:")
        for i, (token, logprob) in enumerate(zip(answer_token_strings, token_logprobs)):
            print(f"{i+1}. '{token}': logprob = {logprob:.4f}, perplexity = {math.exp(-logprob):.4f}")
        
        print(f"\nRun {run + 1} results:")
        print(f"Total logprob: {total_logprob:.4f}")
        print(f"Average logprob: {avg_logprob:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
    
    if not all_perplexities:
        print("Warning: No valid perplexity calculations, returning default value")
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
        print("Warning: All values were filtered as outliers, using original values")
        filtered_perplexities = all_perplexities
    
    # Calculate and return the average perplexity after filtering outliers
    avg_perplexity = sum(filtered_perplexities) / len(filtered_perplexities)
    print(f"\nFinal Results (after filtering outliers):")
    print(f"Original values: {all_perplexities}")
    print(f"Filtered values: {filtered_perplexities}")
    print(f"Number of outliers removed: {len(all_perplexities) - len(filtered_perplexities)}")
    print(f"Average perplexity: {avg_perplexity:.4f}")
    print(f"Standard deviation: {math.sqrt(sum((x - avg_perplexity) ** 2 for x in filtered_perplexities) / len(filtered_perplexities)):.4f}")
    
    return avg_perplexity


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
            "context": """
            Christopher Columbus was an Italian explorer and navigator born in 1451 in the Republic of Genoa.
            
            The Renaissance was a period in European history that spanned from the 14th to the 17th century.
            During this time, many great artists like Leonardo da Vinci and Michelangelo created their masterpieces.
            The printing press was invented by Johannes Gutenberg around 1440, revolutionizing the spread of information.
            
            In 1492, Columbus sailed across the Atlantic Ocean, hoping to find a new route to Asia.
            His voyages were sponsored by the Catholic Monarchs of Spain, Ferdinand and Isabella.
            The Spanish Inquisition was established in 1478 to maintain Catholic orthodoxy in Spain.
            
            The Ottoman Empire was expanding during this period, capturing Constantinople in 1453.
            The Black Death had devastated Europe in the 14th century, killing millions of people.
            The Hundred Years' War between England and France ended in 1453.
            """,
            "answer": "1451"
        },
        {
            "name": "Nationality",
            "question": "What nationality was Marie Curie?",
            "context": """
            Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity.
            
            The field of physics saw many breakthroughs in the late 19th and early 20th centuries.
            Albert Einstein published his theory of relativity in 1905, revolutionizing our understanding of space and time.
            Niels Bohr developed the Bohr model of the atom in 1913, which explained atomic structure.
            
            Poland has a rich history of scientific contributions.
            Nicolaus Copernicus, who proposed the heliocentric model of the solar system, was also Polish.
            The Polish-Lithuanian Commonwealth was one of the largest and most populous countries in 16th and 17th century Europe.
            
            Radioactivity was first discovered by Henri Becquerel in 1896.
            The Curie family made significant contributions to the study of radiation.
            Pierre Curie, Marie's husband, was a French physicist who worked alongside her.
            """,
            "answer": "Polish"
        },
        {
            "name": "Capital City",
            "question": "What is the capital of Denmark?",
            "context": """
            Denmark is a Nordic country with a population of around 5.8 million. Its capital and largest city is Copenhagen.
            
            The Nordic countries include Sweden, Norway, Finland, Iceland, and Denmark.
            These countries are known for their high standard of living and social welfare systems.
            The Scandinavian Peninsula consists of Sweden and Norway, while Denmark is located on the Jutland Peninsula.
            
            Copenhagen is famous for its historic architecture and modern design.
            The Little Mermaid statue, based on Hans Christian Andersen's fairy tale, is a major tourist attraction.
            Tivoli Gardens, one of the oldest amusement parks in the world, is located in Copenhagen.
            
            The Danish monarchy is one of the oldest in the world, dating back to the Viking Age.
            Denmark is a constitutional monarchy with a parliamentary system.
            The country is known for its bicycle culture and environmental sustainability initiatives.
            """,
            "answer": "Copenhagen"
        },
        {
            "name": "Scientific Discovery",
            "question": "What did Marie Curie discover and when?",
            "context": """
            Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity.
            In 1898, she and her husband Pierre discovered two new elements: polonium and radium.
            
            The field of physics saw many breakthroughs in the late 19th and early 20th centuries.
            Albert Einstein published his theory of relativity in 1905, revolutionizing our understanding of space and time.
            Niels Bohr developed the Bohr model of the atom in 1913, which explained atomic structure.
            
            Radioactivity was first discovered by Henri Becquerel in 1896 when he noticed that uranium salts emitted rays that could expose photographic plates.
            The Curie family made significant contributions to the study of radiation.
            Pierre Curie, Marie's husband, was a French physicist who worked alongside her.
            
            The Nobel Prize in Physics was awarded to Becquerel and the Curies in 1903 for their work on radioactivity.
            Marie Curie later won a second Nobel Prize in Chemistry in 1911 for her discovery of radium and polonium.
            """,
            "answer": "Marie Curie discovered the elements polonium and radium in 1898"
        },
        {
            "name": "Historical Event",
            "question": "What happened during Columbus's first voyage to the Americas?",
            "context": """
            Christopher Columbus was an Italian explorer and navigator born in 1451 in the Republic of Genoa.
            In 1492, Columbus set sail from Spain with three ships: the Santa Maria, the Pinta, and the Niña.
            
            The Renaissance was a period in European history that spanned from the 14th to the 17th century.
            During this time, many great artists like Leonardo da Vinci and Michelangelo created their masterpieces.
            The printing press was invented by Johannes Gutenberg around 1440, revolutionizing the spread of information.
            
            Columbus's first voyage lasted from August 3, 1492, to March 15, 1493.
            He landed in the Bahamas on October 12, 1492, thinking he had reached Asia.
            The Santa Maria ran aground on Christmas Day 1492 and had to be abandoned.
            
            The Ottoman Empire was expanding during this period, capturing Constantinople in 1453.
            The Black Death had devastated Europe in the 14th century, killing millions of people.
            The Hundred Years' War between England and France ended in 1453.
            """,
            "answer": "Columbus sailed from Spain with three ships, landed in the Bahamas on October 12, 1492, and the Santa Maria was lost on Christmas Day"
        },
        {
            "name": "Geographical Feature",
            "question": "What are some notable geographical features of Denmark?",
            "context": """
            Denmark is a Nordic country with a population of around 5.8 million. Its capital and largest city is Copenhagen.
            
            The Nordic countries include Sweden, Norway, Finland, Iceland, and Denmark.
            These countries are known for their high standard of living and social welfare systems.
            The Scandinavian Peninsula consists of Sweden and Norway, while Denmark is located on the Jutland Peninsula.
            
            Denmark consists of the Jutland Peninsula and over 400 islands, with Zealand being the largest.
            The country is mostly flat, with its highest point being Møllehøj at 170.86 meters above sea level.
            Denmark has a long coastline and is surrounded by the North Sea and the Baltic Sea.
            
            The Danish monarchy is one of the oldest in the world, dating back to the Viking Age.
            Denmark is a constitutional monarchy with a parliamentary system.
            The country is known for its bicycle culture and environmental sustainability initiatives.
            """,
            "answer": "Denmark consists of the Jutland Peninsula and over 400 islands, with Zealand being the largest, and has a mostly flat landscape with its highest point at Møllehøj"
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