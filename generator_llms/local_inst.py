import os
import requests
import time
from requests.exceptions import RequestException
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import threading

# Load matching tokenizer locally
# MODEL = "generator_llms/Qwen2.5-14B-Instruct-Q5_K_M.gguf"  
# MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
# TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
MODEL = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
TOKENIZER_MODEL = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
 
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

# Create a semaphore to limit concurrent requests
MAX_CONCURRENT_REQUESTS = 4  # Adjust based on your server's capacity
request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

def generate_answer(context: str, prompt: str, max_retries=3) -> str:
    """
    Generate an answer using the local LLM given context and prompt.
    
    Args:
        context: The context information
        prompt: The question or prompt to answer
        max_retries: Maximum number of retry attempts
        
    Returns:
        str: The generated answer
    """
    headers = {"Content-Type": "application/json"}
    
    # Format the context and prompt for chat completion
    messages = format_qa_chat_messages(prompt, context)
    
    # Prepare the payload for the API call
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0    
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            res = response.json()
            
            if "choices" not in res or not res["choices"]:
                raise ValueError("Invalid response from LLM")
                
            # Extract and return the generated text
            return res["choices"][0]["message"]["content"].strip()
            
        except (RequestException, ValueError) as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error generating answer after {max_retries} attempts: {e}")
                return ""
            # Exponential backoff: wait longer between each retry
            time.sleep(2 ** attempt)
            print(f"Retry attempt {attempt + 1}/{max_retries} after error: {e}")

def format_qa_chat_messages(question: str, context: str) -> list:
    """
    Format a question and context into chat messages for instruction-tuned models.
    
    Args:
        question: The question to be answered
        context: The context information
        
    Returns:
        list: List of message dictionaries
    """
    system_message = f"""Use the following contexts (some might be irrelevant) on demand:
Contexts:
{context}

Important: You MUST directly answer the question without any other text and thinking."""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]
    
    return messages


def generate_answer_zero_shot(prompt: str) -> str:
    """
    Generate an answer using the local LLM with a zero-shot prompt.
    
    Args:
        prompt: The question or prompt to answer
        
    Returns:
        str: The generated answer
    """
    headers = {"Content-Type": "application/json"}
    
    # Format the prompt as chat messages
    messages = format_zero_shot_chat_messages(prompt)
    
    # Prepare the payload for the API call
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0    
    }
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "choices" not in res or not res["choices"]:
            raise ValueError("Invalid response from LLM")
            
        # Extract and return the generated text
        return res["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""
    

def format_zero_shot_chat_messages(question: str) -> list:
    """
    Format a question into chat messages for zero-shot prompting.
    
    Args:
        question: The question to be answered
        
    Returns:
        list: List of message dictionaries
    """
    system_message = "Important: You MUST directly answer the question without any other text and thinking."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]
    
    return messages


def call_llm(prompt: str) -> str:
    """
    Call the LLM with a simple prompt and get a short response.
    """
    # Prepare the payload for the API call
    headers = {"Content-Type": "application/json"}
    
    # Format as chat messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 10,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0    
    }
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "choices" not in res or not res["choices"]:
            raise ValueError("Invalid response from LLM")
            
        # Extract and return the generated text
        return res["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

def check_if_response_is_correct_llm(response: str, gold_answers: list[str]) -> bool:
    """
    Check if the generated answer is correct by comparing it to the gold answers.
    
    Args:
        response: The response from the LLM
        gold_answers: The gold answers
        
    Returns:
        bool: True if the generated answer is correct, False otherwise
    """
    
    prompt = f"Please check if any of the golden answers is contained in the following response: {response}\n\nGolden answers: {str(gold_answers)}\n\nPlease directly answer with 'yes' or 'no'."
    
    yes_or_no = call_llm(prompt)
    
    if "yes" in yes_or_no.lower():
        return True
    elif "no" in yes_or_no.lower():
        return False
    else:
        yes_or_no = call_llm(prompt)
        if "yes" in yes_or_no.lower():
            return True
        elif "no" in yes_or_no.lower():
            return False
        else:
            return False

def check_if_context_contains_golden_answers(context: str, gold_answers: list[str]) -> bool:
    """
    Check if any of the golden answers are semantically included in the context using soft matching.
    
    Args:
        context: The context to check
        gold_answers: List of golden answers to look for
        
    Returns:
        bool: True if any golden answer is semantically found in the context, False otherwise
    """
    prompt = f"""Please analyze the following context and determine if it contains information that matches or is semantically equivalent to any of the golden answers.

Context:
{context}

Golden answers: {str(gold_answers)}

Important instructions:
1. Look for equivalent information even if worded differently
2. Answer with just 'yes' if you find any relevant match, or 'no' if none of the golden answers are present in any form

Please directly answer with 'yes' or 'no'."""
    
    yes_or_no = call_llm(prompt)
    
    if "yes" in yes_or_no.lower():
        return True
    elif "no" in yes_or_no.lower():
        return False
    else:
        # If first attempt didn't give clear yes/no, try one more time
        yes_or_no = call_llm(prompt)
        if "yes" in yes_or_no.lower():
            return True
        elif "no" in yes_or_no.lower():
            return False
        else:
            return False

def generate_answer_with_semaphore(context: str, prompt: str) -> str:
    """
    Generate an answer with rate limiting using a semaphore.
    """
    with request_semaphore:
        return generate_answer(context, prompt)

def process_batch(requests: list[tuple[str, str]], max_workers=4) -> list[str]:
    """
    Process a batch of requests with rate limiting.
    
    Args:
        requests: List of (context, prompt) tuples
        max_workers: Maximum number of worker threads
        
    Returns:
        list[str]: List of generated answers
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_answer_with_semaphore, context, prompt)
            for context, prompt in requests
        ]
        return [future.result() for future in futures]