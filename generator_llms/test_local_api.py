import unittest
from generator_llms import local_api

class TestLocalAPI(unittest.TestCase):
    def test_generate_answer(self):
        print("\n=== Testing generate_answer with context ===")
        context = "The capital of France is Paris. The capital of Germany is Berlin."
        question = "What is the capital of France?"
        print(f"Context: {context}")
        print(f"Question: {question}")
        answer = local_api.generate_answer(context, question)
        print(f"Answer: {answer}")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertIn("Paris", answer)

    def test_generate_answer_zero_shot(self):
        print("\n=== Testing generate_answer_zero_shot ===")
        question = "What is the capital of France?"
        print(f"Question: {question}")
        answer = local_api.generate_answer_zero_shot(question)
        print(f"Answer: {answer}")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertIn("Paris", answer)

    def test_format_qa_prompt(self):
        print("\n=== Testing format_qa_prompt ===")
        context = "Test context"
        question = "Test question"
        print(f"Original context: {context}")
        print(f"Original question: {question}")
        formatted_context, formatted_prompt = local_api.format_qa_prompt(question, context)
        print(f"Formatted context: {formatted_context}")
        print(f"Formatted prompt: {formatted_prompt}")
        self.assertIn("Contexts:", formatted_context)
        self.assertIn(context, formatted_context)
        self.assertIn("Question:", formatted_prompt)
        self.assertIn(question, formatted_prompt)

    def test_format_zero_shot_prompt(self):
        print("\n=== Testing format_zero_shot_prompt ===")
        question = "Test question"
        print(f"Original question: {question}")
        formatted_prompt = local_api.format_zero_shot_prompt(question)
        print(f"Formatted prompt: {formatted_prompt}")
        self.assertIn("Question:", formatted_prompt)
        self.assertIn(question, formatted_prompt)
        self.assertIn("Important:", formatted_prompt)

    def test_call_llm(self):
        print("\n=== Testing call_llm ===")
        prompt = "What is 2+2?"
        print(f"Prompt: {prompt}")
        response = local_api.call_llm(prompt)
        print(f"Response: {response}")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_check_if_response_is_correct_llm(self):
        print("\n=== Testing check_if_response_is_correct_llm ===")
        response = "The answer is 4"
        gold_answers = ["4", "four"]
        print(f"Response to check: {response}")
        print(f"Gold answers: {gold_answers}")
        is_correct = local_api.check_if_response_is_correct_llm(response, gold_answers)
        print(f"Is correct: {is_correct}")
        self.assertIsInstance(is_correct, bool)

if __name__ == '__main__':
    unittest.main() 