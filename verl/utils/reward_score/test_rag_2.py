from verl.utils.reward_score.rag_2 import extract_titles_and_texts

def run_test_case(name, test_input, expected_count, expected_titles):
    print(f"\n=== Testing: {name} ===")
    print(f"Input: {test_input}")
    result = extract_titles_and_texts(test_input)
    
    print(f"Expected count: {expected_count}")
    print(f"Actual count: {len(result)}")
    print("Expected titles:", expected_titles)
    print("Actual titles:", [title for title, _ in result])
    
    success = len(result) == expected_count and all(
        result[i][0] == expected_titles[i] for i in range(len(result))
    )
    print("✅ PASS" if success else "❌ FAIL")
    return success

def main():
    # Common test documents
    doc1 = 'Doc 1(Title: "Test Doc 1") This is the content of document 1'
    doc2 = 'Doc 2(Title: "Test Doc 2") This is the content of document 2'
    doc3 = 'Doc 3(Title: "Test Doc 3") This is the content of document 3'
    
    def create_test_input(info_content, important_info=None):
        base = f'<information>{info_content}</information>'
        if important_info:
            base += f'\n<important_info>{important_info}</important_info>'
        return base

    # Test cases
    test_cases = [
        {
            "name": "Basic numeric IDs [1,2]",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}", "[1,2]"),
            "expected_count": 2,
            "expected_titles": ["Test Doc 1", "Test Doc 2"]
        },
        {
            "name": "Spaced numeric IDs [1, 2]",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}", "[1, 2]"),
            "expected_count": 2,
            "expected_titles": ["Test Doc 1", "Test Doc 2"]
        },
        {
            "name": "Doc prefix format [Doc1, Doc2]",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}", "[Doc1, Doc2]"),
            "expected_count": 2,
            "expected_titles": ["Test Doc 1", "Test Doc 2"]
        },
        {
            "name": "Quoted Doc format [\"Doc1\", \"Doc2\"]",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}", '["Doc1", "Doc2"]'),
            "expected_count": 2,
            "expected_titles": ["Test Doc 1", "Test Doc 2"]
        },
        {
            "name": "Mixed formats [1, Doc2, \"Doc3\"]",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}", '[1, Doc2, "Doc3"]'),
            "expected_count": 3,
            "expected_titles": ["Test Doc 1", "Test Doc 2", "Test Doc 3"]
        },
        {
            "name": "Duplicate IDs [1, 1, Doc2]",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}", "[1, 1, Doc2]"),
            "expected_count": 2,
            "expected_titles": ["Test Doc 1", "Test Doc 2"]
        },
        {
            "name": "No important_info tag",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}"),
            "expected_count": 3,
            "expected_titles": ["Test Doc 1", "Test Doc 2", "Test Doc 3"]
        },
        {
            "name": "Empty important_info tag []",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}", "[]"),
            "expected_count": 0,
            "expected_titles": []
        },
        {
            "name": "Invalid IDs [invalid, 1, Doc2]",
            "input": create_test_input(f"{doc1}\n{doc2}\n{doc3}", "[invalid, 1, Doc2]"),
            "expected_count": 2,
            "expected_titles": ["Test Doc 1", "Test Doc 2"]
        },
        {
            "name": "Multiple info blocks",
            "input": f"""
            <information>{doc1}\n{doc2}</information>
            <important_info>[1]</important_info>
            <information>{doc3}</information>
            <important_info>[3]</important_info>
            """,
            "expected_count": 2,
            "expected_titles": ["Test Doc 1", "Test Doc 3"]
        }
    ]

    # Run all test cases
    print("Starting tests...")
    total_tests = len(test_cases)
    passed_tests = 0

    for test_case in test_cases:
        if run_test_case(
            test_case["name"],
            test_case["input"],
            test_case["expected_count"],
            test_case["expected_titles"]
        ):
            passed_tests += 1

    print(f"\nTest Summary: {passed_tests}/{total_tests} tests passed")

if __name__ == '__main__':
    main() 