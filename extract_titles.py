from chatbot import Chatbot

def test_extract_titles():
    # Initialize the chatbot
    bot = Chatbot()
    
    # Test cases for title extraction
    test_cases = [
        'I liked "The Notebook"',
        'You are a great bot!',
        'I enjoyed "Titanic (1997)" and "Scream 2 (1997)"',
        'I didn\'t really like "Titanic (1997)".',
        'I never liked "Titanic (1997)".',
        'I really enjoyed "Titanic (1997)"',
        'I saw "Titanic (1997)".',
    ]
    
    # Run each test case and print the results
    for input_text in test_cases:
        titles = bot.extract_titles(bot.preprocess(input_text))
        print(f"Input: '{input_text}'")
        print(f"Extracted titles: {titles}")
        print("---")

if __name__ == "__main__":
    test_extract_titles() 