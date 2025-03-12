from chatbot import Chatbot

def test_extract_sentiment():
    # Initialize the chatbot
    bot = Chatbot()
    
    # Test cases for sentiment extraction
    test_cases = [
        # Positive sentiment
        'I liked "The Notebook"',
        'I loved "10 Things I Hate About You"',
        'I really enjoyed "Titanic (1997)"',
        
        # Negative sentiment
        'I didn\'t really like "Titanic (1997)".',
        'I never liked "Titanic (1997)".',
        'I hated "Ex Machina"',
        
        # Neutral sentiment
        'I saw "Titanic (1997)".',
        'Have you watched "The Matrix"?',
        
        # Complex sentiment with negation
        'I don\'t think I disliked "The Revenant"',
        'I don\'t hate "The Room", it\'s so bad it\'s good',
        
        # Mixed sentiment (should be positive overall)
        '"Titanic (1997)" started out terrible, but the ending was totally great and I loved it!',
        
        # From the test cases image
        'I didn\'t really like "Titanic (1997)".',   # Should return -1
        'I never liked "Titanic (1997)".',          # Should return -1
        'I really enjoyed "Titanic (1997)"',        # Should return 1
        'I saw "Titanic (1997)".',                 # Should return 0
    ]
    
    # Run each test case and print the results
    for input_text in test_cases:
        sentiment = bot.extract_sentiment(bot.preprocess(input_text))
        print(f"Input: '{input_text}'")
        print(f"Sentiment: {sentiment} ({sentiment_to_text(sentiment)})")
        print("---")

def sentiment_to_text(sentiment):
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

if __name__ == "__main__":
    test_extract_sentiment() 