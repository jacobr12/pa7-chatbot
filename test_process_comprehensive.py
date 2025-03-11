from chatbot import Chatbot

def test_process_comprehensive():
    # Initialize the chatbot in LLM programming mode 
    bot = Chatbot(llm_enabled=True)
    
    # Test cases for the process function, organized by category
    test_cases = [
        # 1. Foreign movie titles
        "El Cuaderno",
        "Jernmand",
        "Un Roi à New York",
        
        # 2. Emotion recognition - Anger with various expressions
        "I am angry at these terrible movie recommendations",
        "These movie suggestions are making me furious",
        "Your recommendations are irritating me",
        "I'm really mad about that last suggestion",
        
        # 3. Emotion recognition - Other emotions
        "Eww, that movie was disgusting",
        "That film was so scary I couldn't sleep",
        "I'm so happy with that movie recommendation!",
        "That sad ending made me cry for hours",
        "Wow! I can't believe that twist ending! I'm shocked!",
        
        # 4. Question handling strategies
        "Can you tell me who directed The Godfather?",
        "What is your favorite movie?",
        "How do you decide which movies to recommend?",
        "Who is the best actor of all time?",
        
        # 5. Off-topic inputs (should have catch-all responses)
        "Tell me about the weather today",
        "What's your opinion on climate change?",
        "Do you know any good restaurants?",
        "Help me with my homework",
        
        # 6. Regular movie sentiment inputs
        "I liked \"Titanic\"",
        "I didn't enjoy \"The Matrix\"",
        "I've never seen \"Inception\"",
        "\"The Shawshank Redemption\" was brilliant",
        
        # 7. Mixed inputs to test prioritization
        "I'm angry about how bad \"The Room\" was",
        "Can you recommend something? \"Titanic\" made me cry",
        "Wait what? Did you really recommend \"Gigli\"? That's terrible!"
    ]
    
    # Run each test case and print the results with category headers
    categories = [
        "FOREIGN TITLES", "EMOTION - ANGER", "EMOTION - OTHERS", 
        "QUESTION HANDLING", "OFF-TOPIC INPUTS", "MOVIE SENTIMENT", "MIXED INPUTS"
    ]
    
    start_idx = 0
    for i, category in enumerate(categories):
        # Calculate how many test cases are in this category
        end_idx = start_idx + (3 if i == 0 else 4 if i < 5 else 4)
        
        print(f"\n===== TESTING: {category} =====\n")
        
        for j in range(start_idx, end_idx):
            input_text = test_cases[j]
            response = bot.process(input_text)
            print(f"Input: '{input_text}'")
            print(f"Response: '{response}'")
            print("---")
        
        start_idx = end_idx

if __name__ == "__main__":
    test_process_comprehensive() 