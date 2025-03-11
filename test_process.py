from chatbot import Chatbot

def test_process():
    # Initialize the chatbot in LLM programming mode
    bot = Chatbot(llm_enabled=True)
    
    # Test cases for the process function
    test_cases = [
        # Foreign titles
        "El Cuaderno",
        "Jernmand",
        
        # Emotional statements
        "I am angry at you for your bad recommendations",
        "Ugh that movie was a disaster",
        "Ewww that movie was so gruesome. Stop making stupid recommendations",
        "Wait what? You recommended Titanic?",
        
        # General movie statements
        "I liked \"Titanic\"",
        "I didn't enjoy \"The Matrix\"",
        "I've never seen \"Inception\"",
        
        # Questions
        "What movies are you going to recommend today?",
        "Can you tell me about good horror movies?"
    ]
    
    # Run each test case and print the results
    for input_text in test_cases:
        response = bot.process(input_text)
        print(f"Input: '{input_text}'")
        print(f"Response: '{response}'")
        print("---")

if __name__ == "__main__":
    test_process() 