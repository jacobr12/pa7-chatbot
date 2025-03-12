from chatbot import Chatbot

def test_extract_emotion():
    # Initialize the chatbot in LLM mode (needed for emotion extraction)
    bot = Chatbot(llm_enabled=True)
    
    # Test cases for emotion extraction
    test_cases = [
        "I am angry at you for your bad recommendations",
        "Ugh that movie was a disaster",
        "Ewww that movie was so gruesome. Stop making stupid recommendations",
        "Wait what? Titanic sank?",
        "What movies are you going to recommend today",
        "I love this movie so much!",
        "That film made me feel so sad",
        "This horror movie really scared me",
        "I'm surprised by how good that movie was!",
        "I'm really frustrated with these recommendations"
    ]
    
    # Run each test case and print the results
    for input_text in test_cases:
        emotions = bot.extract_emotion(bot.preprocess(input_text))
        print(f"Input: '{input_text}'")
        print(f"Detected emotions: {emotions}")
        print("---")

if __name__ == "__main__":
    test_extract_emotion() 