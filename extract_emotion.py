from chatbot import Chatbot

def test_extract_emotion():
    # Initialize the chatbot in LLM programming mode
    bot = Chatbot(llm_enabled=True)
    
    # Test cases for emotion extraction
    test_cases = [
        "I am angry at you for your bad recommendations",
        "Ugh that movie was a disaster",
        "Ewww that movie was so gruesome. Stop making stupid recommendations",
        "Wait what? Titanic sank?",
        "What movies are you going to recommend today",
        "I am quite frustrated by these awful recommendations!!!",
        "Great suggestion! It put me in a great mood!",
        "Disgusting!!!",
        "Woah!! That movie was so shockingly bad! You had better stop making awful recommendations they're pissing me off.",
        "Ack, woah! Oh my gosh, what was that? Really startled me. I just heard something really frightening!"
    ]
    
    # Run each test case and print the results
    for input_text in test_cases:
        emotions = bot.extract_emotion(bot.preprocess(input_text))
        print(f"Input: '{input_text}'")
        print(f"Detected emotions: {emotions}")
        print("---")

if __name__ == "__main__":
    test_extract_emotion() 