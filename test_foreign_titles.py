from chatbot import Chatbot

def test_foreign_titles():
    # Initialize the chatbot in LLM programming mode
    bot = Chatbot(llm_enabled=True)
    
    # Test cases for foreign titles
    test_cases = [
        "El Cuaderno",
        "Jernmand",
        "Un Roi à New York",
        "Tote Männer Tragen Kein Plaid"
    ]
    
    # Run each test case and print the results
    for title in test_cases:
        indices = bot.find_movies_by_title(title)
        print(f"Title: '{title}' => Indices: {indices}")
        if indices:
            # Access the full movie title including year from the dataset
            full_title = bot.titles[indices[0]][0]
            print(f"Matching movie: '{full_title}'")
        print("---")

if __name__ == "__main__":
    test_foreign_titles() 