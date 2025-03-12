from chatbot import Chatbot
import numpy as np

def test_recommend():
    # Initialize the chatbot
    bot = Chatbot()
    
    # Create a user ratings vector where:
    # 1 = liked movie
    # -1 = disliked movie
    # 0 = not rated
    user_ratings = np.zeros(len(bot.titles))
    
    # Add some positive ratings for well-known movies
    # Choose some popular movies by finding their indices
    movie_titles_to_rate = {
        "Titanic (1997)": 1,              # Positive rating
        "The Shawshank Redemption": 1,    # Positive rating
        "The Godfather": 1,               # Positive rating
        "The Matrix": 1,                  # Positive rating
        "Star Wars": 1,                   # Positive rating
        "Pulp Fiction": -1,               # Negative rating
        "The Room": -1                    # Negative rating
    }
    
    # Find indices for the movies and set ratings
    print("Setting up test ratings:")
    for title, rating in movie_titles_to_rate.items():
        # Find movies with similar titles
        indices = []
        for i, movie in enumerate(bot.titles):
            if title.lower() in movie[0].lower():
                indices.append(i)
        
        if indices:
            # Use the first matching movie
            idx = indices[0]
            user_ratings[idx] = rating
            print(f"Rated '{bot.titles[idx][0]}' as {rating}")
        else:
            print(f"Could not find movie: {title}")
    
    # Get recommendations (top 10)
    k = 10
    recommendations = bot.recommend(user_ratings, bot.ratings, k)
    
    # Display the recommendations
    print(f"\nTop {k} movie recommendations:")
    if recommendations:
        for i, movie_idx in enumerate(recommendations):
            print(f"{i+1}. {bot.titles[movie_idx][0]}")
    else:
        print("No recommendations found.")

if __name__ == "__main__":
    test_recommend() 