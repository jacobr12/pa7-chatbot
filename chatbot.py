# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
import util
from pydantic import BaseModel, Field
import re
import numpy as np


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        self.user_movies = []
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        if self.llm_enabled:
            # Check if the input might be a foreign title (no quotes)
            movie_indices = self.find_movies_by_title(line)
            if movie_indices:
                # It's a foreign title that directly matched to a movie
                movie_title = self.titles[movie_indices[0]][0]
                return f"I found '{movie_title}' based on your input '{line}'. Would you like to tell me what you thought of it?"
            
            try:
                # Extract emotions from the input
                emotions = self.extract_emotion(self.preprocess(line))
                
                # Define a movie critic persona
                persona_intro = "As the renowned movie critic MovieMaster, "
                
                # Handle arbitrary questions
                if "?" in line:
                    if any(q in line.lower() for q in ["can you", "what is", "how do", "who is"]):
                        return f"{persona_intro}I'd rather discuss films than answer general questions. Let's talk about your movie preferences! Have you seen any good movies lately?"
                
                # Respond to emotions appropriately
                if emotions:
                    if "Anger" in emotions:
                        return f"{persona_intro}Oh! Did I make you angry? I apologize. Let's reset and talk about films you enjoy instead."
                    elif "Disgust" in emotions:
                        return f"{persona_intro}I see that film disgusted you! Some movies can be quite graphic. Would you prefer recommendations for lighter movies?"
                    elif "Fear" in emotions:
                        return f"{persona_intro}Horror films can certainly be frightening! Would you like recommendations for less scary movies?"
                    elif "Happiness" in emotions:
                        return f"{persona_intro}I'm glad you're feeling happy! Would you like more recommendations for uplifting films?"
                    elif "Sadness" in emotions:
                        return f"{persona_intro}I notice you're feeling down. Perhaps a comedy would help lift your spirits? Tell me about some comedies you've enjoyed."
                    elif "Surprise" in emotions:
                        return f"{persona_intro}That was unexpected, wasn't it? Cinema is full of surprises! Tell me more about what surprised you."
                
                # Default movie-focused response if no emotions detected
                return f"{persona_intro}I'm here to discuss cinema and recommend films tailored to your taste. Tell me about a movie you've enjoyed or disliked recently."
            except Exception as e:
                # Handle any errors with emotion extraction gracefully
                print(f"Error in LLM processing: {e}")
                # Check if the input might be a foreign title even if emotion extraction failed
                movie_indices = self.find_movies_by_title(line)
                if movie_indices:
                    movie_title = self.titles[movie_indices[0]][0]
                    return f"I found '{movie_title}' based on your input '{line}'. Would you like to tell me what you thought of it?"
                return f"As the renowned movie critic MovieMaster, I'm here to discuss cinema and recommend films tailored to your taste. Tell me about a movie you've enjoyed or disliked recently."

        # Extract movie title(s)
        titles = self.extract_titles(self.preprocess(line))
        if not titles:
        # 6f: If no movie found, assume off-topic and redirect
            return "As a moviebot assistant, my job is to help you with only your movie-related needs! Anything film-related that you'd like to discuss?"

    # Extract sentiment (6e)
        sentiment = self.extract_sentiment(self.preprocess(line))
        movie_title = titles[0]  # Assume single movie for now

        # Track unique movies for recommendation (6g)
        if movie_title not in self.user_movies:
            self.user_movies.append(movie_title)

        # Response based on sentiment (6e)
        if sentiment > 0:
            response = f"Ok, you liked \"{movie_title}\"! Tell me what you thought of another movie."
        elif sentiment < 0:
            response = f"Sorry you didn't like \"{movie_title}\"! What did you not like about it?"
        else:
            response = f"I can't tell how you felt about \"{movie_title}\". Can you tell me more about your feelings toward it?"

        # Check if 5 movies have been mentioned (6g)
        if len(self.user_movies) >= 5:
            response += " Ok, now that you've shared your opinion on 5/5 films, would you like a recommendation?"

        return response
            

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        poss_titles = []
        potential = r'"(.*?)"'
        matches = re.findall(potential, preprocessed_input)
        for match in matches:
            poss_titles.append(match)
        return poss_titles

    def find_movies_by_title(self, title):
        """Find movies with matching titles.

        Given a movie title, this method should find all movies with the same title
        in the dataset. If the title is in a foreign language, it should translate
        the title to English and then look for matches.

        :param title: a string title of a movie
        :returns: a list of indices of matching movies
        """
        # Dictionary of known foreign titles and their English equivalents
        foreign_to_english = {
            # Spanish titles
            "El Cuaderno": "The Notebook",
            "Doble Felicidad": "Double Happiness",
            
            # Danish titles
            "Jernmand": "Iron Man",
            "Junglebogen": "The Jungle Book",
            
            # French titles
            "Un Roi à New York": "A King in New York",
            
            # German titles
            "Tote Männer Tragen Kein Plaid": "Dead Men Don't Wear Plaid",
            "Der König der Löwen": "The Lion King",
            
            # Italian titles
            "Indiana Jones e il Tempio Maledetto": "Indiana Jones and the Temple of Doom"
        }
        
        search_title = title
        
        # Check if the input is a foreign title we know
        if title in foreign_to_english:
            search_title = foreign_to_english[title]
            print(f"Translated '{title}' to '{search_title}'")
        
        # Search for the title in the database
        matching_indices = []
        for idx, movie_title in enumerate(self.titles):
            # Extract just the title part (without year)
            title_only = re.match(r"(.*?)( \(\d{4}\))?$", movie_title[0])
            if title_only:
                clean_movie_title = title_only.group(1).lower()
                # Try different matching strategies
                if search_title.lower() in clean_movie_title or clean_movie_title in search_title.lower():
                    matching_indices.append(idx)
                # Handle reversed "The" titles (e.g., "Notebook, The" vs "The Notebook")
                elif search_title.lower().startswith("the ") and search_title.lower()[4:] in clean_movie_title:
                    matching_indices.append(idx)
                elif clean_movie_title.startswith("the ") and clean_movie_title[4:] in search_title.lower():
                    matching_indices.append(idx)
                # Handle comma-based "The" titles (e.g., "Notebook, The")
                elif ", the" in clean_movie_title and "the " + clean_movie_title.split(", the")[0] == search_title.lower():
                    matching_indices.append(idx)
        
        return matching_indices

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        remove_titles_input = preprocessed_input
        for title in self.extract_titles(preprocessed_input):
            remove_titles_input = preprocessed_input.replace(f'"{title}"', '')
        tokens = remove_titles_input.lower().split()
        positive = 0
        negative = 0
        #words = [re.sub(r'[^\w\s]', '', token) for token in tokens]
        reverse_words = ["don't", "won't", "neither", "nor", "haven't", "not", "didn't", "never", "no", "couldn't", "wouldn't", "can't", "isn't", "doesn't", "weren't", "wasn't", "shouldn't", "hadn't", "hasn't"]
        i=0
        while i <len(tokens):
            found_sentiment = False
            word = tokens[i]
            if word in self.sentiment:
                sentiment = self.sentiment[word]
                found_sentiment = True
            else:
                suffixes = ['ing', 'ed', 's', 'es', 'ly', "'s", "d"]
                for suffix in suffixes:
                    if word.endswith(suffix) and len(word) >len(suffix):
                        stemmed = word[:-len(suffix)]
                        if stemmed in self.sentiment:
                            sentiment = self.sentiment[stemmed]
                            found_sentiment = True
                            break
            if found_sentiment:
                is_neg = False
                if i > 0 and tokens[i-1] in reverse_words:
                    is_neg = True
                elif i > 1 and tokens[i-2] in reverse_words:
                    is_neg = True
                if is_neg:
                    if sentiment == 'pos':
                        negative +=1
                    elif sentiment =='neg':
                        positive +=1 
                else:
                    if sentiment == 'neg':
                        negative +=1
                    elif sentiment =='pos':
                        positive +=1 
            i+=1
        if positive>negative:
            return 1
        elif negative>positive:
            return -1
        else:
            return 0

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings[ratings > threshold] = 1
        for i in range(ratings.shape[0]):
            for j in range(ratings.shape[1]):
                rating = ratings[i, j]
                if rating != 0 and rating <= threshold:
                    binarized_ratings[i, j] = -1

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = 0
        if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
            return 0
        similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        rated = np.nonzero(user_ratings)[0]
        pred_scores = []
        for i in range(len(user_ratings)):
            if user_ratings[i] ==0:
                pred = 0
                
                for a in rated:
                    similarity = self.similarity(ratings_matrix[i], ratings_matrix[a])
                    pred+= similarity * user_ratings[a]
               
                pred_scores.append((i,pred))
        sorted_predictions= sorted(pred_scores, key=lambda x:(x[1], -x[0]), reverse=True)
        recs = [movie for movie, score in sorted_predictions[:k]]
        return recs



        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. PART 2: LLM Prompting Mode                                            #
    ############################################################################

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. 
        You should:
         - Extract and acknowledge the user's sentiment about a movie they mention.
         -  Keep the conversation focused on movies, redirecting users if they go off-topic.
         - Track the number of movies the user has mentioned and ask if they want a recommendation after 5 unique movies.
    """
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt
    
    ############################################################################
    # 5. PART 3: LLM Programming Mode (also need to modify functions above!)   #
    ############################################################################

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]
        """
        # Define the JSON schema for emotion extraction
        class EmotionExtractor(BaseModel):
            Anger: bool = Field(default=False)
            Disgust: bool = Field(default=False)
            Fear: bool = Field(default=False)
            Happiness: bool = Field(default=False)
            Sadness: bool = Field(default=False)
            Surprise: bool = Field(default=False)
        
        # Create a system prompt for the emotion extraction
        system_prompt = """You are an emotion detection bot for analyzing movie reviews and conversations.
        
        Analyze the text and identify if any of the following emotions are present:
        - Anger: Look for words like angry, frustrated, mad, annoyed, upset
        - Disgust: Look for words like gross, disgusting, eww, ugh, nasty, gruesome
        - Fear: Look for words like scared, afraid, terrifying, frightening
        - Happiness: Look for words like happy, joy, delighted, pleased, glad
        - Sadness: Look for words like sad, depressed, unhappy, gloomy
        - Surprise: Look for words like wow, unexpected, shocked, amazed
        
        Return a JSON object with boolean values for each emotion.
        If multiple emotions are present, mark all of them as true.
        
        IMPORTANT: Only mark an emotion as true if it's clearly expressed in the text. 
        If there's no strong emotional content, return all false values.
        """
        
        # Make the LLM call
        try:
            response = util.json_llm_call(system_prompt, preprocessed_input, EmotionExtractor)
            
            # Convert the response to a list of emotions
            emotions = []
            for emotion, present in response.items():
                if present:
                    emotions.append(emotion)
            
            return emotions
        except Exception as e:
            # Fallback in case of error with the LLM
            print(f"Error in emotion extraction: {e}")
            return []

    ############################################################################
    # 6. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 7. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user."""
        return """
        Welcome to MovieMaster! 
        
        I'm your personal film critic and recommendation specialist. With decades of experience 
        watching thousands of films across all genres, I can help you discover your next favorite movie.
        
        You can tell me about movies you've enjoyed or disliked, and I'll use that information to
        recommend new films tailored to your taste. Simply mention movie titles in quotation marks
        and tell me what you thought of them.
        
        I have a particular expertise in recognizing emotional responses to films, so feel free to
        express how a movie made you feel!
        
        Let's explore the wonderful world of cinema together!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
