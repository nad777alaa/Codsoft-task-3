import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = {
    'title': ['The Matrix', 'The Matrix Reloaded', 'The Matrix Revolutions', 'John Wick', 'Inception'],
    'genre': ['Action, Sci-Fi', 'Action, Sci-Fi', 'Action, Sci-Fi', 'Action, Thriller', 'Action, Sci-Fi'],
    'cast': ['Keanu Reeves, Laurence Fishburne', 'Keanu Reeves, Laurence Fishburne', 'Keanu Reeves, Laurence Fishburne', 'Keanu Reeves', 'Leonardo DiCaprio, Joseph Gordon-Levitt']
}

df = pd.DataFrame(data)



df['combined_features'] = df['genre'] + ' ' + df['cast']


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 3 most similar movies
    sim_scores = sim_scores[1:4]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar movies
    return df['title'].iloc[movie_indices]


recommended_movies = get_recommendations('The Matrix')
print(recommended_movies)


