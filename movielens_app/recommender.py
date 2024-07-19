import os
import pandas as pd
import joblib
from surprise import Dataset, Reader, Prediction
from surprise.model_selection import train_test_split
from collections import defaultdict
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

@st.cache_resource
def load_models_and_data():
    # Set up paths
    data_path = 'C:/source/Orion_Innovation_internship_repos/MovieLens-recommender/Dataset'
    model_path = 'C:/source/Orion_Innovation_internship_repos/MovieLens-recommender/models'

    # Check if files exist
    required_files = ['movie.csv', 'rating.csv', 'link.csv', 'tag.csv', 'genome_scores.csv', 'genome_tags.csv',
                      'svd_model.pkl', 'tfidf_vectorizer.pkl']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_path if f.endswith('.csv') else model_path, f))]
    if (missing_files):
        raise FileNotFoundError(f"Missing files: {missing_files}")

    # Load datasets
    movies = pd.read_csv(os.path.join(data_path, 'movie.csv'))
    ratings = pd.read_csv(os.path.join(data_path, 'rating.csv'))
    links = pd.read_csv(os.path.join(data_path, 'link.csv'))
    tags = pd.read_csv(os.path.join(data_path, 'tag.csv'))
    genome_scores = pd.read_csv(os.path.join(data_path, 'genome_scores.csv'))
    genome_tags = pd.read_csv(os.path.join(data_path, 'genome_tags.csv'))

    # Merge datasets on movieId
    data = pd.merge(ratings, movies, on='movieId')
    data = data[['userId', 'movieId', 'rating']]
    reader = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(data, reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)

    # Load the saved model
    algo = joblib.load(os.path.join(model_path, 'svd_model.pkl'))
    predictions = algo.test(testset)

    def round_rating(rating, scale=(0.5, 5.0), step=0.5):
        return round(min(max(rating, scale[0]), scale[1]) / step) * step

    rounded_predictions = [Prediction(uid, iid, true_r, round_rating(est), details)
                           for uid, iid, true_r, est, details in predictions]

    def get_top_n(predictions, n=10):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        return top_n

    top_n_collaborative = get_top_n(rounded_predictions, n=10)

    # Content-Based Filtering Part
    genome_tags.rename(columns={'tag': 'tag_name'}, inplace=True)
    genome_scores = genome_scores.merge(genome_tags, on='tagId', suffixes=('', '_tag'))
    genome_scores.drop(columns=[col for col in genome_scores.columns if col.endswith('_tag') and col != 'tag_name'], inplace=True)
    movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tag_name', values='relevance').fillna(0)
    movies_with_tags = movies.set_index('movieId').join(movie_tag_matrix).reset_index()
    movies_with_tags['combined_features'] = movies_with_tags.apply(
        lambda row: ' '.join(row['genres'].replace('|', ' ').lower().split()), axis=1
    )
    for tag in movie_tag_matrix.columns:
        movies_with_tags['combined_features'] += ' ' + tag

    # Convert the combined_features to TF-IDF vectors
    tfidf_vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.pkl'))
    tfidf_matrix = tfidf_vectorizer.transform(movies_with_tags['combined_features'])

    # Use TruncatedSVD to reduce dimensionality
    svd = joblib.load(os.path.join(model_path, 'svd_matrix.pkl'))
    svd_matrix = svd.transform(tfidf_matrix)

    # Create a mapping from movieId to index in the SVD matrix
    movie_id_to_index = pd.Series(movies_with_tags.index, index=movies_with_tags['movieId']).to_dict()

    return {
        "movies": movies,
        "ratings": ratings,
        "movies_with_tags": movies_with_tags,
        "svd_matrix": svd_matrix,
        "tfidf_vectorizer": tfidf_vectorizer,
        "movie_id_to_index": movie_id_to_index
    }, top_n_collaborative

def get_user_based_recommendations(user_id, n, data):
    user_ratings = data["ratings"][data["ratings"]["userId"] == user_id]
    if user_ratings.empty:
        return []
    user_movie_indices = data["movies_with_tags"][data["movies_with_tags"]["movieId"].isin(user_ratings["movieId"])].index
    sim_scores = cosine_similarity(data["svd_matrix"][user_movie_indices], data["svd_matrix"]).mean(axis=0)
    top_indices = sim_scores.argsort()[-n:][::-1]
    return data["movies_with_tags"].iloc[top_indices]["title"].tolist()

def get_popular_movies(n, data):
    popular_movies = data["ratings"].groupby("movieId").agg({"rating": "mean", "userId": "count"}).reset_index()
    popular_movies.columns = ["movieId", "avg_rating", "num_ratings"]
    popular_movies = popular_movies[popular_movies["num_ratings"] > 50]  # You can adjust this threshold
    top_movies = popular_movies.sort_values(by="avg_rating", ascending=False).head(n)
    top_movie_titles = data["movies"][data["movies"]["movieId"].isin(top_movies["movieId"])]["title"].tolist()
    return top_movie_titles

def get_user_reviews(user_id):
    # Assuming that 'ratings' DataFrame contains user reviews
    ratings = pd.read_csv('C:/source/Orion_Innovation_internship_repos/MovieLens-recommender/Dataset/rating.csv')
    movies = pd.read_csv('C:/source/Orion_Innovation_internship_repos/MovieLens-recommender/Dataset/movie.csv')

    # Merge ratings with movies to get movie titles
    user_ratings = ratings[ratings['userId'] == user_id].merge(movies, on='movieId')

    # Extract relevant information
    user_reviews = user_ratings[['title', 'rating']].apply(lambda x: f"{x['title']} - Rating: {x['rating']}", axis=1).tolist()

    return user_reviews

# Hybrid recommendation function
def hybrid_recommendation(user_id, top_n_collaborative, n, data):
    collab_recs = top_n_collaborative.get(user_id, [])
    if not collab_recs:
        return get_popular_movies(n, data)
    
    recs = defaultdict(float)
    for movie_id, est_rating in collab_recs:
        recs[movie_id] += est_rating
    
    content_recs = get_user_based_recommendations(user_id, n, data)
    for title in content_recs:
        movie_data = data["movies_with_tags"][data["movies_with_tags"]["title"] == title]
        if not movie_data.empty:
            movie_id = movie_data["movieId"].values[0]
            if movie_id in data["movie_id_to_index"]:
                recs[movie_id] += cosine_similarity(data["svd_matrix"][data["movie_id_to_index"][movie_id]].reshape(1, -1), data["svd_matrix"]).mean()
    
    top_recs = sorted(recs.items(), key=lambda x: x[1], reverse=True)[:n]
    rec_movies = []
    for movie_id, rating in top_recs:
        movie_title = data["movies_with_tags"][data["movies_with_tags"]["movieId"] == movie_id]["title"].values[0]
        rec_movies.append((movie_title, rating))
    
    return rec_movies


