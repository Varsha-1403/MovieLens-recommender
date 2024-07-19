import streamlit as st
import pandas as pd
from recommender import hybrid_recommendation, load_models_and_data
import random

# Set the title of the app
st.title('Movie Recommendation App')

# Load models and data
data, top_n_collaborative = load_models_and_data()

# Sidebar for user input
st.sidebar.header('User Input Features')

# Number of recommendations
num_recommendations = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Pick a random user
user_id = random.choice(data['ratings']['userId'].unique())

# Display random user and their past ratings
st.sidebar.write(f"Random User ID: {user_id}")

user_ratings = data["ratings"][data["ratings"]["userId"] == user_id]
if not user_ratings.empty:
    user_movies = pd.merge(user_ratings, data["movies"], on="movieId")
    st.sidebar.write("User's Past Ratings:")
    st.sidebar.dataframe(user_movies[["title", "rating"]])

# Assume recommendations is returned from hybrid_recommendation(user_id, top_n_collaborative, n, data)
recommendations = hybrid_recommendation(user_id, top_n_collaborative, n, data)

if not recommendations:
    print("No recommendations found.")
else:
    for title, rating, poster_url in recommendations:
        # Process each recommendation tuple
        print(f"Title: {title}, Rating: {rating}, Poster URL: {poster_url}")


# Main content
st.write("""
### Welcome to the Movie Recommendation App!
We've generated recommendations for a random user. Check the sidebar for the user ID and their past ratings.
""")





