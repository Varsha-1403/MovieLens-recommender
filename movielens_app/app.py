import streamlit as st
from recommender import hybrid_recommendation, load_models_and_data, get_user_reviews
import random

# Set the title of the app
st.title('Movie Recommendation App')

# Main content
st.write("""
### Welcome to the Movie Recommendation App!
Enter the number of recommendations you want in the sidebar, then click 'Generate Recommendations' to see your personalized movie recommendations.
""")

# Sidebar for user input
st.sidebar.header('User Input Features')

# Generate random user ID
user_id = random.randint(1, 1000)  # Assuming user IDs are in the range 1 to 1000
st.sidebar.write(f"Your randomly assigned user ID is: {user_id}")

# Display user's past reviews
user_reviews = get_user_reviews(user_id)
if user_reviews:
    st.sidebar.write("### Your Past Reviews")
    for review in user_reviews:
        st.sidebar.write(f"- {review}")
else:
    st.sidebar.write("No past reviews found for this user.")

# Number of recommendations
num_recommendations = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Load models and data
data, top_n_collaborative = load_models_and_data()

# Button to generate recommendations
if st.sidebar.button("Generate Recommendations"):
    recommendations = hybrid_recommendation(user_id, top_n_collaborative, num_recommendations, data)
    st.write("### Recommendations")
    st.dataframe(recommendations)

