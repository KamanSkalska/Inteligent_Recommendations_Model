from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("recommendation_model.keras")
print("Model loaded successfully!")

import numpy as np

# Example: Predict the rating for User 100 and Article 5
user_id = np.array([100])  # Replace with real user ID
article_id = np.array([5])  # Replace with real article ID

predicted_rating = model.predict([user_id, article_id])
print(f"Predicted Rating: {predicted_rating[0][0]:.2f}")

