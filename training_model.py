import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

from preparing_data import train_data

# Features (Users, Articles) and Target (Ratings)
X = train_data[["User", "Article"]].values
y = train_data["Rating"].values

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 42 number set to make each split equal

print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")

#Define input dimensions
num_users = train_data["User"].nunique()
num_articles = train_data["Article"].nunique()

# Input layers
user_input = Input(shape=(1,))
article_input = Input(shape=(1,))

# Embedding layers
user_embedding = Embedding(input_dim=num_users + 1, output_dim=50)(user_input)
article_embedding = Embedding(input_dim=num_articles + 1, output_dim=50)(article_input)

# Flatten embeddings
user_flat = Flatten()(user_embedding)
article_flat = Flatten()(article_embedding)

# Concatenate features
concat_layer = Concatenate()([user_flat, article_flat])

# Fully connected layers
dense1 = Dense(128, activation="relu")(concat_layer)
dense2 = Dense(64, activation="relu")(dense1)
output = Dense(1)(dense2)  # Single output for predicted rating

# Compile the model
model = Model(inputs=[user_input, article_input], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Model summary
model.summary()

# Train the model
history = model.fit(
    [X_train[:, 0], X_train[:, 1]], y_train,  # Separate user and article inputs
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    epochs=20,
    batch_size=32
)

model.save("recommendation_model.keras")
print("Model saved successfully!")












''' Encode users and articles as categorical indices
#train_data["UserID"] = train_data["User"].astype("category")
#train_data["ArticleID"] = train_data["Article"].astype("category")

# Features (Users, Articles) and Target (Ratings)
#X = train_data[["UserID", "ArticleID"]].values
#y = train_data["Rating"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the embedding dimensions
embedding_dim = 8

# Input layers
user_input = keras.Input(shape=(1,), name="User")
article_input = keras.Input(shape=(1,), name="Article")

# Embedding layers
user_embedding = layers.Embedding(input_dim=len(user_tag_ratings), output_dim=embedding_dim)(user_input)
article_embedding = layers.Embedding(input_dim=len(article_tags), output_dim=embedding_dim)(article_input)

# Flatten embeddings
user_vector = layers.Flatten()(user_embedding)
article_vector = layers.Flatten()(article_embedding)

# Concatenate embeddings and pass through Dense layers
concat = layers.Concatenate()([user_vector, article_vector])
dense1 = layers.Dense(16, activation="relu")(concat)
dense2 = layers.Dense(8, activation="relu")(dense1)
output = layers.Dense(1, activation="linear")(dense2)  # Predicts rating

# Build and compile model
model = keras.Model(inputs=[user_input, article_input], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Print model summary
model.summary()'''
