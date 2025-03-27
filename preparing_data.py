import pandas as pd
import numpy as np

# Load the CSV file
file_path = r"C:\Users\weron\Desktop\praca inzynierska\responses.csv"
df = pd.read_csv(file_path)

#write only part of csv data
df_users_ratings= df.iloc[:, 0:150]

#remove Nan values
df_users_ratings = df_users_ratings.fillna(0)

#change all floats to integers
df_users_ratings = df_users_ratings.map(lambda x: int(x) if isinstance(x, float) else x)


#print("Data frame 1.:",df_users_ratings.head())

#creating 2nd df with example articles data
columns = df_users_ratings.copy().columns
num_rows = 10
num_cols = len(columns) - 1
data = []

for _ in range(num_rows):
    row = np.zeros(num_cols, dtype=int)  # Initialize with all zeros
    ones_indices = np.random.choice(num_cols, size=5, replace=False)  # Choose 5 random positions
    row[ones_indices] = 1  # Set selected positions to 1
    data.append(row)

df_articles = pd.DataFrame(data, columns=columns[1:])  # Exclude the first column from original columns

df_articles.insert(0, columns[0], 0)

#print(df_articles)

# Create all (user, article) pairs
user_article_pairs = []
ratings = []

for user in df_users_ratings.index:
    for article in df_articles.index:
        #write rows from dfs
        user_vector = df_users_ratings.loc[user].values
        article_vector = df_articles.loc[article].values

        #change to int
        user_vector = pd.to_numeric(user_vector, errors='coerce')
        article_vector = pd.to_numeric(article_vector, errors='coerce')

        # Replace NaN values with 0 in the vectors
        user_vector = np.nan_to_num(user_vector, nan=0).astype(int)
        article_vector = np.nan_to_num(article_vector, nan=0).astype(int)
        #print("User vector: ",user_vector)
        #print("Article vector: ",article_vector)
        # Compute similarity (dot product as a simple relevance score)
        relevance_score = np.dot(user_vector, article_vector)
        user_article_pairs.append((user, article))
        ratings.append(relevance_score)

# Convert to DataFrame
train_data = pd.DataFrame(user_article_pairs, columns=["User", "Article"])
train_data["Rating"] = ratings

print(train_data)  # Shows user-article pairs with computed relevance
print("Relevance score:", relevance_score)
train_data.to_csv('trained_data.csv', index=False)  # index=False prevents saving the index









