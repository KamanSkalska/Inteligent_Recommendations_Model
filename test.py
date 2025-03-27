import pandas as pd
article_tags = pd.DataFrame({
    "friends": [1, 0, 1, 0, 1],
    "family": [1, 1, 0, 1, 0],
    "folk_music": [1, 0, 1, 1, 0],
    "alcohol": [0, 1, 0, 1, 1],
    "smoking": [0, 1, 1, 0, 1],
    "fake": [0, 1, 0, 1, 1]
}, index=["Article1", "Article2", "Article3", "Article4", "Article5"])

print(article_tags)