# SPORTS-RETAIL-DATA-INSIGHTS


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Load your Excel data into a DataFrame
file_path = 'path_to_your_file.xlsx'  # Update this with the path to your file
df = pd.read_excel(file_path)

# Assuming your feature names are in a column named 'features'
# Replace spaces with underscores in all multi-word phrases
df['features'] = df['features'].apply(lambda x: re.sub(r'\s+', '_', x))

# Join all feature names into a single string
text = ' '.join(df['features'].tolist())

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)

# Function to replace underscores with spaces for display
def underscore_to_space(word):
    return word.replace('_', ' ')

# Custom function to draw the word cloud
def draw_word_cloud(wordcloud):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud.recolor(color_func=lambda *args, **kwargs: (0, 0, 0)), interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Replace underscores with spaces in the final word cloud visualization
wordcloud.words_ = {underscore_to_space(k): v for k, v in wordcloud.words_.items()}

# Display the word cloud
draw_word_cloud(wordcloud)

