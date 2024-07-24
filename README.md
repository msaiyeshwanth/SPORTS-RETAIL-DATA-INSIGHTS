# SPORTS-RETAIL-DATA-INSIGHTS


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load your Excel data into a DataFrame
file_path = 'path_to_your_file.xlsx'  # Update this with the path to your file
df = pd.read_excel(file_path)

# Assuming your feature names are in a column named 'features'
# Replace spaces with underscores in all multi-word phrases
df['features'] = df['features'].str.replace(' ', '_')

# Join all feature names into a single string
text = ' '.join(df['features'].tolist())

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)

# Custom function to draw the word cloud with underscores replaced by spaces
def draw_word_cloud(wordcloud):
    # Create a new dictionary with underscores replaced by spaces
    word_freq = {word.replace('_', ' '): freq for word, freq in wordcloud.words_.items()}
    
    # Generate a new word cloud with the modified word frequencies
    new_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(new_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Display the word cloud
draw_word_cloud(wordcloud)
