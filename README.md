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

# Create and display a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)

# Replace underscores with spaces for display
wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text.replace('_', ' '))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
