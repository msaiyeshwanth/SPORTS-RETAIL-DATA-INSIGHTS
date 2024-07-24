# SPORTS-RETAIL-DATA-INSIGHTS


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load your Excel data into a DataFrame
file_path = 'path_to_your_file.xlsx'  # Update this with the path to your file
df = pd.read_excel(file_path)

# Assuming your feature names are in a column named 'features'
# Join all feature names into a single string
text = ' '.join(df['features'].tolist())

# Create and display a sentence cloud
sentence_cloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(sentence_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
