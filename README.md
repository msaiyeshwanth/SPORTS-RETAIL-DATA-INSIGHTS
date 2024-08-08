# SPORTS-RETAIL-DATA-INSIGHTS

plt.figure(figsize=(10, 6))

# Normalize for color mapping but store actual values for annotation
normed_df = df.copy()
for i in range(normed_df.shape[0]):
    normed_df.iloc[i, :] = (normed_df.iloc[i, :] - normed_df.iloc[i, :].min()) / (normed_df.iloc[i, :].max() - normed_df.iloc[i, :].min())

# Create the heatmap
sns.heatmap(normed_df, annot=df, cmap='coolwarm', linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Normalized SHAP Value'})

# Customize the plot
plt.title('Actual SHAP Values Heatmap with Separate Row Hues')
plt.xlabel('Performance Periods')
plt.ylabel('Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()
