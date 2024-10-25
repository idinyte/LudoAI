import matplotlib.pyplot as plt

# Data for the bar chart
groups = ['2 Players', '4 Players']
bar1_values = [67.8, 68.62]
bar2_values = [100-67.8, 31.38]

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
x = range(len(groups))
width = 0.4

# Plot bars for each group
bars1 = ax.bar([pos - 0.5*width for pos in x], bar1_values, width, label='Q-learning')
bars2 = ax.bar([pos + 0.5*width for pos in x], bar2_values, width, label='Deep Q-learning')

# Add labels above each bar
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Set labels, title, and legend
ax.set_ylabel('Winrate')
ax.set_title('Q-learning vs Deep Q-learning')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
