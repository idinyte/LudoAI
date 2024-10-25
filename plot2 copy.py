import matplotlib.pyplot as plt

# Data for the bar chart
groups = ['2 Players', '3 Players', '4 Players']
bar1_values = [77.3, 61.71, 49.77]
bar2_values = [77.48, 65.24, 56.47]
bar3_values = [75.79, 59.93, 52.15]
bar4_values = [85.97, 74.3, 66.21]
bar5_values = [84.94, 74.25, 65.37]

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
x = range(len(groups))
width = 0.15

# Plot bars for each group
bars1 = ax.bar([pos - 2*width for pos in x], bar1_values, width, label='DQL 2 Hidden Layers')
bars2 = ax.bar([pos - 1*width for pos in x], bar2_values, width, label='DQL 4 Hidden Layers')
bars3 = ax.bar([pos + 0*width for pos in x], bar3_values, width, label='DQL 8 Hidden Layers')
bars4 = ax.bar([pos + 1*width for pos in x], bar4_values, width, label='Q-learning')
bars5 = ax.bar([pos + 2*width for pos in x], bar5_values, width, label='No training, highest immediate reward')

# Add labels above each bar
for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Set labels, title, and legend
ax.set_ylabel('Winrate')
ax.set_title('Deep Neural Network parameter tunning. Testing on 10000 games.')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend(loc='lower right')

# Show the plot
plt.tight_layout()
plt.show()
