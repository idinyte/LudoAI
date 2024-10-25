import matplotlib.pyplot as plt

# Data for the bar chart
groups = ['lr=0.001, \n df=0.4', 'lr=0.01, \n df=0.4', 'lr=0.1, \n df=0.4', 'lr=0.001, \n df=0.8', 'lr=0.01, \n df=0.8', 'lr=0.1, \n df=0.8', 
          'lr=0.001, \n df=0.9', 'lr=0.01, \n df=0.9', 'lr=0.1, \n df=0.9', 'lr=0.001, \n df=0.95', 'lr=0.01, \n df=0.95', 'lr=0.1, \n df=0.95']
bar1_values = [85.4, 83.2, 84.8, 86.0, 85.1, 86.3, 85.5, 85.4, 84.1, 86.2, 84.0, 86.5]
bar2_values = [72.7, 74.7, 72.6, 72.5, 75.5, 77.1, 73.8, 72.1, 73.7, 73.4, 72.8, 76.5]
bar3_values = [65.4, 62.6, 66.7, 63.4, 67.3, 67.5, 65.9, 65.4, 66.4, 66.4, 65.7, 70.8]

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
x = range(len(groups))
width = 0.25

# Plot bars for each group
bars1 = ax.bar([pos - width for pos in x], bar1_values, width, label='2 Players')
bars2 = ax.bar(x, bar2_values, width, label='3 Players')
bars3 = ax.bar([pos + width for pos in x], bar3_values, width, label='4 Players')

# Add labels above each bar
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Set labels, title, and legend
ax.set_ylabel('Winrate')
ax.set_title('Parameter tuning. Winrate 1000 games.')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend(loc='lower right')

# Show the plot
plt.tight_layout()
plt.show()
