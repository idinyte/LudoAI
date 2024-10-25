import pandas as pd
import matplotlib.pyplot as plt

# Read the data from winslist.txt
with open('DeepQlearning/pretrained/20240422102011_4p_4hl/loss_list.txt', 'r') as file:
    data = file.read().splitlines()

# Convert data to a list of integers
wins = list(map(float, data))

# Create a pandas Series
wins_series = pd.Series(wins)

# Calculate the moving average
window_size = 1000  # You can change the window size as needed
moving_average = wins_series.rolling(window=window_size).mean()

with open('DeepQlearning/pretrained/20240422102011_4p_4hl/epsilon_list.txt', 'r') as file:
    epsilon_list = file.read().splitlines()
epsilon_list = list(map(float, epsilon_list))

# Calculate the win percentage
win_percentage = moving_average


plt.figure(figsize=(10, 5))
plt.plot(win_percentage, color='b')
# plt.plot(range(len(epsilon_list)), epsilon_list, label='Epsilon', color='r')
plt.xlabel('Episode')
plt.title('DQL smooth l1 loss')
plt.legend()
plt.grid(True)
plt.show()
