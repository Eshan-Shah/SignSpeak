import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SIGN = "Hello"  # Change this to any label in your dataset

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]

# Load data
df = pd.read_csv("data/gestures.csv")
samples = df[df['label'] == SIGN].drop('label', axis=1).astype(float)
mean_pose = samples.mean().values

def is_valid_hand(hand_data):
    return not np.allclose(hand_data, 0)

def plot_hand(hand_data, color, alpha=1.0, lw=1):
    x = hand_data[0::3]
    y = hand_data[1::3]
    for i, j in HAND_CONNECTIONS:
        plt.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=lw, alpha=alpha)
    plt.scatter(x, y, c=color, s=30, alpha=alpha)

# --- Plot all samples first ---
plt.figure(figsize=(7, 7))
for i, row in samples.iterrows():
    data = row.values
    hand1 = data[:63]
    hand2 = data[63:126]

    if is_valid_hand(hand1):
        plot_hand(hand1, color='gray', alpha=0.3)
    if is_valid_hand(hand2):
        plot_hand(hand2, color='gray', alpha=0.3)

# --- Overlay the average pose in blue ---
avg_hand1 = mean_pose[:63]
avg_hand2 = mean_pose[63:126]

if is_valid_hand(avg_hand1):
    plot_hand(avg_hand1, color='blue', alpha=1.0, lw=2)
if is_valid_hand(avg_hand2):
    plot_hand(avg_hand2, color='red', alpha=1.0, lw=2)

# --- Final Touches ---
plt.gca().invert_yaxis()
plt.axis('equal')
plt.grid(True)
plt.title(f"Samples + Average Pose for '{SIGN}'")
plt.legend(['Hand 1 avg (blue)', 'Hand 2 avg (red)'], loc='upper right')
plt.show()
