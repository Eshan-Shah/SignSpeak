def visualiseData():   
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
    if "label" not in df.columns:
        print("❌ 'label' column not found.")
        return
    
    samples = df[df['label'] == SIGN].drop('label', axis=1).astype(float)
    if samples.empty:
        print(f"⚠️ No samples found for label '{SIGN}'")
        return
    
    mean_pose = samples.mean().values

    def is_valid_hand(hand_data):
        return not np.allclose(hand_data, 0, atol=1e-6)

    def plot_hand(hand_data, color, alpha=1.0, lw=1, label=None):
        x = hand_data[0::3]
        y = hand_data[1::3]
        for i, j in HAND_CONNECTIONS:
            plt.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=lw, alpha=alpha)
        plt.scatter(x, y, c=color, s=30, alpha=alpha, label=label)

    # Plot
    plt.figure(figsize=(7, 7))

    # Plot all sample poses
    for _, row in samples.iterrows():
        data = row.values
        hand1 = data[:63]
        hand2 = data[63:126]

        if is_valid_hand(hand1):
            plot_hand(hand1, color='gray', alpha=0.3)
        if is_valid_hand(hand2):
            plot_hand(hand2, color='gray', alpha=0.3)

    # Overlay average poses
    avg_hand1 = mean_pose[:63]
    avg_hand2 = mean_pose[63:126]

    if is_valid_hand(avg_hand1):
        plot_hand(avg_hand1, color='blue', alpha=1.0, lw=2, label='Avg Hand 1 (blue)')
    if is_valid_hand(avg_hand2):
        plot_hand(avg_hand2, color='red', alpha=1.0, lw=2, label='Avg Hand 2 (red)')

    # Final touches
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Samples + Average Pose for '{SIGN}'")
    plt.legend(loc='upper right')
    plt.show()

visualiseData()