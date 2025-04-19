import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_scores(score_file='score.p', window_size=10, save_path='score_plot.png'):
    """
    Loads scores from a pickle file and plots them, including a rolling average.

    Args:
        score_file (str): Path to the pickle file containing the score list.
        window_size (int): Window size for calculating the rolling average.
        save_path (str): Path to save the generated plot image. If None, only shows the plot.
    """
    if not os.path.exists(score_file):
        print(f"Error: Score file not found at {score_file}")
        return

    try:
        with open(score_file, 'rb') as f:
            # Load the score list (scores are averaged per print_interval)
            scores = pickle.load(f) 
    except Exception as e:
        print(f"Error loading score file {score_file}: {e}")
        return

    if not scores:
        print("Score file is empty.")
        return
        
    print(f"Loaded {len(scores)} score intervals from {score_file}")

    # Calculate intervals (assuming print_interval was 10 in training)
    # You might need to adjust this if your print_interval was different
    # The first score is after 10 epochs, second after 20, etc.
    print_interval_in_training = 10 
    intervals = np.arange(1, len(scores) + 1) * print_interval_in_training

    # Calculate rolling average
    rolling_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    # Adjust intervals for rolling average plot (it starts later)
    rolling_intervals = intervals[window_size - 1:] 

    plt.figure(figsize=(12, 6))
    plt.plot(intervals, scores, label='Avg Score per Interval', alpha=0.6)
    if len(rolling_avg) > 0:
       plt.plot(rolling_intervals, rolling_avg, label=f'Rolling Average (window={window_size})', color='red', linewidth=2)
    
    plt.title('Super Mario DQN Training Progress')
    plt.xlabel(f'Training Epochs (Intervals of {print_interval_in_training})')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training scores for Mario DQN.')
    parser.add_argument('--file', type=str, default='score.p',
                        help='Path to the score pickle file (default: score.p)')
    parser.add_argument('--window', type=int, default=10,
                        help='Window size for rolling average (default: 10)')
    parser.add_argument('--save', type=str, default='score_plot.png',
                        help='Path to save the plot image (default: score_plot.png). Set to "None" to disable saving.')

    args = parser.parse_args()
    
    save_location = args.save if args.save.lower() != 'none' else None
    plot_scores(score_file=args.file, window_size=args.window, save_path=save_location)