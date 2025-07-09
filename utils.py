import matplotlib.pyplot as plt
import torch


def plot_signals(signals, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.plot(signals.T.detach().numpy())  # Transpose so each signal is a separate line
    plt.xlabel('Sample index')
    plt.ylabel('Signal amplitude')
    plt.title('Simulated Ultrasound Echoes at Receivers')
    plt.legend([f"Receiver {i}" for i in range(signals.shape[0])])
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-0.5, 1)
    plt.show()

def plot_predictions_scene(sender_pos, receiver_pos, object_pos, step, loss, true_objs = None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(*sender_pos, label='Sender', color='blue', s=100)
    ax.scatter(receiver_pos[:, 0], receiver_pos[:, 1], label='Receivers', color='green', s=100)
    ax.scatter(object_pos[:, 0].detach(), object_pos[:, 1].detach(),
               label='Predictions', color='orange', s=100)
    if true_objs is not None:
        ax.scatter(true_objs[:, 0], true_objs[:, 1], label='True Objects', color='red', s=100)
    ax.set_title(f"Step {step} | Loss: {loss.item():.4f}")
    ax.grid(True)
    ax.set_xlabel("X Position in m")
    ax.set_ylabel("Y Position in m")
    ax.legend()
    plt.xlim(-50, 80)
    plt.ylim(-5, 110)
    plt.show()
