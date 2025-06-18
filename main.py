import math
import torch
import matplotlib.pyplot as plt

import utils
from sound_simulator import SoundSimulator

if __name__ == "__main__":
    sender_pos = torch.tensor([0., 0.])
    receiver_pos = torch.tensor([[-20., 0], [-10., 0], [10., 0], [20., 0]])
    obj_pos = torch.tensor([[15.,40.]])

    A = 10.
    T = 1.
    n_samples = 44100
    dt = T / n_samples
    c = 343.0
    sigma = 0.005
    f = 20000

    simulator = SoundSimulator(A=A, f=f, c=c, sigma=sigma, T=T, n_samples=n_samples, sender_pos=sender_pos, receiver_pos=receiver_pos)
    signals = simulator.simulate_echoes(object_pos=obj_pos).detach()
    predict_positions = simulator.predict_position(signals, plot=True, true_obj=obj_pos)
    predicted_signals = simulator.simulate_echoes(object_pos=predict_positions).detach()
    utils.plot_signals(signals)
    utils.plot_signals(predicted_signals)
