import torch
import utils
from sound_simulator import SoundSimulator



if __name__ == "__main__":
    sender_pos = torch.tensor([0., 0.])
    receiver_pos = torch.tensor([[-20., 0], [-10., 0.], [10., 0.], [20., 0]])
    obj_pos = torch.tensor([[20.,60.],[25.,60.],[30.,60.],[35.,60.],[40.,60.],[45.,60.],[50.,60.]])
    #obj_pos = torch.tensor([[40., 70.]])
    A = 50.
    T = 1.
    n_samples = 44100
    dt = T / n_samples
    c = 343.0
    sigma = 0.005
    f = 2000

    simulator = SoundSimulator(A=A, f=f, c=c, sigma=sigma, T=T, n_samples=n_samples, sender_pos=sender_pos,
                               receiver_pos=receiver_pos)
    signals = simulator.simulate_echoes(object_pos=obj_pos).detach()
    utils.plot_signals(signals)
    predict_positions = simulator.predict_position(signals, plot=True, true_obj=obj_pos)
    predicted_signals = simulator.simulate_echoes(object_pos=predict_positions).detach()
    utils.plot_signals(signals)
    utils.plot_signals(predicted_signals)


