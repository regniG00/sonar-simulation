import torch
import utils
from sound_simulator import SoundSimulator





if __name__ == "__main__":
    sender_pos = torch.tensor([0., 0])
    receiver_pos = torch.tensor([[-0.1, -0], [0.1, -0]])
    obj_pos = torch.tensor([[20.,60.],[25.,60.],[30.,60.],[35.,60.],[40.,60.],[45.,60.],[50.,60.]])
    #obj_pos = torch.tensor([[45., 80.],[30., 95.]])

    initial_guess = torch.rand(3, 2)
    initial_guess[:, 0] -= 0.5
    initial_guess[:, 0] *= 2
    initial_guess[:, 1] *= 0.9
    initial_guess[:, 1] += 0.1

    #initial_guess = torch.tensor([[0.3,0.4],[0.35,0.4],[0.4,0.4],[0.45,0.4],[0.25,0.4]])
    #initial_guess = torch.tensor([[0.3,0.4],[0.35,0.4],[0.3,0.45]])
    #initial_guess = torch.tensor([[0.2, 0.7],[0., 0.5]])

    obj_pos = initial_guess

    #obj1 = test_object(torch.tensor([10., 50.]), torch.tensor(0))
    #obj2 = test_object(torch.tensor([-30., 20.]), torch.tensor(0.))

    #obj_pos = torch.cat([obj1, obj2], dim=0)  # shape (6, 2)
    A = 10.
    T = 0.1
    c = 343.0
    sigma = 0.0001
    f = 20000

    simulator = SoundSimulator(A=A, f=f, c=c, sigma=sigma, T=T, sender_pos=sender_pos,
                               receiver_pos=receiver_pos)
    signals = simulator.simulate_echoes(object_pos=obj_pos).detach()
    utils.plot_signals(signals, title="True signals")
    predict_positions = simulator.predict_position(signals, plot=True, true_obj=obj_pos)
    predicted_signals = simulator.simulate_echoes(object_pos=predict_positions).detach()
    utils.plot_signals(signals,title="True signals")
    utils.plot_signals(predicted_signals, title="Predicted signals")


