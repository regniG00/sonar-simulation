import torch
import utils
from sound_simulator import SoundSimulator
import logging

from sound_simulator_b import SoundSimulatorB

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # Just the message, no timestamp/level
        handlers=[
            logging.FileHandler("output_normal.log", mode="w"),  # log file
            logging.StreamHandler()  # keep printing to console
        ]
    )


    sender_pos = torch.tensor([0., 0])
    receiver_pos = torch.tensor([[0.25, -0],[-0.25, -0]])
    #receiver_pos = torch.tensor([[-1., -0], [1., -0]])

    initial_guess = torch.rand(2, 2)
    #initial_guess[:, 0] -= 0.5
    #initial_guess[:, 0] *= 2
    initial_guess[:, 1] *= 0.9
    initial_guess[:, 1] += 0.1

    initial_guesses = torch.tensor([
        [0.0, 0.5],
        [0.5, 0.5],
        [-0.9, 0.9],
        [1.0, 1.0],
        [0.9, 0.9],
        [1.0, 0.1],
        [-1.0, 0.1],
        [-0.5, 0.5],
    ])

    initial_guesses = torch.tensor([
        [[0.4, 0.6],
        [-0.4, 0.6],]
    ])

    # Parameters
    A = 100.
    T = 0.015  # time window not period
    c = 343.0
    sigma = 0.0001
    f = 30000
    samplerate = 960000
    runs = 100

    simulator = SoundSimulator(A=A, f=f, c=c, sigma=sigma, T=T,
                               samplerate=samplerate,
                               sender_pos=sender_pos, receiver_pos=receiver_pos)

    simulatorB = SoundSimulatorB(A=A, f=f, c=c, sigma=sigma, T=T,
                                 samplerate=samplerate,
                                 sender_pos=sender_pos, receiver_pos=receiver_pos)
    steps_tensor = torch.zeros(runs)

    for i in range(runs):
        initial_guess = torch.rand(2, 2)
        #initial_guess[:, 0] -= 0.5
        #initial_guess[:, 0] *= 2
        initial_guess[:, 1] *= 0.9
        initial_guess[:, 1] += 0.1
        obj_pos = initial_guess

        signals = simulator.simulate_echoes(object_pos=obj_pos).detach()
        utils.plot_signals(signals, title="True signals")

        #simulatorB.optimize_reflections_on_grid(signals, plot=True, true_obj=obj_pos)

        predict_positions, s = simulator.predict_position(
            signals,
            plot=True,
            true_obj=obj_pos,
            n_points=obj_pos.shape[0]
        )



        #steps_tensor[i] = s
        #logging.info(f"Run {i} Steps: {s} Loss: {loss}")

    logging.info(f"  Mean steps: {steps_tensor.mean().item():.2f}")
    logging.info(f"  Max steps: {steps_tensor.max().item()}")
    logging.info(f"  Min steps: {steps_tensor.min().item()}")

    exit()
    # Run for each initial guess
    for guess_idx, guess in enumerate(initial_guesses):
        #obj_pos = guess.unsqueeze(0)  # make it shape (1,2) like before
        obj_pos = guess
        steps_tensor = torch.zeros(runs, dtype=torch.float32)

        logging.info(f"\n=== Running with initial guess {guess_idx + 1}: {guess.tolist()} ===")

        for i in range(runs):
            initial_guess = torch.rand(2, 2)
            initial_guess[:, 0] -= 0.5
            initial_guess[:, 0] *= 2
            initial_guess[:, 1] *= 0.9
            initial_guess[:, 1] += 0.1
            obj_pos = initial_guess

            obj_pos = torch.tensor([[-0.9, 0.9]])


            signals = simulator.simulate_echoes(object_pos=obj_pos).detach()
            utils.plot_signals(signals, title="True signals")

            #simulatorB.optimize_reflections_on_grid(signals, plot=True, true_obj=obj_pos)

            predict_positions, s = simulator.predict_position(
                signals,
                plot=True,
                true_obj=obj_pos,
                n_points=obj_pos.shape[0]
            )
            predicted_signals = simulator.simulate_echoes(object_pos=predict_positions).detach()

            steps_tensor[i] = s
            logging.info(f"Run {i} Steps: {s}")

        # Summary stats for this guess
        logging.info(f"Summary for guess {guess.tolist()}:")
        logging.info(f"  Mean steps: {steps_tensor.mean().item():.2f}")
        logging.info(f"  Max steps: {steps_tensor.max().item()}")
        logging.info(f"  Min steps: {steps_tensor.min().item()}")



