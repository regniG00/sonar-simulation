import torch
import utils

class SoundSimulator:
    def __init__(
        self,
        A=10.0,
        f=20000.0,
        T=1.0,
        c=343.0,
        sigma=0.005,
        samplerate=96000,
        sender_pos=None,
        receiver_pos=None
    ):
        self.A = A
        self.f = f
        self.T = T
        self.n_samples = samplerate * T
        self.dt = 1 / samplerate
        self.c = c
        self.sigma = sigma

        self.sender_pos = sender_pos if sender_pos is not None else torch.tensor([0., 0.])
        self.receiver_pos = receiver_pos if receiver_pos is not None else torch.tensor([
            [-20., 0],
            [-10., 0],
            [10., 0],
            [20., 0]
        ])


    def simulate_echoes(self, object_pos):
        sender_dists = torch.norm(object_pos - self.sender_pos, dim=1)
        receiver_dists = torch.cdist(object_pos, self.receiver_pos)
        total_dists = sender_dists[:, None] + receiver_dists
        arrival_time = total_dists / self.c

        t = torch.arange(self.n_samples).float() * self.dt
        t = t.view(1, 1, -1)
        t_obj = arrival_time.unsqueeze(2)

        envelope = torch.exp(-0.5 * ((t - t_obj) / self.sigma) ** 2)
        carrier = torch.cos(2 * torch.pi * self.f * (t - t_obj))
        pulse = envelope * carrier

        attenuation = 1.0 / (total_dists + 1e-6)**2
        pulse = pulse * attenuation.unsqueeze(2)

        result = self.A * pulse.sum(dim=0)
        noise = torch.randn_like(result)
        result = result + noise * 0.05
        return result

    def predict_position(self, signals, n_points = 2, true_obj=None, plot=False):
        n_inits = 1
        kernel = gaussian_filter1d(kernel_size=31, sigma=5.0)
        filtered_signals = apply_1d_filter(signals.abs(), kernel)

        initial_guess = self.initialize_guess(n_inits, n_points, filtered_signals, kernel)
        #initial_guess = self.grid_search_initialization(n_points, filtered_signals, kernel)
        #initial_guess = torch.tensor([[1., 1.], [-1., 1.]])

        #initial_guess = torch.tensor([[0.4, 0.5], [-0.2, 0.5]])
        #initial_guess = torch.tensor([[0.3, 0.5]])
        # = torch.tensor([[0.5,0.5],[0.55,0.5]])

        reflection_points = torch.nn.Parameter(initial_guess.clone())
        optimizer = torch.optim.Adam([reflection_points], lr=0.01)


        utils.plot_signals(filtered_signals, title="True filtered signals")
        steps = 1000000
        for step in range(steps):
            optimizer.zero_grad()
            pred_signals = self.simulate_echoes(reflection_points)
            filtered_pred = apply_1d_filter(pred_signals.abs(), kernel)

            #loss = 100000 * self.time_to_peak_loss(filtered_pred, filtered_signals, self.dt)
            loss = loss_function(filtered_pred, filtered_signals)


            loss.backward()
            optimizer.step()

            with torch.no_grad():
                reflection_points[:, 0].clamp_(-1, 1)
                reflection_points[:, 1].clamp_(0.1, 1)

            print("Step:", step, " Loss:", loss.item(), "Gradients:", reflection_points.grad, "Predicted pos:",
                  reflection_points)

            if(step % 500 == 0 or step == steps - 1) and plot:
                utils.plot_predictions_scene(self.sender_pos, self.receiver_pos, reflection_points, step, loss, true_objs=true_obj)

                if (step % 10000 == 0 or step == steps - 1) and plot:
                    utils.plot_signals(pred_signals.detach())
                    utils.plot_signals(filtered_pred.detach())





            if loss.abs() < 0.01:
                utils.plot_predictions_scene(self.sender_pos, self.receiver_pos, reflection_points, step, loss,
                                             true_objs=true_obj)
                break

        return reflection_points



    def initialize_guess(self, steps, n_points, filtered_target, kernel):
        best_guess = None
        best_loss = None

        for i in range(steps):
            guess = torch.rand(n_points, 2)
            guess[:, 0] -= 0.5
            guess[:, 0] *= 2
            guess[:, 1] *= 0.9
            guess[:, 1] += 0.1
            guess_signal = self.simulate_echoes(guess)
            filtered_guess_signal = apply_1d_filter(guess_signal.abs(), kernel)
            loss = loss_function(filtered_guess_signal, filtered_target)

            print("Guess ", i, ":", guess, " Loss; ", loss)

            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_guess = guess
        print("Best Guess: ", best_guess, " Best Loss: ", best_loss)

        return best_guess


def cosine_similarity_loss(a, b):
    a = a.view(a.size(0), -1)
    b = b.view(b.size(0), -1)

    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)

    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=1)

    return 1 - cos_sim.mean()

def time_to_peak_loss( pred, target, dt = 1):
    pred_peak = soft_argmax(pred)
    target_peak = soft_argmax(target)
    diff = (pred_peak - target_peak)
    diff = diff * dt
    return ((diff)).pow(2).mean()

def soft_argmax( x, beta=100.0):
    x = torch.nn.functional.softmax(x * beta, dim=1)
    indices = torch.arange(x.shape[1], device=x.device).float()
    return (x * indices).sum(dim=1)

def gaussian_filter1d(kernel_size=51, sigma=25.0):
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)

def apply_1d_filter(signals, kernel):
    signals = signals.unsqueeze(1)
    filtered = torch.nn.functional.conv1d(signals, kernel, padding=kernel.shape[-1] // 2)
    return filtered.squeeze(1)

def loss_function(pred, target):
    loss = 1 * torch.mean((target - pred) ** 2)
    loss += 10 * cosine_similarity_loss(target, pred)
    return loss