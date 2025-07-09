import torch
import utils


class SoundSimulator:
    def __init__(
        self,
        A=10.0,
        f=20000.0,
        n_samples=44100,
        T=1.0,
        c=340.0,
        sigma=0.005,
        sender_pos=None,
        receiver_pos=None
    ):
        self.A = A
        self.f = f
        self.n_samples = n_samples
        self.T = T
        self.dt = T / n_samples
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

        attenuation = 1.0 / (total_dists + 1e-6)
        pulse = pulse * attenuation.unsqueeze(2)

        result = self.A * pulse.sum(dim=0)  # sum over objects → [N, T]
        return result

    def predict_position(self, signals, true_obj=None, plot=False):
        initial_guess = torch.rand(4, 2)
        initial_guess[:, 0] *= 0.5
        initial_guess = initial_guess * 100
        #initial_guess = torch.tensor([[20., 70.],[10., 45.]])
        #initial_guess = torch.tensor([[20., 60.], [25., 60.], [30., 60.], [35., 60.], [40., 60.], [45., 60.], [50., 60.]])
        #initial_guess[:,1] -= 2

        reflection_points = torch.nn.Parameter(initial_guess.clone())
        signals_normailized = (signals - signals.mean()) / signals.std()



        #optimizer = torch.optim.SGD([reflection_points], lr=0.1, momentum=0.9)
        optimizer = torch.optim.Adam([reflection_points], lr=0.1)
        kernel = gaussian_filter1d(kernel_size=21, sigma=5.0)
        filtered_signals = apply_1d_filter(signals.abs(), kernel)
        steps = 20000
        for step in range(steps):
            optimizer.zero_grad()

            pred_signals = self.simulate_echoes(reflection_points)
            filtered_pred = apply_1d_filter(pred_signals.abs(), kernel)
            #pred_signals_normalized = (pred_signals - pred_signals.mean()) / pred_signals.std()

            #loss = 100 * self.time_to_peak_loss(pred_signals, signals, self.dt)
            #loss = 100* (torch.mean((signals - pred_signals)**2))
            #loss = 1*torch.mean((signals.abs() - pred_signals.abs()) ** 2)
            #loss *= 10*cosine_similarity_loss(signals, pred_signals)

            loss = 1 * torch.mean((filtered_signals-filtered_pred) ** 2)
            loss += 10 * cosine_similarity_loss(filtered_signals, filtered_pred)


            loss.backward()
            optimizer.step()

            with torch.no_grad():
                reflection_points[:, 0].clamp_(-50, 50)
                reflection_points[:, 1].clamp_(0, 100)

            print("Step:", step, " Loss:", loss.item(), "Gradients:", reflection_points.grad, "Predicted pos:",
                  reflection_points)

            if(step % 500 == 0 or step == steps - 1) and plot:
                utils.plot_predictions_scene(self.sender_pos, self.receiver_pos, reflection_points, step, loss, true_objs=true_obj)
                #utils.plot_signals(pred_signals.detach())
                #utils.plot_signals(filtered_pred.detach())



            if loss.abs() < 0.0001:
                utils.plot_predictions_scene(self.sender_pos, self.receiver_pos, reflection_points, step, loss,
                                             true_objs=true_obj)
                break

        return reflection_points.detach()

    def time_to_peak_loss(self, pred, target, dt):
        pred_peak = self.soft_argmax(pred)
        target_peak = self.soft_argmax(target)
        diff = (pred_peak - target_peak)
        diff = diff  * dt
        return ((diff)).pow(2).mean()

    def soft_argmax(self, x, beta=100.0):
        x = torch.nn.functional.softmax(x * beta, dim=1)
        indices = torch.arange(x.shape[1], device=x.device).float()
        return (x * indices).sum(dim=1)

def cosine_similarity_loss(a, b):
    a = a.view(a.size(0), -1)
    b = b.view(b.size(0), -1)

    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)

    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=1)

    return 1 - cos_sim.mean()

def gaussian_filter1d(kernel_size=51, sigma=5.0):
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)

def apply_1d_filter(signals, kernel):
    signals = signals.unsqueeze(1)
    filtered = torch.nn.functional.conv1d(signals, kernel, padding=kernel.shape[-1] // 2)
    return filtered.squeeze(1)


