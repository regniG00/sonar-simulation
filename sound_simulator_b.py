import torch
import utils


# This is just a copy of the other simulator
# i didnt want to put it in the original file
# this file contains the grid/reflection parameter approach

class SoundSimulatorB:
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
        self.samplerate = samplerate
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
        coords = object_pos[:, :2]
        reflection_strength = object_pos[:, 2]

        sender_dists = torch.norm(coords - self.sender_pos, dim=1)
        receiver_dists = torch.cdist(coords, self.receiver_pos)
        total_dists = sender_dists[:, None] + receiver_dists
        arrival_time = total_dists / self.c
        t = torch.arange(self.n_samples, device=coords.device).float() * self.dt
        t = t.view(1, 1, -1)
        t_obj = arrival_time.unsqueeze(2)

        envelope = torch.exp(-0.5 * ((t - t_obj) / self.sigma) ** 2)
        carrier = torch.cos(2 * torch.pi * self.f * (t - t_obj))
        pulse = envelope * carrier

        attenuation = 1.0 / (total_dists + 1e-6) ** 2
        pulse = pulse * attenuation.unsqueeze(2)

        pulse = pulse * reflection_strength.view(-1, 1, 1)

        result = self.A * pulse.sum(dim=0)
        noise = torch.randn_like(result)
        result = result + noise * 0.05
        return result

    def optimize_reflections_on_grid(self, signals, grid_size=20, plot=False, true_obj=None):
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(0.1, 1, grid_size)
        xv, yv = torch.meshgrid(x, y, indexing="xy")
        coords = torch.stack([xv.flatten(), yv.flatten()], dim=1)
        M = coords.shape[0]

        reflection_strength = torch.nn.Parameter(torch.ones(M))
        object_pos = torch.cat([coords, reflection_strength.unsqueeze(1)], dim=1)

        optimizer = torch.optim.Adam([reflection_strength], lr=0.01)

        kernel = self.gaussian_filter1d(kernel_size=31, sigma=5.0)
        filtered_signals = self.apply_1d_filter(signals.abs(), kernel)

        steps = 5000
        for step in range(steps):
            optimizer.zero_grad()

            object_pos = torch.cat([coords, reflection_strength.unsqueeze(1)], dim=1)

            pred = self.simulate_echoes(object_pos)
            filtered_pred = self.apply_1d_filter(pred.abs(), kernel)

            loss = self.loss_function(filtered_pred, filtered_signals)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                reflection_strength.clamp_(0, 1)

            if step % 500 == 0 and plot:
                print(f"Step {step}, Loss {loss.item()}")

            if (step % 500 == 0 or step == steps - 1) and plot:
                utils.plot_predictions_scene_ref(self.sender_pos, self.receiver_pos, object_pos, step, loss,
                                             true_objs=true_obj)
                utils.plot_signals(pred)

        reflection_final = reflection_strength.sigmoid().detach()
        object_pos_final = torch.cat([coords, reflection_final.unsqueeze(1)], dim=1)

        return object_pos_final


    def cosine_similarity_loss(self, a, b):
        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)

        a = torch.nn.functional.normalize(a, dim=1)
        b = torch.nn.functional.normalize(b, dim=1)

        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=1)

        return 1 - cos_sim.mean()

    def time_to_peak_loss(self,  pred, target):
        pred_peak = self.soft_argmax(pred)
        target_peak = self.soft_argmax(target)
        diff = (pred_peak - target_peak)
        diff = diff * self.dt
        return ((diff)).pow(2).mean()

    def soft_argmax(self,  x, beta=100.0):
        x = torch.nn.functional.softmax(x * beta, dim=1)
        indices = torch.arange(x.shape[1], device=x.device).float()
        return (x * indices).sum(dim=1)

    def gaussian_filter1d(self, kernel_size=51, sigma=25.0):
        x = torch.arange(kernel_size) - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma)**2)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1)

    def apply_1d_filter(self, signals, kernel):
        signals = signals.unsqueeze(1)
        filtered = torch.nn.functional.conv1d(signals, kernel, padding=kernel.shape[-1] // 2)
        filtered = filtered.squeeze(1)
        filtered = torch.nn.functional.normalize(filtered, dim=1)
        return filtered

    def loss_function(self, pred, target):
        loss = 10 * torch.mean((target - pred) ** 2)
        loss += 1 * self.cosine_similarity_loss(target, pred)
        return loss