import torch
import matplotlib.pyplot as plt

import utils

def gauss_temp(t_arrival, n_samples, dt):
    t = torch.arange(n_samples).float() * dt

    gauss = torch.exp(-0.5 * ((t - t_arrival) / 0.1) ** 2)
    carrier = torch.cos(2 * torch.pi * 10 * (t - t_arrival))
    attenuation = 1.0 / (t_arrival + 1e-6)

    pulse = gauss * carrier * attenuation

    return pulse

def cosine_similarity_loss(a, b):
    return (1 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))).mean()

def gaussian_filter1d(kernel_size=51, sigma=5.0):
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)


def apply_filter(signal, kernel):
    signal = signal.view(1, 1, -1)  # reshape to (B, C, L)
    filtered = torch.nn.functional.conv1d(signal, kernel, padding=kernel.shape[-1] // 2)
    return filtered.view(-1)



T = 10.
n_samples = 2000
dt = T / n_samples
gauss1 = gauss_temp(t_arrival=2, n_samples=n_samples, dt=dt)

kernel = gaussian_filter1d(kernel_size=11, sigma=5.0)
filtered_true_gauss = apply_filter(gauss1.abs(), kernel)
initial_time = torch.tensor([3.])
time = torch.nn.Parameter(initial_time)
optimizer = torch.optim.Adam([time], lr=0.1)
steps = 100
for step in range(steps):
    optimizer.zero_grad()
    pred_gauss = gauss_temp(time[0],n_samples, dt)
    filtered_pred_gauss = apply_filter(pred_gauss.abs(), kernel)
    loss = torch.mean((filtered_pred_gauss - filtered_true_gauss)**2)
    loss += 10 * cosine_similarity_loss(filtered_pred_gauss, filtered_true_gauss)
    print("Step:",step,"loss:",loss.item(),"Time:",time.detach()[0])

    loss.backward()
    optimizer.step()

    utils.plot_signals(torch.stack([filtered_true_gauss.abs(),filtered_pred_gauss.abs()],dim=0))

    if loss < 1e-4:
        break


exit()

gauss2 = gauss_temp(t_arrival=2.5, n_samples=n_samples, dt=dt)
gauss3 = gauss_temp(t_arrival=7, n_samples=n_samples, dt=dt)
gauss4 = gauss_temp(t_arrival=1, n_samples=n_samples, dt=dt)
gauss5 = gauss_temp(t_arrival=2.2, n_samples=n_samples, dt=dt)
gauss6 = gauss_temp(t_arrival=1.8, n_samples=n_samples, dt=dt)

utils.plot_signals(torch.stack([gauss1,gauss2,gauss3, gauss4, gauss5, gauss6],dim=0))

mse1_2 = torch.mean((gauss1-gauss2)**2)
mse1_3 = torch.mean((gauss1-gauss3)**2)
mse1_4 = torch.mean((gauss1-gauss4)**2)
mse1_5 = torch.mean((gauss1-gauss5)**2)
mse1_6 = torch.mean((gauss1-gauss6)**2)

fig, ax = plt.subplots()

label=["blue_red","blue_orange","blue_brown","blue_purple","blue_green"]
losses = [mse1_4,mse1_2,mse1_6,mse1_5,mse1_3]

ax.bar(label, losses)
ax.set_title("MSE Loss for blue gaussian")

plt.show()

exit()