import torch
import matplotlib.pyplot as plt

import utils

def gauss_temp(t_arrival, n_samples, dt):
    t = torch.arange(n_samples).float() * dt

    gauss = torch.exp(-0.5 * ((t - t_arrival) / 0.1) ** 2)
    carrier = torch.cos(2 * torch.pi * 10 * (t - t_arrival))
    attenuation = 1.0 / (t_arrival + 1e-6)

    pulse = gauss* carrier * attenuation

    return pulse

def cosine_similarity_loss(a, b):
    return 1 - torch.nn.functional.cosine_similarity(a.abs().unsqueeze(0), b.abs().unsqueeze(0))

T = 1.
n_samples = 44100
dt = T / n_samples
gauss1 = gauss_temp(t_arrival=2, n_samples=n_samples, dt=dt)

initial_time = torch.tensor([0.])
time = torch.nn.Parameter(initial_time)
optimizer = torch.optim.Adam([time], lr=0.1)
steps = 100
for step in range(steps):
    optimizer.zero_grad()
    pred_gauss = gauss_temp(time[0],n_samples, dt)
    loss = cosine_similarity_loss(gauss1,pred_gauss)
    print("Step:",step,"loss:",loss.item(),"Time:",time.detach()[0])

    loss.backward()
    optimizer.step()

    utils.plot_signals(torch.stack([gauss1.abs(),pred_gauss.abs()],dim=0))


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