import torch
import matplotlib.pyplot as plt

import utils



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

def gauss_temp(t_arrival, n_samples, dt, f=10, A=1):
    t = torch.arange(n_samples).float() * dt

    gauss = torch.exp(-0.5 * ((t - t_arrival) / 0.1) ** 2)
    carrier = torch.cos(2 * torch.pi * f * (t - t_arrival))
    attenuation = 1.0 / (t_arrival + 1e-6)

    pulse = gauss * carrier *  attenuation
    noise = torch.randn_like(pulse) * 0.001
    #pulse = pulse + noise

    return pulse

fig, ax = plt.subplots(figsize=(10, 10))
speaker = torch.tensor([0.,0.])
receiver1 = torch.tensor([10.,0.])
receiver2 = torch.tensor([5.,0.])
object = torch.tensor([5.,20.])
ax.scatter(speaker[0],speaker[1], label='Sender', color='blue',marker="v", s=100)
ax.scatter(receiver1[0],receiver1[1], label='Receiver 1', marker="s", color='purple', s=100)
ax.scatter(receiver2[0],receiver2[1], label='Receiver 2', marker="s", color='red', s=100)
ax.scatter(object[0],object[1], label='Object', color='orange', s=100)

ax.plot([speaker[0], object[0]], [speaker[1], object[1]], color='gray', linestyle='--', label='Speaker → Object')
ax.plot([object[0], receiver1[0]], [object[1], receiver1[1]], color='purple', linestyle='--', label='Object → Receiver 1')
ax.plot([object[0], receiver2[0]], [object[1], receiver2[1]], color='red', linestyle='--', label='Object → Receiver 2')


ax.grid(True)
ax.legend()
plt.show()

exit()


T = 10.
n_samples = 2000
dt = T / n_samples
gauss1 = gauss_temp(t_arrival=7, n_samples=n_samples, dt=dt)

kernel = gaussian_filter1d(kernel_size=11, sigma=5.0)
filtered_true_gauss = apply_filter(gauss1.abs() * 100000000, kernel)
#filtered_true_gauss[filtered_true_gauss < 0.01] = 0
initial_time = torch.tensor([2.])
time = torch.nn.Parameter(initial_time)
optimizer = torch.optim.Adam([time], lr=0.1)
steps = 500
for step in range(steps):
    optimizer.zero_grad()
    pred_gauss = gauss_temp(time[0],n_samples, dt)
    filtered_pred_gauss = apply_filter(pred_gauss.abs() * 100000000, kernel)
    test = filtered_pred_gauss * filtered_true_gauss
    #filtered_pred_gauss[filtered_pred_gauss < 0.01] = 0
    #loss = torch.mean((filtered_pred_gauss - filtered_true_gauss)**2)
    loss = 10*cosine_similarity_loss(filtered_pred_gauss, filtered_true_gauss)
    print("Step:",step,"loss:",loss.item(),"Time:",time.detach()[0])
    print(filtered_pred_gauss.norm().item(), filtered_true_gauss.norm().item())

    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        utils.plot_signals(torch.stack([filtered_true_gauss.abs(),filtered_pred_gauss.abs()],dim=0))

    if loss < 1e-7:
        break


exit()







T = 1.
n_samples = 2000
f=10
dt = T / n_samples
gauss1 = gauss_temp(t_arrival=0.5, n_samples=n_samples, dt=dt)
t = torch.arange(n_samples).float() * dt

plt.figure(figsize=(10,10))
plt.plot(t,gauss1)
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title(f"Gabor function f={f}")
plt.show()

plt.plot(t,gauss1.abs())
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title(f"Absolute Gabor function f={f}")
plt.show()


sigma= 50.0
kernel = gaussian_filter1d(kernel_size=1001, sigma=50.0)
filtered_true_gauss = apply_filter(gauss1.abs(), kernel)
plt.plot(t,filtered_true_gauss)
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title(f"Filtered Gabor function f={f}")
plt.show()

exit()


gauss2 = gauss_temp(t_arrival=2.5, n_samples=n_samples, dt=dt)
gauss3 = gauss_temp(t_arrival=7, n_samples=n_samples, dt=dt)
gauss4 = gauss_temp(t_arrival=1, n_samples=n_samples, dt=dt)
gauss5 = gauss_temp(t_arrival=2.2, n_samples=n_samples, dt=dt)
gauss6 = gauss_temp(t_arrival=1.8, n_samples=n_samples, dt=dt)

t = torch.arange(n_samples).float() * dt
plt.figure(figsize=(10,10))
plt.plot(t,gauss1)
plt.plot(t,gauss2)
plt.plot(t,gauss3)
plt.plot(t,gauss4)
plt.plot(t,gauss5)
plt.plot(t,gauss6)
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Amplitude")
#plt.title(f"Gabor function f={f}")
plt.show()


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






