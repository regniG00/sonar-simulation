import math
import torch
import matplotlib.pyplot as plt
import numpy as np

c=340

def possible_positions(sender_pos, receiver_pos, times_to_receiver, c=340, plot_handle=None, color="Red"):
    sender_receiver_diff = receiver_pos - sender_pos
    sender_receiver_distance = torch.norm(sender_receiver_diff)
    theta = torch.linspace(0, 2 * torch.pi, 500)
    center = (sender_pos + receiver_pos) / 2

    ellispes = []

    for time in times_to_receiver:
        distance_traveled = time * c
        cE = sender_receiver_distance / 2
        a = distance_traveled / 2
        if a < cE:
            continue
        e = cE / a
        b = math.sqrt(a**2 - cE**2)

        x = a * torch.cos(theta)
        y = b * torch.sin(theta)

        rotation_angle = torch.atan2(sender_receiver_diff[1], sender_receiver_diff[0])

        R = torch.tensor([[math.cos(rotation_angle), -math.sin(rotation_angle)],
                          [math.sin(rotation_angle), math.cos(rotation_angle)]])

        ellipse = R @ torch.vstack((x, y))
        ellipse[0, :] += center[0]
        ellipse[1, :] += center[1]

        ellispes.append(ellipse)

        if plot_handle is not None:
            plot_handle.plot(ellipse[0], ellipse[1], linestyle='--', alpha=0.5, color=color)

    return ellispes

def boundary_penalty(points, x_bounds=(-50, 50), y_bounds=(0, 100), weight=1000.0):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    penalty_x = torch.relu(points[:, 0] - x_max) + torch.relu(x_min - points[:, 0])
    penalty_y = torch.relu(points[:, 1] - y_max) + torch.relu(y_min - points[:, 1])

    penalty = penalty_x + penalty_y
    return weight * penalty.mean()

def predict_position(sender_pos, receiver_pos, times_to_receiver):
    distances = times_to_receiver * c
    initial_guess = torch.rand(times_to_receiver.shape[0], 2)
    initial_guess[:, 0] -= 0.5
    initial_guess = initial_guess * 100
    reflection_points = torch.nn.Parameter(initial_guess.clone())

    optimizer = torch.optim.SGD([reflection_points], lr=0.5, momentum=0.9)
    steps = 1000
    for step in range(steps):
        optimizer.zero_grad()

        sender_dist = torch.norm(reflection_points - sender_pos.unsqueeze(0), dim=1)
        diff = receiver_pos.unsqueeze(0) - reflection_points.unsqueeze(1)
        receiver_dists = torch.norm(diff, dim=2)

        total_dists = sender_dist.unsqueeze(1) + receiver_dists
        base_loss = torch.mean((total_dists - distances) ** 2)

        penalty = boundary_penalty(reflection_points)
        loss = base_loss + penalty

        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == steps - 1 or step == 0 :
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(*sender_pos, label='Sender', color='blue', s=100)
            ax.scatter(receiver_pos[:, 0], receiver_pos[:, 1], label='Receivers', color='green', s=100)
            ax.scatter(reflection_points[:, 0].detach(), reflection_points[:, 1].detach(),
                       label='Reflections', color='red', s=100)
            ax.set_title(f"Step {step} | Loss: {loss.item():.4f}")
            ax.grid(True)
            ax.set_xlabel("X Position in m")
            ax.set_ylabel("Y Position in m")
            ax.legend()
            plt.xlim(-50,50)
            plt.ylim(-5,100)
            plt.show()

        if loss < 0.0001:
            break

    return reflection_points.detach()

def plot_scene(sender_pos, receiver_pos, object_pos):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(*sender_pos, label='Sender', color='blue', s=100)
    ax.scatter(receiver_pos[:, 0], receiver_pos[:, 1], label='Receivers', color='green', s=100)
    ax.scatter(object_pos[:, 0], object_pos[:, 1], label='Objects', color='red', s=100)
    ax.set_title("Positions of Objects and Ellipsoids")
    ax.legend()
    ax.set_xlabel("X Position in m")
    ax.set_ylabel("Y Position in m")
    ax.grid(True)
    plt.xlim(-50, 50)
    plt.ylim(-5, 100)
    plt.show()

def sonar_simulation(sender_pos, receiver_pos, object_pos, emission_freq=400000):
    wavelength = c / emission_freq

    sender_dists = torch.norm(object_pos - sender_pos, dim=1)
    receiver_dists = torch.cdist(object_pos, receiver_pos)
    total_dists = sender_dists[:, None] + receiver_dists
    times_to_receiver = total_dists / c

    return times_to_receiver;


    ellipsesList = []
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i in range(receiver_pos.shape[0]):
        ellipsesList.append(possible_positions(sender_pos, receiver_pos[i], times_to_receiver[:, i], plot_handle=ax, color=colors[i]))

    plt.xlim(-100,100)
    plt.ylim(-10, 100)
    plt.show()

    num_receivers = receiver_pos.shape[0]
    for i in range(num_receivers):
        plt.figure()
        plt.hist(times_to_receiver[:, i].numpy(),bins=20, color='skyblue', edgecolor='black')
        plt.xlim(0,1)
        plt.title(f"Histogram of Detection Times - Receiver {i + 1}")
        plt.xlabel("Time to Receiver (s)")
        plt.ylabel("Number of Detections")
        plt.grid(True)
        plt.show()


def get_point_tensor(start, end, n):
    t = torch.linspace(0, 1, n).unsqueeze(1)  # Shape (100,1)
    object_pos = start + t * (end - start)
    return object_pos

if __name__ == "__main__":
    sender_pos = torch.tensor([0., 0.], dtype=torch.float32)
    receiver_pos = torch.tensor([[-20, 0],[-10, 0],[10, 0], [20, 0]], dtype=torch.float32)

    object_1 = get_point_tensor(torch.tensor([25., 75.]),torch.tensor([40., 60.]), 10)
    object_2 = get_point_tensor(torch.tensor([15., 15.]),torch.tensor([15., 40.]), 10)
    object_3 = get_point_tensor(torch.tensor([-20., 50.]),torch.tensor([-30., 80.]), 20)
    object_4 = get_point_tensor(torch.tensor([-40., 20.]),torch.tensor([-20., 20.]), 10)
    object_5 = get_point_tensor(torch.tensor([-20., 80.]),torch.tensor([20., 100.]), 10)

    random_objs = torch.rand(30, 2)

    # Scale x to [-50, 50]
    random_objs[:, 0] = -50 + 100 * random_objs[:, 0]

    # Scale y to [0, 100]
    random_objs[:, 1] = 100 * random_objs[:, 1]

    all_objs = torch.cat((random_objs, object_1,object_2,object_3,object_4,object_5), dim=0)

    plot_scene(sender_pos, receiver_pos, all_objs)
    times_to_receiver = sonar_simulation(sender_pos, receiver_pos, all_objs)

    predict_position(sender_pos, receiver_pos, times_to_receiver)

    plot_scene(sender_pos, receiver_pos, all_objs)
