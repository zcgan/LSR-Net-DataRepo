import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, ifft2
from scipy.integrate import trapezoid


def diff_with_finite_diff_2d(f, Lx, Ly):
    Nx, Ny = f.shape
    dx = Lx / Nx 
    dy = Ly / Ny  
    df_dx = np.zeros_like(f)
    df_dy = np.zeros_like(f)


    df_dx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
    df_dx[0, :] = (f[1, :] - f[-1, :]) / (2 * dx)    
    df_dx[-1, :] = (f[0, :] - f[-2, :]) / (2 * dx)


    df_dy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
    df_dy[:, 0] = (f[:, 1] - f[:, -1]) / (2 * dy)   
    df_dy[:, -1] = (f[:, 0] - f[:, -2]) / (2 * dy)

    return df_dx, df_dy

def compute_cahn_hilliard_energy(c, W=2, kappa=8, L=16 * np.pi):
    c = c.squeeze()
    N = c.shape[-1]  
    dx = L / N  
    f_bulk = W * c ** 2 * (1 - c) ** 2


    grad_cx,grad_cy = diff_with_finite_diff_2d(c,L,L)


    f_interf = 0.5 * kappa * (grad_cx ** 2 + grad_cy ** 2)


    f_total = f_bulk + f_interf


    total_energy =trapezoid(trapezoid(f_total, dx=dx, axis=1), dx=dx, axis=0)

    return total_energy

def compute_initial_energy(test_loader, device):

    initial_energy = []
    with torch.no_grad():
        for batch in test_loader:
            initial_data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to(device)
            initial_np = initial_data.cpu().numpy()
            energy_val = compute_cahn_hilliard_energy(initial_np)
            initial_energy.append((energy_val, energy_val))
    return initial_energy


def compute_test_energy(model,test_loader, idx,device):
    test_energy = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to(device)
            target = batch[:, idx, :, :].unsqueeze(1).to(torch.float32).to(device)
            out_test, *_ = model(data)
            energy_pred = compute_cahn_hilliard_energy(out_test.cpu().numpy())
            energy_target = compute_cahn_hilliard_energy(target.cpu().numpy())
            test_energy.append((energy_pred, energy_target))
         

    return test_energy

def compute_infer_energy(model, test_loader, idx, device,out_test_prev=None):
    model.eval()
    infer_energy = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            current_step = idx
            data = out_test_prev[i]  
            target = batch[:, current_step, :, :].unsqueeze(1).to(torch.float32).to(device)


            out_test, *_ = model(data)

            energy_pred = compute_cahn_hilliard_energy(out_test.cpu().numpy())
            energy_target = compute_cahn_hilliard_energy(target.cpu().numpy())
            infer_energy.append((energy_pred, energy_target))

    return infer_energy



def save_all_energy_snapshots(out, x_labels, save_dir, T, prefix):
    os.makedirs(save_dir, exist_ok=True)
    sample_idx = 0
    num_steps = len(out)

    for idx in range(num_steps):
        img = out[idx][sample_idx].squeeze()

        fig, ax = plt.subplots(figsize=(1.2, 1.2), dpi=300)  
        ax.imshow(img, cmap='jet', interpolation='none')  
        ax.set_title(x_labels[idx], fontsize=12)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=0.85, bottom=0)  

        clean_label = x_labels[idx].split('(')[0]  
        filename = f"{prefix}_energy_snapshot_{clean_label}_T{T}s.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()




def plot_energy(Total_energy_mean, T, out, prefix=None, save_path=None):
   
    plt.rcParams['font.family'] = 'Nimbus Roman'

    num_steps = len(Total_energy_mean)

    x_labels = []
    for i in range(num_steps):
        if i == 0:
            x_labels.append("0") 
        elif i == 1:
            x_labels.append(f"T") 
        else:
            x_labels.append(f"{i}T") 

    energy_pred = np.array([step[0] for step in Total_energy_mean])
    energy_target = np.array([step[1] for step in Total_energy_mean])

    font_size = 22
    fig, ax1 = plt.subplots(figsize=(7, 5))  


    ax1.plot(range(num_steps), energy_pred, 'b--o', label="Predicted Energy", linewidth=2, markersize=6)
    ax1.plot(range(num_steps), energy_target, 'r-o', label="Target Energy", linewidth=2, markersize=6)

    ax1.set_xticks(range(num_steps))
    ax1.set_xticklabels(x_labels, rotation=45, fontsize=font_size)
    ax1.set_xlabel("Time", fontsize=font_size)
    ax1.set_ylabel("Free Energy", fontsize=font_size)
    ax1.set_title(f"Cahn Hilliard Energy: T={T}", fontsize=font_size + 2)
    ax1.legend(fontsize=font_size)

    ax1.tick_params(axis='both', labelsize=font_size)

    plt.tight_layout()


    full_save_path = os.path.join(save_path, "Energy_PIC")
    os.makedirs(full_save_path, exist_ok=True)
    filename = f"{prefix}_energy_plot_T{T}s.png"
    plt.savefig(os.path.join(full_save_path, filename), dpi=300)
    plt.close()


    save_all_energy_snapshots(out, x_labels, full_save_path, T, prefix)