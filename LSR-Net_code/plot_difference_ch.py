import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_test(model, save_path,test_loader,layer,idx,prefix='test',sample_idx=[0, 10, 20, 30]):
    model_path = os.path.join(save_path, 'model_best_weight.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    with torch.no_grad():
        for i, batch  in enumerate(test_loader):
            data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to('cuda')
            target = batch[:, idx, :, :].unsqueeze(1).to(torch.float32).to('cuda')
            out_test, result_list_All, mid_list, fourier, amplitudes, shifts, betas, _, _ = model(
                data) 
            mid_list[-1] = [target]
            if i% 10 == 0:
                plot_diff(mid_list,save_path,layer,i,idx,prefix=prefix)
            if i in sample_idx:
                plot_single_diff(mid_list, save_path, layer, sample_idx=i, step_idx=idx, prefix=prefix)
                # plot_slice(mid_list, save_path, layer, sample_idx=i, step_idx=idx, prefix=prefix)

def plot_infer(model, save_path, test_loader, layer, idx, out_test_prev=None,sample_idx=[0, 10, 20, 30]):
    model_path = os.path.join(save_path, 'model_best_weight.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 选择当前步和前一步
            current_step = idx
            data = out_test_prev[i]  # 其余轮使用前一轮推理的结果

            target = batch[:, current_step, :, :].unsqueeze(1).to(torch.float32).to('cuda')

            # 进行推理
            out_test, result_list_All, mid_list, _, _, _, _, _, _ = model(data)
            mid_list[-1] = [target]

            if i % 10 == 0:
                plot_diff(mid_list, save_path, layer, i, idx, prefix=f'step_{current_step}')
            if i in sample_idx:
                plot_single_diff(mid_list, save_path, layer, sample_idx=i, step_idx=current_step, prefix=f'step_{current_step}')
               




def plot_diff(mid_outputs, save_path, layer, i, idx, prefix='inference'):
    difference_path = os.path.join(save_path, 'color_difference')
    os.makedirs(difference_path, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    input_data = mid_outputs[0][0].squeeze(0).squeeze(0).cpu().numpy()
    target = mid_outputs[-1][0].squeeze(0).squeeze(0).cpu().numpy()

    if layer >= len(mid_outputs):
        print(f"Warning: layer index {layer} out of range for mid_outputs with length {len(mid_outputs)}")
        layer = len(mid_outputs) - 1

    out = mid_outputs[layer][2].squeeze(0).squeeze(0).cpu().numpy()


    common_vmin = min(input_data.min(), target.min(), out.min())
    common_vmax = max(input_data.max(), target.max(), out.max())
    error_vmin = 0.0
    error_vmax = np.max(np.abs(out - target))


    font_size = 16


    im0 = axes[0].imshow(input_data, cmap='jet', vmin=common_vmin, vmax=common_vmax)
    axes[0].set_title('Initial Condition', fontsize=font_size)

    im1 = axes[1].imshow(target, cmap='jet', vmin=common_vmin, vmax=common_vmax)
    axes[1].set_title('Ground Truth', fontsize=font_size)

    im2 = axes[2].imshow(out, cmap='jet', vmin=common_vmin, vmax=common_vmax)
    axes[2].set_title('Prediction', fontsize=font_size)

    absolute_error = np.abs(out - target)
    im3 = axes[3].imshow(absolute_error, cmap='jet', vmin=error_vmin, vmax=error_vmax)
    axes[3].set_title('Absolute Error', fontsize=font_size)


    for ax, im in zip(axes, [im0, im1, im2, im3]):
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=font_size) 

    for ax in axes:
        ax.tick_params(axis='both', labelsize=font_size)

    plt.tight_layout()
    filename = f"{prefix}_{idx}_{i}_{layer}_diff.png"
    plt.savefig(os.path.join(difference_path, filename), dpi=300)
    plt.close()







def plot_single_diff(mid_outputs, save_path, layer, sample_idx, step_idx, prefix='inference'):
    plt.rcParams['font.family'] = 'Nimbus Roman'


    single_difference_path = os.path.join(save_path, 'single_difference_new', f'idx={sample_idx}')
    os.makedirs(single_difference_path, exist_ok=True)


    input_data = mid_outputs[0][0].squeeze(0).squeeze(0).cpu().numpy()
    target = mid_outputs[-1][0].squeeze(0).squeeze(0).cpu().numpy()
    if layer >= len(mid_outputs):
        print(f"Warning: layer index {layer} out of range")
        layer = len(mid_outputs) - 1
    out = mid_outputs[layer][2].squeeze(0).squeeze(0).cpu().numpy()

 
    common_vmin = min(input_data.min(), target.min(), out.min())
    common_vmax = max(input_data.max(), target.max(), out.max())
    error_vmin = 0.0
    error_vmax = np.max(np.abs(out - target))
    font_size = 25

    def save_image(data, title, filename, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
        # ax.set_title(title, fontsize=font_size)
        ax.tick_params(axis='both', labelsize=font_size)

      
        ax.set_xticks([])  
        ax.set_yticks([]) 


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.4)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=font_size)

        plt.tight_layout()
        plt.savefig(os.path.join(single_difference_path, filename), dpi=300)
        plt.close()


    save_image(input_data, 'Initial Condition', f'{prefix}_{sample_idx}_{step_idx}_initial_condition.png')
    save_image(target, 'Ground Truth', f'{prefix}_{sample_idx}_{step_idx}_ground_truth.png', common_vmin, common_vmax)
    save_image(out, 'Prediction', f'{prefix}_{sample_idx}_{step_idx}_prediction.png', common_vmin, common_vmax)
    save_image(np.abs(out - target), 'Absolute Error', f'{prefix}_{sample_idx}_{step_idx}_absolute_error.png', error_vmin, error_vmax)


def plot_slice(mid_outputs, save_path, layer, sample_idx, step_idx, x_coords=[0, 50, 100], prefix='inference'):
    slice_dir = os.path.join(save_path, 'slice_difference', f'idx={sample_idx}')
    os.makedirs(slice_dir, exist_ok=True)

    font_size = 25

    input_tensor = mid_outputs[0][0].squeeze(0).squeeze(0).cpu().numpy()
    pred_tensor = mid_outputs[layer][2].squeeze(0).squeeze(0).cpu().numpy()
    target_tensor = mid_outputs[-1][0].squeeze(0).squeeze(0).cpu().numpy()

    H, W = input_tensor.shape
    y_axis = np.linspace(0, 1, H)


    colors = cm.viridis(np.linspace(0, 1, len(x_coords)))

    def plot_single_slice(data_tensor, title, filename):
        plt.figure(figsize=(10, 5))
        for i, x in enumerate(x_coords):
            plt.plot(y_axis, data_tensor[:, x], label=f'x = {x}', color=colors[i], linewidth=2)

        plt.xlabel('y', fontsize=font_size)
        plt.ylabel('u', fontsize=font_size)
        plt.title(title, fontsize=font_size)
        plt.grid(False)
        plt.legend(loc='upper right', fontsize=font_size - 4) 
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(slice_dir, filename), dpi=300)
        plt.close()

    plot_single_slice(input_tensor, 'Initial Condition', f'{prefix}_{sample_idx}_{step_idx}_initial_slices.png')
    plot_single_slice(target_tensor, 'Ground Truth', f'{prefix}_{sample_idx}_{step_idx}_target_slices.png')
    plot_single_slice(pred_tensor, 'Prediction', f'{prefix}_{sample_idx}_{step_idx}_prediction_slices.png')
