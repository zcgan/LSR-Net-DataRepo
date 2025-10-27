
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional.regression import relative_squared_error
import torch
import pandas as pd
import os
from datetime import datetime
import torch.nn as nn
import csv


class LR_Trainer():
    def __init__(self, train_loader, val_loader, test_loader, model, optimizer, scheduler, loss, epoch, device, path,
                 resume_path, layer,resume=False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = loss
        self.device = device
        self.scheduler = scheduler
        self.epoch = epoch
        self.lowest_loss = 100
        self.save_path = path
        self.train_result = pd.DataFrame(columns=['Epoch', 'train loss', 'val loss'])
        self.Four_param = pd.DataFrame(columns=['Phase', 'Epoch', 'Layer', 'Amplitude', 'Shift', 'Beta','A_par'])
        self.start_epoch = 0
        self.layer = layer

        # 用于累积 multipliers
        self.train_multipliers = []
        self.val_multipliers = []

        if resume:
            checkpoint = torch.load(os.path.join(resume_path, 'model_best_weight.pth'))
            self.model.load_state_dict(checkpoint['model_state_dict'])



    def train(self, writer=None):
        since = datetime.utcnow()

        # for epoch in range(self.start_epoch, self.epoch):
        for epoch in range(self.start_epoch, self.epoch):
            # 训练阶段
            losses, train_amplitudes, train_shifts, train_betas,train_multipliers,train_apar = self.train_epoch(epoch)
            # Store train multipliers for this epoch
            self.train_multipliers.append(train_multipliers)

            # 验证阶段
            val_loss, val_amplitudes, val_shifts, val_betas, val_multipliers, val_apar = self.val_epoch(epoch)
            # Store val multipliers for this epoch
            self.val_multipliers.append(val_multipliers)

            if writer:
                writer.add_scalar('Loss/train', losses, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)

            temp_df = pd.DataFrame({'Epoch': epoch + 1, 'train loss': losses, 'val loss': val_loss}, index=[0])
            self.train_result = pd.concat([self.train_result, temp_df], ignore_index=True)


            for layer_idx in range(len(train_amplitudes)):
                amplitude_list = train_amplitudes[
                    layer_idx].detach().cpu().numpy().tolist()  # Detach and convert to list
                shift_list = train_shifts[layer_idx].detach().cpu().numpy().tolist()
                beta_list = train_betas[layer_idx].detach().cpu().numpy().tolist()
                apar_list = train_apar[layer_idx].detach().cpu().numpy().tolist()

                val_amplitudes_list = val_amplitudes[layer_idx].detach().cpu().numpy().tolist()
                val_shifts_list = val_shifts[layer_idx].detach().cpu().numpy().tolist()
                val_betas_list = val_betas[layer_idx].detach().cpu().numpy().tolist()
                val_apar_list = val_apar[layer_idx].detach().cpu().numpy().tolist()



                self.Four_param = pd.concat([
                    self.Four_param,
                    pd.DataFrame({
                        'Phase': ['train', 'val'],
                        'Epoch': [epoch + 1, epoch + 1],
                        'Layer': [layer_idx, layer_idx],
                        'Amplitude': [amplitude_list, val_amplitudes_list],
                        'Shift': [shift_list, val_shifts_list],
                        'Beta': [beta_list, val_betas_list],
                        'A_par':[apar_list,val_apar_list]
                    })
                ], ignore_index=True)

            self.scheduler.step()
            print("\n[epoch %d] Loss_val: %.4f" % (epoch + 1, val_loss))

            is_best = val_loss < self.lowest_loss
            if is_best:
                self.lowest_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'loss': val_loss,
                }, os.path.join(self.save_path, 'model_best_weight.pth'))
                print('lowest_loss %f' % (self.lowest_loss))



            # 覆盖保存 self.Three_param，而不是追加
            self.Four_param.to_csv(os.path.join(self.save_path, 'Four_param.csv'), index=False)

            # 覆盖保存 train_result
            self.train_result.to_csv(os.path.join(self.save_path, 'train_result.csv'), index=False)

        end = datetime.utcnow()
        time_len = (end - since).total_seconds()  # second

        # 训练结束后保存所有 multipliers
        self.save_multipliers()

        return self.lowest_loss, time_len

    def save_multipliers(self):
        # Convert accumulated multipliers to numpy arrays for saving
        # Move each tensor to CPU before converting to numpy
        train_multipliers_np = np.array([[layer_multipliers.cpu().numpy() for layer_multipliers in epoch_multipliers]
                                         for epoch_multipliers in self.train_multipliers], dtype=object)
        val_multipliers_np = np.array([[layer_multipliers.cpu().numpy() for layer_multipliers in epoch_multipliers]
                                       for epoch_multipliers in self.val_multipliers], dtype=object)

        # Define the file paths for saving the multipliers as .npy files
        train_multipliers_path = os.path.join(self.save_path, 'train_multipliers.npy')
        val_multipliers_path = os.path.join(self.save_path, 'val_multipliers.npy')

        # Save the multipliers to .npy files
        np.save(train_multipliers_path, train_multipliers_np, allow_pickle=True)
        np.save(val_multipliers_path, val_multipliers_np, allow_pickle=True)

        print(f"Saved train multipliers to {train_multipliers_path}")
        print(f"Saved validation multipliers to {val_multipliers_path}")

    def train_epoch(self, epoch):
        losses = 0
        all_amplitudes = []
        all_shifts = []
        all_betas = []
        all_a = []
        batch_multipliers =[]

        for i, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()

            data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to(self.device)
            target = batch[:, -1, :, :].unsqueeze(1).to(torch.float32).to(self.device)

            # 前向传播，获取out_train和amplitudes, shifts
            out_train, seq, _, _, amplitudes, shifts, betas, multipliers,a_par = self.model(data)


            loss = self.criterion(out_train, target) / (self.criterion(target, 0 * target) + 1e-8)
            loss.backward()



            self.optimizer.step()
            losses += loss.item()


            # 收集当前 batch 的 amplitudes 和 shifts
            all_amplitudes.append(amplitudes)
            all_shifts.append(shifts)
            all_betas.append(betas)
            all_a.append(a_par)
            # 收集当前 batch 的 multipliers
            batch_multipliers.append(multipliers)

            print("epoch: %d  [%d/%d] | loss: %.4f" %
                  (epoch + 1, i, len(self.train_loader), loss.item()))

        losses /= len(self.train_loader)

        # 计算每层的平均 amplitude 和 shift
        avg_amplitudes = [torch.mean(torch.stack(layer_amplitudes), dim=0) for layer_amplitudes in zip(*all_amplitudes)]
        avg_shifts = [torch.mean(torch.stack(layer_shifts), dim=0) for layer_shifts in zip(*all_shifts)]
        avg_betas = [torch.mean(torch.stack(layer_betas), dim=0) for layer_betas in zip(*all_betas)]
        avg_a = [torch.mean(torch.stack(layer_a), dim=0) for layer_a in zip(*all_a)]

        # 计算每层 multipliers 的平均值
        avg_multipliers = []
        for layer_idx in range(len(batch_multipliers[0])):  # 对每一层进行处理
            # 收集所有批次的同一层 multipliers
            layer_multipliers = [batch[layer_idx] for batch in batch_multipliers]  # 提取同一层
            # 确保所有 multipliers 都是 Tensor
            layer_multipliers = [multiplier[0].clone().detach() if isinstance(multiplier, list) else multiplier
                                 for multiplier in layer_multipliers]
            avg_multipliers.append(torch.mean(torch.stack(layer_multipliers), dim=0))  # 计算平均值



        return losses, avg_amplitudes, avg_shifts, avg_betas,avg_multipliers,avg_a

    def val_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        all_amplitudes = []
        all_shifts = []
        all_betas = []
        all_a = []
        batch_multipliers = []

        for i, batch  in enumerate(self.val_loader, 0):
            with torch.no_grad():
                data_val = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to(self.device)
                target_val = batch[:, -1, :, :].unsqueeze(1).to(torch.float32).to(self.device)

                out_val, seq, _, _, amplitudes, shifts,betas,multipliers,a_par = self.model(data_val)

                mse_loss_val = self.criterion(out_val, target_val) / (
                        self.criterion(target_val, 0 * target_val) + 1e-8)
                val_loss += mse_loss_val.item()

                # 收集所有层的 amplitudes 和 shifts
                all_amplitudes.append(amplitudes)
                all_shifts.append(shifts)
                all_betas.append(betas)
                all_a.append(a_par)
                # 收集当前 batch 的 multipliers
                batch_multipliers.append(multipliers)

        val_loss /= len(self.val_loader)

        # 计算每层的平均 amplitude 和 shift
        avg_amplitudes = [torch.mean(torch.stack(layer_amplitudes), dim=0) for layer_amplitudes in zip(*all_amplitudes)]
        avg_shifts = [torch.mean(torch.stack(layer_shifts), dim=0) for layer_shifts in zip(*all_shifts)]
        avg_betas = [torch.mean(torch.stack(betas_shifts), dim=0) for betas_shifts in zip(*all_betas)]
        avg_a = [torch.mean(torch.stack(layer_a), dim=0) for layer_a in zip(*all_a)]

        # 计算每层 multipliers 的平均值
        avg_multipliers = []
        for layer_idx in range(len(batch_multipliers[0])):  # 对每一层进行处理
            # 收集所有批次的同一层 multipliers
            layer_multipliers = [batch[layer_idx] for batch in batch_multipliers]  # 提取同一层
            # 确保所有 multipliers 都是 Tensor
            layer_multipliers = [multiplier[0].clone().detach() if isinstance(multiplier, list) else multiplier
                                 for multiplier in layer_multipliers]
            avg_multipliers.append(torch.mean(torch.stack(layer_multipliers), dim=0))  # 计算平均值


        return val_loss, avg_amplitudes, avg_shifts,avg_betas,avg_multipliers,avg_a


    def set_test_loader(self, test_loader):
        self.test_loader = test_loader

    def vis_mid_out_slices_fixed_y(self, mid_outputs, i, prefix='test', y_coords=[0, 50, 100, 127]):
        num_layers = self.layer
        num_states = 5  # 每层的状态数量（包括输入和目标）

        for layer_number in range(1, num_layers + 1):  # 层数从1开始
            # Check if this is the last layer
            if layer_number != num_layers:
                continue  # Skip all but the last layer
            fig, axis = plt.subplots(len(y_coords), num_states, figsize=(20, 5 * len(y_coords)))
            ax = axis.reshape(-1, num_states)

            for idx, y in enumerate(y_coords):
                x = np.linspace(-1, 1, len(mid_outputs[0][0][:, :, :, y].squeeze(0).to('cpu').numpy()[-1]))

                # 绘制输入
                slice_input = mid_outputs[0][0][:, :, :, y].squeeze().to('cpu').numpy()
                ax[idx, 0].plot(x, slice_input, color='blue')
                ax[idx, 0].set_title(f'Input (y = {y})')
                ax[idx, 0].set_xlabel('x')
                ax[idx, 0].set_ylabel('u')
                ax[idx, 0].grid(True)

                # 绘制中间层输出
                for j in range(num_states-2):  # 从0开始的索引
                    slice_data = mid_outputs[layer_number][j][:, :, :, y].squeeze().to('cpu').numpy()
                    ax[idx, j + 1].plot(x, slice_data, color=['green', 'red', 'orange'][j % 3])
                    ax[idx, j + 1].set_title(f'State {j + 1} (y = {y})')
                    ax[idx, j + 1].set_xlabel('x')
                    ax[idx, j + 1].set_ylabel('u')
                    ax[idx, j + 1].grid(True)

                # 绘制目标
                slice_target = mid_outputs[-1][0][:, :, :, y].squeeze().to('cpu').numpy()
                ax[idx, -1].plot(x, slice_target, color='purple')
                ax[idx, -1].set_title(f'Target (y = {y})')
                ax[idx, -1].set_xlabel('x')
                ax[idx, -1].set_ylabel('u')
                ax[idx, -1].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'{prefix}_{i}_layer{layer_number}_mid_state_slices_yfix.png'),
                        dpi=300)
            plt.close()

    def vis_mid_out_slices_fixed_x(self, mid_outputs, i, prefix='test',  x_coords=[0, 50, 100, 127]):
        num_layers = self.layer
        num_states = 5  # 每层的状态数量（包括输入和目标）

        for layer_number in range(1, num_layers + 1):  # 层数从1开始
            # Check if this is the last layer
            if layer_number != num_layers:
                continue  # Skip all but the last layer

            fig, axis = plt.subplots(len(x_coords), num_states, figsize=(20, 5 * len(x_coords)))
            ax = axis.reshape(-1, num_states)

            for idx, x in enumerate(x_coords):
                x = np.linspace(-1, 1, len(mid_outputs[0][0][:, :, x, :].squeeze(0).to('cpu').numpy()[-1]))

                # 绘制输入
                slice_input = mid_outputs[0][0][:, :, x, :].squeeze().to('cpu').numpy()
                ax[idx, 0].plot(x, slice_input, color='blue')
                ax[idx, 0].set_title(f'Input (x = {x})')
                ax[idx, 0].set_xlabel('y')
                ax[idx, 0].set_ylabel('u')
                ax[idx, 0].grid(True)

                # 绘制中间层输出
                for j in range(num_layers):  # 从0开始的索引
                    slice_data = mid_outputs[layer_number][j][:, :, x, :].squeeze().to('cpu').numpy()
                    ax[idx, j + 1].plot(x, slice_data, color=['green', 'red', 'orange'][j % 3])
                    ax[idx, j + 1].set_title(f'State {j + 1} (x = {x})')
                    ax[idx, j + 1].set_xlabel('y')
                    ax[idx, j + 1].set_ylabel('u')
                    ax[idx, j + 1].grid(True)

                # 绘制目标
                slice_target = mid_outputs[-1][0][:, :, x, :].squeeze().to('cpu').numpy()
                ax[idx, -1].plot(x, slice_target, color='purple')
                ax[idx, -1].set_title(f'Target (x = {x})')
                ax[idx, -1].set_xlabel('y')
                ax[idx, -1].set_ylabel('u')
                ax[idx, -1].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'{prefix}_{i}_layer{layer_number}_mid_state_slices_xfix.png'),
                        dpi=300)
            plt.close()


    def test(self, vis_mid_out=True, vis_mid_out_slices=True,vis_fourier_out=False):
        model_path = os.path.join(self.save_path, 'model_best_weight.pth')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()

        all_out_tests = []  # 用于存储所有的 out_test 结果
        mse_loss_test_all = []
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):

                data = batch[:, 0, :, :].unsqueeze(1).to(torch.float32).to('mps')
                target = batch[:, -1, :, :].unsqueeze(1).to(torch.float32).to('mps')

                # data, target = batch[0].unsqueeze(1).to(torch.float32).to('mps'), batch[-1].unsqueeze(1).to(torch.float32).to('mps')

                out_test, result_list_All,mid_list,fourier,amplitudes, shifts,betas,_,_ = self.model(data) # result_list_all 包含了所有的状态
                result_list_All.append(target)
                # mid_list.append(target)
                mid_list[-1] = [target]

                result_list = [data]
                mse_loss_test = self.criterion(out_test, target) / (self.criterion(target, 0 * target) + 1e-8)
                mse_loss_test_all.append(mse_loss_test.item())

                result_list.append(out_test)
                result_list.append(target)
                # 保存每次迭代的 out_test
                all_out_tests.append(out_test)
                if vis_mid_out:
                    if i % 10 == 0:
                        # self.vis_mid_out(result_list_All, i, prefix='test')
                        self.vis_mid_out(mid_list, i, prefix='test')
                if vis_mid_out_slices:
                    if i % 10 == 0:
                        # self.vis_mid_out_slices_fixed_y(result_list_All, i, prefix='test')
                        self.vis_mid_out_slices_fixed_y(mid_list, i, prefix='test')
                        # self.vis_mid_out_slices_fixed_x(mid_list, i, prefix='test')
                if vis_fourier_out:
                    if i % 10 == 0:
                        self.vis_fourier_out(fourier,i,prefix='test')
        return mse_loss_test_all ,mse_loss_test, result_list, all_out_tests


    def test_inference(self, out_test_initial, idx, vis_mid_out=True, vis_mid_out_slices=True):
        model_path = os.path.join(self.save_path, 'model_best_weight.pth')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()
        all_out_tests = []  # 用于存储所有的 out_test 结果
        mse_error_test_all = []

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                out_test, result_list_All, mid_list,_,_,_,_,_,_= self.model(out_test_initial[i])
                target =  batch[idx].unsqueeze(1).to(torch.float32).to('mps')

                # 保存当前的结果
                result_list = [out_test_initial, out_test, target]
                result_list_All.append(batch[idx].unsqueeze(1).to(torch.float32).to('mps'))

                # 计算 MSE 损失
                mse_error = self.criterion(out_test, target) / (
                            self.criterion(target, 0 * target) + 1e-8)
                mse_error_test_all.append(mse_error.item())

                # # 计算相对误差
                out_test_cpu = out_test[0][0].to('cpu')
                target_cpu =target[0][0].to('cpu')
                relative_error = relative_squared_error(out_test_cpu, target_cpu)

                # 保存每次迭代的 out_test
                all_out_tests.append(out_test)
                mid_list[-1] = [target]
                # 可视化中间结果
                if vis_mid_out and i % 50 == 0:
                    self.vis_mid_out(mid_list, i, prefix=f'inference_{idx}')

                if vis_mid_out_slices and i % 50 == 0:
                    self.vis_mid_out_slices_fixed_y(mid_list, i, prefix=f'inference_{idx}')

        return mse_error_test_all, all_out_tests, mse_error, relative_error



    def vis_fourier_out(self, fourier_out, i, prefix='test'):
        # Create a figure with one row and three columns for FFT visualization
        fig, ax_row = plt.subplots(1, 3, figsize=(
        24, 8))  # Three columns for ffx, fixed x cross-section, and fixed y cross-section

        for num in range(min(len(fourier_out[0]), 3)):  # Ensure we don't go beyond 3 elements
            # Get the Fourier transform (ffx) magnitude
            ffx_magnitude = torch.abs(fourier_out[num].squeeze(0).squeeze(0))  # Magnitude of Fourier transform

            # Plot 1: Full Fourier transform magnitude (log scale)
            ax_row[0].imshow(ffx_magnitude.cpu().detach().numpy())
            ax_row[0].set_title(f"{prefix} {num + 1}: FFT cale)")
            ax_row[0].axis('off')  # Turn off axis display for a cleaner look

            # Plot 2: Cross-section along a fixed x-axis (y varies, x = 128/2)
            mid_x = ffx_magnitude.shape[1] // 2  # Middle x index (128/2)
            ffx_cross_section_y = ffx_magnitude[:,
                                  mid_x].cpu().detach().numpy()  # Slice along the y-axis at fixed x
            ax_row[1].plot(ffx_cross_section_y)
            ax_row[1].set_title(f"{prefix} {num + 1}: FFT cross-section (y, x={mid_x})")
            ax_row[1].set_xlabel("y")
            ax_row[1].set_ylabel("Magnitude")

            # Plot 3: Cross-section along a fixed y-axis (x varies, y = 128/2)
            mid_y = ffx_magnitude.shape[0] // 2  # Middle y index (128/2)
            ffx_cross_section_x = ffx_magnitude[mid_y,
                                  :].cpu().detach().numpy()  # Slice along the x-axis at fixed y
            ax_row[2].plot(ffx_cross_section_x)
            ax_row[2].set_title(f"{prefix} {num + 1}: FFT cross-section (x, y={mid_y})")
            ax_row[2].set_xlabel("x")
            ax_row[2].set_ylabel("Magnitude")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'{prefix}_{i}_fourier_state.png'), dpi=300)
        plt.close()

    def vis_mse_out(self, all_mse):
        fig, ax = plt.subplots()
        x_labels = ['T', '2T', '3T', '4T']
        mse_values = all_mse[:4]  # Assuming all_mse has MSE values for T, 2T, 3T, and 4T
        mse_values = [mse_value.cpu().item() if isinstance(mse_value, torch.Tensor) else mse_value for mse_value in
                      mse_values]

        ax.plot(x_labels, mse_values, marker='o')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('MSE Loss')
        ax.set_title('MSE Loss over Time Steps')
        plt.savefig(os.path.join(self.save_path, 'mse_loss_over_time.png'), dpi=300)
        plt.close()



    def vis_mid_out(self, mid_outputs, i, prefix='test'):
        num_layers = self.layer
        num_states = 5  # 每层的状态数量（包括输入和目标）

        for layer_number in range(1, num_layers + 1):  # 层数从1开始
            # Check if this is the last layer
            if layer_number != num_layers:
                continue  # Skip all but the last layer
            fig, axes = plt.subplots(2, num_states, figsize=(20, 12))  # 设置为两行 num_states 列

            # 第一列：原始数据
            input_data = mid_outputs[0][0].squeeze(0).squeeze(0).cpu().numpy()  # 获取输入数据
            target = mid_outputs[-1][0].squeeze(0).squeeze(0).cpu().numpy()  # 获取目标数据

            axes[0, 0].imshow(input_data, cmap='jet')
            axes[0, 0].set_title(f'Layer {layer_number} - Input')

            # 显示当前层的输出
            for state_number in range(len(mid_outputs[layer_number])):  # 遍历该层的所有输出
                out = mid_outputs[layer_number][state_number].squeeze(0).squeeze(0).cpu().numpy()  # 当前层输出
                if state_number < num_states - 1:  # 确保不超出范围，留出一列给目标数据
                    axes[0, state_number + 1].imshow(out, cmap='jet')
                    axes[0, state_number + 1].set_title(f'Layer {layer_number} - Output {state_number + 1}')

            # 在最后一列显示目标数据
            axes[0, num_states - 1].imshow(target, cmap='jet')  # 目标数据在第一行的最后一列
            axes[0, num_states - 1].set_title(f'Layer {layer_number} - Target')

            # 第二行：对数绝对值图
            axes[1, 0].imshow(np.log(np.abs(input_data) + 1e-8), cmap='jet')  # 输入的对数图
            axes[1, 0].set_title(f'Layer {layer_number} - Input (log(abs))')

            for state_number in range(len(mid_outputs[layer_number])):  # 遍历该层的所有输出
                out = mid_outputs[layer_number][state_number].squeeze(0).squeeze(0).cpu().numpy()  # 当前层输出的对数图
                if state_number < num_states - 1:  # 确保不超出范围，留出一列给目标数据
                    axes[1, state_number + 1].imshow(np.log(np.abs(out) + 1e-8), cmap='jet')
                    axes[1, state_number + 1].set_title(f'Layer {layer_number} - Output {state_number + 1} (log(abs))')

            # 显示目标的对数图
            axes[1, num_states - 1].imshow(np.log(np.abs(target) + 1e-8), cmap='jet')  # 目标数据在第二行的最后一列
            axes[1, num_states - 1].set_title(f'Layer {layer_number} - Target (log(abs))')

            # 添加颜色条
            for ax_row in axes.flat:  # 使用 flat 方便遍历所有子图
                cbar = fig.colorbar(ax_row.images[-1], ax=ax_row, orientation='horizontal', fraction=0.046, pad=0.1)
                cbar.ax.xaxis.set_ticks_position('bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'{prefix}_{i}_layer_{layer_number}_mid_state.png'), dpi=300)
            plt.close()  # 关闭当前图形，释放内存

    def vis_relative_error_out(self, all_relative_errors):
        fig, ax = plt.subplots()
        x_labels = ['T', '2T', '3T', '4T']
        relative_error_values = all_relative_errors[:4]  # Assuming all_relative_errors has values for T, 2T, 3T, and 4T

        ax.plot(x_labels, relative_error_values, marker='o')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Relative Error')
        ax.set_title('Relative Error over Time Steps')
        plt.savefig(os.path.join(self.save_path, 'relative_error_over_time.png'), dpi=300)
        plt.close()
