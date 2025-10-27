
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional.regression import relative_squared_error
import torch
import pandas as pd
import os
from datetime import datetime
import torch.nn as nn
import csv


class Wise_LR_Trainer():
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
        self.Pre_train_result = pd.DataFrame(columns=['Epoch', 'train loss', 'val loss'])
        self.Pre_Four_param = pd.DataFrame(columns=['Phase', 'Epoch', 'Layer', 'Amplitude', 'Shift', 'Beta','A_par'])
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

        last_amplitudes = []  # 存储每个层的amplitude
        last_shifts = []  # 存储每个层的shift
        last_betas = []  # 存储每个层的beta

        for epoch in range(self.start_epoch, self.epoch):
            # 训练阶段
            losses, train_amplitudes, train_shifts, train_betas, train_multipliers, train_apar = self.train_epoch(epoch)
            self.train_multipliers.append(train_multipliers)

            # 验证阶段
            val_loss, val_amplitudes, val_shifts, val_betas, val_multipliers, val_apar = self.val_epoch(epoch)
            self.val_multipliers.append(val_multipliers)

            if writer:
                writer.add_scalar('Loss/train', losses, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)

            temp_df = pd.DataFrame({'Pre Epoch': epoch + 1, 'Pre train loss': losses, 'Pre val loss': val_loss}, index=[0])
            self.Pre_train_result = pd.concat([self.Pre_train_result, temp_df], ignore_index=True)

            for layer_idx in range(len(train_amplitudes)):
                amplitude_list = train_amplitudes[layer_idx].detach().cpu().numpy().tolist()
                shift_list = train_shifts[layer_idx].detach().cpu().numpy().tolist()
                beta_list = train_betas[layer_idx].detach().cpu().numpy().tolist()
                apar_list = train_apar[layer_idx].detach().cpu().numpy().tolist()

                val_amplitudes_list = val_amplitudes[layer_idx].detach().cpu().numpy().tolist()
                val_shifts_list = val_shifts[layer_idx].detach().cpu().numpy().tolist()
                val_betas_list = val_betas[layer_idx].detach().cpu().numpy().tolist()
                val_apar_list = val_apar[layer_idx].detach().cpu().numpy().tolist()

                self.Pre_Four_param = pd.concat([
                    self.Pre_Four_param,
                    pd.DataFrame({
                        'Phase': ['train', 'val'],
                        'Epoch': [epoch + 1, epoch + 1],
                        'Layer': [layer_idx, layer_idx],
                        'Amplitude': [amplitude_list, val_amplitudes_list],
                        'Shift': [shift_list, val_shifts_list],
                        'Beta': [beta_list, val_betas_list],
                        'A_par': [apar_list, val_apar_list]
                    })
                ], ignore_index=True)

                # 仅在最后一个 epoch 保存 amplitudes, shifts, betas
                if epoch == self.epoch - 1:
                    last_amplitudes = train_amplitudes
                    last_shifts = train_shifts
                    last_betas = train_betas

                # # 仅在最后一个 layer 保存 amplitudes, shifts, betas
                # if layer_idx == len(train_amplitudes) - 1:  # 只有最后一个 layer
                #     if epoch == self.epoch - 1:  # 仅在最后一个 epoch
                #         last_amplitudes = amplitude_list
                #         last_shifts = shift_list
                #         last_betas = beta_list

            self.scheduler.step()
            print("\n[epoch %d] Loss_val: %.4f" % (epoch + 1, val_loss))

            # # 添加提前停止的判断
            if val_loss < 0.01:
                last_amplitudes = train_amplitudes
                last_shifts = train_shifts
                last_betas = train_betas
                print(f"Validation loss has reached {val_loss:.4f}, stopping training early.")
                break

            is_best = val_loss < self.lowest_loss
            if is_best:
                self.lowest_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'loss': val_loss,
                }, os.path.join(self.save_path, 'model_best_weight.pth'))
                print('lowest_loss %f' % (self.lowest_loss))

            self.Pre_Four_param.to_csv(os.path.join(self.save_path, 'Pre_Four_param.csv'), index=False)
            self.Pre_train_result.to_csv(os.path.join(self.save_path, 'Pre_train_result.csv'), index=False)

        end = datetime.utcnow()
        time_len = (end - since).total_seconds()

        # 训练结束后保存所有 multipliers
        self.save_multipliers()
        return self.lowest_loss, last_amplitudes, last_shifts, last_betas

    def save_multipliers(self):

        train_multipliers_np = np.array([[layer_multipliers.cpu().numpy() for layer_multipliers in epoch_multipliers]
                                         for epoch_multipliers in self.train_multipliers], dtype=object)
        val_multipliers_np = np.array([[layer_multipliers.cpu().numpy() for layer_multipliers in epoch_multipliers]
                                       for epoch_multipliers in self.val_multipliers], dtype=object)

        # Define the file paths for saving the multipliers as .npy files
        train_multipliers_path = os.path.join(self.save_path, 'Pre_train_multipliers.npy')
        val_multipliers_path = os.path.join(self.save_path, 'Pre_val_multipliers.npy')

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

        for i, batch  in enumerate(self.train_loader):
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

            print("Pre epoch: %d  [%d/%d] | Pre loss: %.4f" %
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

        for i, batch in enumerate(self.val_loader, 0):
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