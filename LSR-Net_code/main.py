import sys
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from CH_GRF_dataset import CH_GRF_Dataset, train_val_dataset
from Network import Wise_SpectralTransform
from torch.utils.data import DataLoader
from torchvision import transforms


from Auto_GRF_Normal_CH_LR_trainer_5T import LR_Trainer
from Auto_GRF_Normal_CH_Wise_Pre_Trainer import Wise_LR_Trainer
from plot_difference_ch import *
from plot_energy_ch_k8_me05 import *
from torchinfo import summary
import os
import torch.nn as nn
import numpy as np
import csv
import pdb
pdb.set_trace = lambda: None
def count_model_params(model):
    """Function to count the total number of parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
def save_model_summary(model, outf, model_name="model_summary.txt", net_name="model",batch_size = None):
    """Function to save the model summary and total parameter count to a file."""
    # Get the model summary string
    summary_str = summary(model, input_size=(batch_size, 1, 128, 128), device='cuda')

    # Count total parameters
    total_params = count_model_params(model)

    # File path for saving the summary
    summary_file_path = os.path.join(outf, model_name)

    # Open the file in append mode ('a') to add the summary to the file
    with open(summary_file_path, 'a') as f:
        # Add the network's name and parameter count
        f.write(f"Model: {net_name}\n")  # Use the network name here
        f.write(str(summary_str))  # Save the summary
        f.write('\n' + '-' * 80 + '\n')  # Add separator

        # Save the total parameter count
        # f.write(f"Total number of parameters in {net_name}: {total_params}\n")
        f.write('-' * 80 + '\n')  # Add another separator for clarity

    print(f"Model summary and parameter count for {net_name} saved to {summary_file_path}")
    return total_params  # Return the parameter count of the current model

def generate_inference_times(t, total_time):
    times = list(range(t * 2, total_time, t))
    if times and times[-1] + t <= total_time:
        times.append(times[-1] + t)
    return times


def generate_test_times(t, total_time):
    times = list(range(t, total_time, t))
    if times and times[-1] + t <= total_time:
        times.append(times[-1] + t)
    return times



def main(num_layers, kernel_sizes,model_times):


    data_path = os.path.join(r'/home/qianhou/FFT_PDE/Cahn_Hillard_inference/GRF_ICs/Inference_Case_v2_sos/k8_m0.5_ch_grf_T70_500_old.h5') 

    device = 'cuda'
    pre_lr = 1e-3
    lr = 1e-3
    seed = 999
    pre_epoch = 10
    epoch = 50
    model_type = 'LR'
    channe_features = [7]
    batch_size=4
    forward_steps = [2]
    results = {}
    a = 'cir_999'

    T_all = [7]  
    total_time = 49
    grad_coef =0.3

    for model_time in model_times:
        for T in T_all:
            for features in channe_features:
                for kernel_size in kernel_sizes:
                    for forward_step in forward_steps:
            
                        total_params_all_models = 0  #
                        idx_initial = T-1
                        outf = f'./V2_k8_me05_Grad{grad_coef}/Test_SOS/Autostep{forward_step}/model{model_time}/CH_T{T}s_T{total_time}/Layer{num_layers}_channel{features}_ks_{kernel_size}/_a_{a}_pre_epoch_{pre_epoch}/bs{batch_size}_{model_type}_lr{lr}_epoch{epoch}/'
                        dataset = CH_GRF_Dataset(data_path, transform=transforms.ToTensor())
                        dataset = train_val_dataset(dataset, seed)

                        loader_train = DataLoader(dataset['train'], batch_size=batch_size)
                        loader_val = DataLoader(dataset['val'], batch_size=batch_size)
                        loader_test = DataLoader(dataset['test'], batch_size=1)

                        criterion = nn.MSELoss(reduction='mean').to(device)


                        all_times = generate_test_times(T, total_time)
                        # all_steps = [x * 10-1 for x in all_times]
                        all_steps = [x -1 for x in all_times]


                        inference_times = generate_inference_times(T, total_time)
                        # inference_steps = [x* 10-1 for x in inference_times]
                        inference_steps = [x -1 for x in inference_times]
                        pdb.set_trace()
                        #
                        # ====== Phase 1: Train amplitude only ======
                
                        initial_parameters = {
                            'amplitudes': [torch.randn(features).to(device) for _ in range(num_layers)], 
                            'shifts': [torch.zeros(features).to(device) for _ in range(num_layers)], 
                            'betas': [torch.ones(features).to(device) for _ in range(num_layers)]  #
                        }

                        pre_net1 = Wise_SpectralTransform(num_layers=num_layers, channel_feat=features, kernel_size=kernel_size,
                                                        phase=1, initial_parameters=initial_parameters)
                        
                        pre_optimizer1 = optim.Adam(pre_net1.parameters(), lr=pre_lr, eps=1e-8, weight_decay=1e-5)
                        # pre_optimizer1 = optim.AdamW(pre_net1.parameters(), lr=lr, eps=1e-8, weight_decay=1e-4)

                        scheduler = StepLR(pre_optimizer1, step_size=16, gamma=0.1)
                        os.makedirs(outf, exist_ok=True)

                        # Save model summary after model initialization
                        total_params_all_models += save_model_summary(pre_net1, outf, model_name="model_summary.txt",
                                                                    net_name="pre_net1",batch_size=batch_size)

                        # Pre-train: only train amplitude
                        pre_trainer1 = Wise_LR_Trainer(loader_train, loader_val, loader_test, pre_net1, pre_optimizer1, scheduler,
                                                    criterion, pre_epoch, device, outf, outf, idx=idx_initial,layer=num_layers,all_steps=all_steps,
                                                    forward_step=forward_step,grad_coef=grad_coef,resume=False)
                        pre_loss, last_amplitudes, _, _ = pre_trainer1.train()

                        # ====== Phase 2: Train beta with fixed amplitude and shift ======
               
                        initial_parameters2 = {
                            'amplitudes': [torch.tensor(amplitude, requires_grad=False).to(device) for amplitude in last_amplitudes],
                            # 固定 amplitude
                            'shifts': [torch.zeros(features).to(device) for _ in range(num_layers)],  
                            'betas': [torch.ones(features).to(device) for _ in range(num_layers)]
                        }

                        pre_net2 = Wise_SpectralTransform(num_layers=num_layers, channel_feat=features, kernel_size=kernel_size,
                                                        phase=2, initial_parameters=initial_parameters2)
                        # Save model summary for phase 2 model
                        total_params_all_models += save_model_summary(pre_net2, outf, model_name="model_summary.txt",
                                                                    net_name="pre_net2",batch_size=batch_size)

                        pre_optimizer2 = optim.Adam(pre_net2.parameters(), lr=pre_lr, eps=1e-8, weight_decay=1e-5)




                        scheduler = StepLR(pre_optimizer2, step_size=16, gamma=0.1)

                        # Phase 2: Train beta with fixed amplitude and shift
                        pre_trainer2 = Wise_LR_Trainer(loader_train, loader_val, loader_test, pre_net2, pre_optimizer2, scheduler,
                                                    criterion, pre_epoch, device, outf, outf, idx=idx_initial,layer=num_layers,all_steps=all_steps,
                                                    forward_step=forward_step,grad_coef=grad_coef,resume=False)
                        pre_loss, _, _ , last_betas = pre_trainer2.train()

                        # ====== Phase 3: Train shift with fixed amplitude and beta ======
                
                        initial_parameters3 = {
                            'amplitudes': [torch.tensor(amplitude, requires_grad=False).to(device) for amplitude in last_amplitudes],
                            # 固定 amplitude
                            'shifts': [torch.randn(features).to(device) for _ in range(num_layers)],
                            'betas': [torch.tensor(beta, requires_grad=False).to(device) for beta in last_betas] 
                        }


                        pre_net3 = Wise_SpectralTransform(num_layers=num_layers, channel_feat=features, kernel_size=kernel_size,
                                                        phase=3, initial_parameters=initial_parameters3)
                        # Save model summary for phase 2 model
                        total_params_all_models += save_model_summary(pre_net3, outf, model_name="model_summary.txt",
                                                                    net_name="pre_net3",batch_size=batch_size)


                        pre_optimizer3 = optim.Adam(pre_net3.parameters(), lr=pre_lr, eps=1e-8, weight_decay=1e-5)

       



                        scheduler = StepLR(pre_optimizer3, step_size=16, gamma=0.1)

                        # Phase 3: Train shift with fixed amplitude and beta
                        pre_trainer3 = Wise_LR_Trainer(loader_train, loader_val, loader_test, pre_net3, pre_optimizer3, scheduler,
                                                    criterion, pre_epoch, device, outf, outf, idx=idx_initial,layer=num_layers,all_steps=all_steps,
                                                    forward_step=forward_step,grad_coef=grad_coef,resume=False)
                        pre_loss, _, last_shifts,_ = pre_trainer3.train()

                        # ====== Phase 4: Train all parameters together ======
                        initial_parameters4 = {
                            'amplitudes': [torch.tensor(amplitude).to(device) for amplitude in last_amplitudes], 
                            'shifts': [torch.tensor(shift).to(device) for shift in last_shifts], 
                            'betas': [torch.tensor(beta).to(device) for beta in last_betas]  
                        }
                        # Save initial parameters
                        initial_parameters_path = os.path.join(outf, 'initial_parameters4.pth')
                        torch.save(initial_parameters4, initial_parameters_path)
                        print(f'Initial parameters saved at {initial_parameters_path}')

             
                        net = Wise_SpectralTransform(num_layers=num_layers, channel_feat=features, kernel_size=kernel_size,phase=4,
                                                    initial_parameters=initial_parameters4)
                        # # Save model summary for phase 2 model
                        total_params_all_models += save_model_summary(net, outf, model_name="model_summary.txt", net_name="net",batch_size=batch_size)
                        # Print total parameters after all phases
                        print(f"Total number of parameters for all networks: {total_params_all_models}")
                        # Append the total parameters to the file
                        with open(os.path.join(outf, "model_summary.txt"), 'a') as f:
                            f.write(f"\nTotal number of parameters for all networks: {total_params_all_models}\n")
                            f.write("=" * 80 + "\n")
                    
                        optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-8, weight_decay=1e-5)

                    
                        scheduler = StepLR(optimizer, step_size=16, gamma=0.1)

                        # # Phase 4: Train all parameters together
                        trainer = LR_Trainer(loader_train, loader_val, loader_test, net, optimizer, scheduler, criterion, epoch,
                                            device, outf, outf, idx=idx_initial,layer=num_layers,all_steps=all_steps,
                                            forward_step=forward_step,grad_coef=grad_coef,resume=False)
                        lowest_loss, time = trainer.train()

                        print(f'training time spent: {time}s')

                        mse_all_10, loss_test, result_list, out_test_10,init_data = trainer.test()
                        print('loss_test', loss_test)
                        print('--------------------------------')
                        results["Cahn--Hilliard"] = trainer.train_result
                        
                        plot_test(model=net, save_path=outf, test_loader=loader_test, layer=num_layers,idx=idx_initial)




            
                        initial_energy = compute_initial_energy(test_loader=loader_test, device=device)

           
                        test_energy = compute_test_energy(model=net, test_loader=loader_test, idx=idx_initial,device=device)

 
                        # 原有代码：Total_energy=[test_energy]
                        Total_energy = [initial_energy] + [test_energy]

                        initial_energy_mean = np.mean(initial_energy, axis=0)
                        Total_energy_mean = [initial_energy_mean] + [np.mean(test_energy, axis=0)]


                        all_mse = []
                        all_mse.append(np.mean(mse_all_10))
                        loader_test_extra = DataLoader(dataset['test'], batch_size=1)
                        trainer.set_test_loader(loader_test_extra)

                     
                        out_test_prev = out_test_10
                        Total_mse = [mse_all_10]

                        all_out_test = []
                        all_out_test.append(np.array([tensor.cpu().numpy() for tensor in init_data]))  
                        all_out_test.append(np.array([tensor.cpu().numpy() for tensor in out_test_prev]))  
                        for step in inference_steps:
                            mse_all, out_test, mse, relative_error = trainer.test_inference(out_test_prev, step)
                    
                            # pdb.set_trace()
                            all_out_test.append(np.array([tensor.cpu().numpy() for tensor in out_test])) 
            
                            all_mse.append(np.mean(mse_all))
                            Total_mse.append(mse_all)
                            infer_engergy =  compute_infer_energy(model=net, test_loader=loader_test, idx=step, device=device,
                                                                out_test_prev=out_test_prev)
                            Total_energy.append(infer_engergy)
                            Total_energy_mean.append((np.mean(infer_engergy, axis=0)))
                            plot_infer(model=net, save_path=outf, test_loader=loader_test, layer=num_layers,
                                    idx=step, out_test_prev=out_test_prev)
                            out_test_prev = out_test 

                            print('--------------------------------')
                    
                        trainer.vis_mse_out(all_mse,T)
                        prefix = 'FFT'
 
                        plot_energy(Total_energy_mean, T, out=all_out_test, prefix=prefix, save_path=outf)

       
                        csv_path = os.path.join(outf, 'Total_mse_mean.csv')
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(["T1"] + [f"T{i + 2}" for i in range(len(inference_steps))])  # Write header
                            writer.writerow(all_mse)  # Write the MSE values as a single row


         
                        csv_path = os.path.join(outf, 'Total_mse.csv')
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(["T1"] + [f"T{i + 2}" for i in range(len(inference_steps))])
                            writer.writerows(zip(*Total_mse))  

                   
                        csv_path = os.path.join(outf, f'{prefix}_Total_energy_mean.csv')
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(["T0(initial)", "T1"] + [f"T{i + 2}" for i in range(len(inference_steps))])
                            writer.writerow(Total_energy_mean) 

       
                        csv_path = os.path.join(outf, f'{prefix}_Total_energy.csv')
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(["T0(initial)", "T1"] + [f"T{i + 2}" for i in range(len(inference_steps))])
                            writer.writerows(zip(*Total_energy)) 


if __name__ == '__main__':
    layers = [3]
    kernel_sizes = [7]


    model_times=[1,2,3,4,5]
    for num_layers in layers:
        main(num_layers, kernel_sizes=kernel_sizes,model_times=model_times)
