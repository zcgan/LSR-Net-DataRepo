import h5py
import matplotlib.pyplot as plt


file_path = './LSR_Net_Datasets/inference_tests/Infer_FH.h5'


sample_idx = 0
step_idx = -1

with h5py.File(file_path, 'r') as f:
    data = f['data'][:]


img = data[sample_idx, step_idx, :, :]

# 绘制图像
plt.figure(figsize=(6, 5))
plt.imshow(img, cmap='jet')
plt.colorbar(label='Value')
plt.title(f"Sample {sample_idx}, Time Step Index {step_idx}")
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
