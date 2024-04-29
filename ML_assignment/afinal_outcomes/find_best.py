import torch
import numpy as np

lr  = [0.01, 0.001, 0.0001, 0.0005, 5e-5]
bs = [32, 64, 128, 256, 512]

min_loss = np.inf
best_model = torch.load("seq2seq_64_0.001.pt")
# for b in bs:
#     for l in lr:
#         model_params = torch.load("trans_" + str(b) + "_" + str(l) + ".pt")
#         val_loss = model_params['loss']
#         if val_loss < min_loss:
#             min_loss = val_loss
#             best_model = model_params

print("Best model params: ", best_model['loss'])
print(best_model['lr'])
print(best_model['batch_size'])
print(best_model['epoch'])