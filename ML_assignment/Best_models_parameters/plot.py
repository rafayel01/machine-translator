import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("seq2seq_LSTM_best_plot_val_loss", "rb") as f:
    tr_loss = pickle.load(f)


with open("seq2seq_LSTM_best_plot_val_perp", "rb") as f:
    tr_perp = pickle.load(f)

x = np.arange(1, 101)
print("Val loss: ", min(tr_loss))
print("Val perp: ", min(tr_perp))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))
fig.suptitle('Validation Loss and perplexity curves for Transformer')
ax1.set_xlabel("Epochs")
ax2.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Perplexity")
ax1.set_title("Loss vs Epochs")
ax2.set_title("Perplexity vs Epochs")
ax1.plot(x, tr_loss)
ax2.plot(x, tr_perp)

fig.savefig("Seq2seq_lstm_valid_loss_perp.png")