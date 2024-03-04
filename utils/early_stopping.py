import os

import numpy as np
import torch
from cprint import cprint


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, folder_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, folder_path)

        elif score < self.best_score:
            self.counter += 1
            cprint.info("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model, folder_path)

    def save_checkpoint(self, val_loss, model, folder_path):
        cprint.info('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'.format(self.val_loss_min, val_loss))
        checkpoint_folder = os.path.join(folder_path, 'weights')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        torch.save(model.state_dict(), checkpoint_folder + '/model.pth')
        self.val_loss_min = val_loss
