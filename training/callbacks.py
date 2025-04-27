# training/callbacks.py

import os
import torch
import numpy as np


class Callback:
    """Base class for all callbacks."""
    def on_epoch_start(self, trainer): pass
    def on_epoch_end(self, trainer): pass
    def on_batch_start(self, trainer): pass
    def on_batch_end(self, trainer): pass
    def on_train_end(self, trainer): pass

class ModelCheckpoint(Callback):
    def __init__(self, save_dir, monitor='val_loss', mode='min'):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = np.inf if mode == 'min' else -np.inf

    def on_epoch_end(self, trainer: "ExpertTrainer"):
        score = trainer.metrics[self.monitor][-1]
        improved = score < self.best_score if self.mode == 'min' else score > self.best_score
        if improved:
            self.best_score = score
            trainer._save_checkpoint(0,"-best")
            print(f"[Checkpoint] Saved best model...")

class EarlyStopping(Callback):
    def __init__(self, patience=5, monitor='val_loss', mode='min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, trainer):
        score = trainer.metrics[self.monitor][-1]
        improved = score < self.best_score if self.mode == 'min' else score > self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] Patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("[EarlyStopping] Stopping training.")
                self.should_stop = True
