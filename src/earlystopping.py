import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='checkpoint_model.pth'):

        self.patience = patience    # 設定ストップカウンタ
        self.verbose = verbose      # 表示の有無
        self.counter = 0            # 現在のカウンタ値
        self.best_score = None      # ベストスコア
        self.early_stop = False     # ストップフラグ
        self.val_loss_min = np.Inf  # 前回のベストスコア記憶用
        self.path = path            # ベストモデル格納path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:                                                       # 1Epoch目
            self.best_score = score                                                       # ベストスコアとして記録
            self.checkpoint(val_loss, model)

        elif score <= self.best_score:                                                    # ベストスコアを更新できなかった場合
            self.counter += 1                                                             # ストップカウンタ+1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')    # 現在のカウンタ表示
            if self.counter >= self.patience:                                             # 設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:                                                                             # ベストスコアを更新した場合
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0                                                              # ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        if self.verbose:                                                                  # 更新値を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.module.state_dict(), self.path)                                  # ベストモデルを保存
        self.val_loss_min = val_loss                                                      # lossを記録
