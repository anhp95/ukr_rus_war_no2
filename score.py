import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats


class Score:
    """
    List of scores used in the study
    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        self.score_r = 0

        self.score_rmse = 0
        self.score_n_rmse = 0

        self.score_mbe = 0
        self.score_n_mbe = 0

        self.cal_pearsonr()
        self.cal_rmse()
        self.cal_mbe()
        self.print_score()

    def cal_pearsonr(self):
        self.score_r = stats.pearsonr(self.y_true, self.y_pred)[0]

    def cal_rmse(self):
        self.score_rmse = mean_squared_error(self.y_true, self.y_pred, squared=False)
        self.score_n_rmse = np.sum(np.square(self.y_true - self.y_pred)) / np.sum(
            np.square(self.y_true)
        )

    def cal_mbe(self):
        self.score_mbe = np.mean(self.y_true - self.y_pred)
        self.score_n_mbe = np.sum(self.y_true - self.y_pred) / np.sum(self.y_true)

    def print_score(self):
        print(f"R: {self.score_r}")
        print(f"RMSE: {self.score_rmse}")
        print(f"NormRMSE: {self.score_n_rmse}")
        print(f"MBE: {self.score_mbe}")
        print(f"NormMBE: {self.score_n_mbe}")


# %%
# rmse_test = mean_squared_error(y_test, y_pred, squared=False)
# rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)

# r2_test = r2_score(y_test, y_pred)
# r2_train = r2_score(y_train, y_pred_train)

# norm_rmse_test_2 = np.sum(np.square(y_test - y_pred)) / np.sum(
#     np.square(y_test)
# )
# norm_rmse_train_2 = np.sum(np.square(y_train - y_pred_train)) / np.sum(
#     np.square(y_train)
# )
# print("y max:", np.amax(y_test), "y min:", np.amin(y_test))
# norm_rmse_test = (
#     norm_rmse_test_2,
#     rmse_test / np.amax(y_test),
#     rmse_test / (np.amax(y_test) - np.amin(y_test)),
# )
# norm_rmse_train = (
#     norm_rmse_train_2,
#     rmse_train / np.amax(y_train),
#     rmse_train / (np.amax(y_train) - np.amin(y_train)),
# )

# MBE_test = np.mean(y_pred - y_test)
# MBE_train = np.mean(y_pred_train - y_train)

# norm_MBE_test = np.sum(y_pred - y_test) / np.sum(y_test)
# norm_MBE_train = np.sum(y_pred_train - y_train) / np.sum(y_train)

# print(f"rmse test: {rmse_test}")
# print(f"rmse train: {rmse_train}")

# print(f"norm rmse test: {norm_rmse_test}")
# print(f"norm rmse train: {norm_rmse_train}")

# print(f"mbe test: {MBE_test}")
# print(f"mbe train: {MBE_train}")

# print(f"norm mbe test: {norm_MBE_test}")
# print(f"norm mbe train: {norm_MBE_train}")

# print(f"r2 score test: {r2_test}")
# print(f"r2 score train: {r2_train}")
