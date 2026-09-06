!pip install pytorch-forecasting lightning scikit-learn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
from lightning.pytorch import Trainer
from scipy.special import expit  # logistic function

import time

def simulate_data(n_dpy=100, n_year=5):

    n = n_dpy * n_year

    # parameters
    theta0 = -0.8
    theta1 = 0.78
    beta_season1 = 0.2
    beta_season2 = -1.4
    beta_seasonality2 = 0.3
    beta_seasonality3 = 0.9
    gamma00 = 0.9
    gamma01 = -0.2
    gamma10 = 0.1
    gamma11 = -0.7

    t_vec = np.arange(1, n+1)

    # covariates (same structure as R)
    Is_Season1 = np.tile(np.concatenate([np.ones(30), np.zeros(70)]), n_year)
    Is_Season2 = np.tile(np.concatenate([np.zeros(30), np.ones(40), np.zeros(30)]), n_year)

    Is_Seasonality2 = np.tile(np.tile(np.concatenate([np.zeros(3), np.ones(2)]), 20), n_year)
    Is_Seasonality3 = np.tile(np.concatenate([np.zeros(80), np.ones(5), np.zeros(15)]), n_year)

    # linear predictor
    lc = (beta_season1 * Is_Season1 +
          beta_season2 * Is_Season2 +
          beta_seasonality2 * Is_Seasonality2 +
          beta_seasonality3 * Is_Seasonality3)

    y_vec = np.zeros(n + 1, dtype=int)

    y_vec[0] = np.random.choice([0, 1, 2])

    for t in range(1, n + 1):

        prev_y = y_vec[t - 1]

        I0 = int(prev_y == 0)
        I1 = int(prev_y == 1)   # category 2 is baseline

        eta0 = lc[t - 1] + gamma00 * I0 + gamma01 * I1
        eta1 = lc[t - 1] + gamma10 * I0 + gamma11 * I1

        F0 = expit(theta0 - eta0)
        F1 = expit(theta1 - eta1)

        p0 = F0
        p1 = F1 - F0
        p2 = 1 - p0 - p1

        probs = np.array([p0, p1, p2])
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum()

        y_vec[t] = np.random.choice([0, 1, 2], p=probs)

    df = pd.DataFrame({
        "t": t_vec,
        "Response": y_vec[1:],
        "Is_Season1": Is_Season1,
        "Is_Season2": Is_Season2,
        "Is_Seasonality2": Is_Seasonality2,
        "Is_Seasonality3": Is_Seasonality3
    })

    return df
  
def fit_one_tft(df):

    ########################
    df_tft = df.copy()

    df_tft["time_idx"] = np.arange(len(df_tft))
    df_tft["series_id"] = 0

    # lag features (important for TFT)
    df_tft["lag1"] = df_tft["Response"].shift(1)
    df_tft = df_tft.dropna().reset_index(drop=True)

    test_size = int(0.2 * len(df_tft))
    train_tft = df_tft.iloc[:-test_size]
    test_tft  = df_tft.iloc[-test_size:]

    #########################
    training = TimeSeriesDataSet(
        train_tft,
        time_idx="time_idx",
        target="Response",
        group_ids=["series_id"],

        max_encoder_length=0,
        max_prediction_length=1,

        time_varying_known_reals=[
            "Is_Season1","Is_Season2",
            "Is_Seasonality2","Is_Seasonality3",
            "lag1"
        ],

        target_normalizer=None,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        test_tft,
        predict=False,
        stop_randomization=True
    )

    train_loader = training.to_dataloader(train=True, batch_size=64)
    val_loader = validation.to_dataloader(train=False, batch_size=64)

    ########################
    tft = TemporalFusionTransformer.from_dataset(
        training,
    hidden_size=8,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=4,
        loss=CrossEntropy(),
        output_size=3,   # 🔥 3 categories
    )

    trainer = Trainer(
        max_epochs=5,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False
    )

    trainer.fit(tft, train_loader, val_loader)

    ##########################
    raw_predictions = tft.predict(val_loader, mode="raw")

    logits = raw_predictions.prediction
    preds_tft = torch.argmax(logits, dim=-1).squeeze(-1).cpu().numpy()

    actuals = []

    for x, y in val_loader:
      actuals.append(y[0])

    actuals = torch.cat(actuals).flatten().cpu().numpy()

    acc_tft = accuracy_score(actuals, preds_tft)
    f1_tft  = f1_score(actuals, preds_tft, average="weighted")

    #############################
    return acc_tft, f1_tft
  
# ==============================
# LSTM DATASET
# ==============================
class SequenceDataset(Dataset):
    def __init__(self, df, seq_len=20):
        self.seq_len = seq_len

        features = df[[
            "Response",   # lagged target
            "Is_Season1","Is_Season2",
            "Is_Seasonality2","Is_Seasonality3"
        ]].values

        target = df["Response"].values

        self.X = []
        self.y = []

        for i in range(len(df) - seq_len):
            self.X.append(features[i:i+seq_len])
            self.y.append(target[i+seq_len])

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
      
# ==============================
# LSTM MODEL
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)   # 🔥 3 categories

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ==============================
# LSTM FIT FUNCTION
# ==============================
def fit_one_lstm(df):

    df_lstm = df.copy()

    seq_len = 1

    test_size = int(0.2 * len(df_lstm))
    train_df = df_lstm.iloc[:-test_size].reset_index(drop=True)
    test_df  = df_lstm.iloc[-test_size:].reset_index(drop=True)

    train_dataset = SequenceDataset(train_df, seq_len)
    test_dataset  = SequenceDataset(test_df, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = LSTMModel(input_size=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training
    model.train()
    for epoch in range(10):
        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

    # evaluation
    model.eval()

    preds_list = []
    actuals_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            preds_list.append(preds)
            actuals_list.append(y_batch)

    preds = torch.cat(preds_list).cpu().numpy()
    actuals = torch.cat(actuals_list).cpu().numpy()

    acc = accuracy_score(actuals, preds)
    f1  = f1_score(actuals, preds, average="weighted")

    return acc, f1
  
n_sim = 1

results_tft = []
results_lstm = []

time_tft = []
time_lstm = []

for i in range(n_sim):
    print(f"Simulation {i+1}/{n_sim}")
    df = simulate_data(n_dpy=100, n_year=5)
    # TFT
    start = time.time()
    res_tft = fit_one_tft(df)
    time_tft.append(time.time() - start)

    # LSTM
    start = time.time()
    res_lstm = fit_one_lstm(df)
    time_lstm.append(time.time() - start)

    results_tft.append(res_tft)
    results_lstm.append(res_lstm)


results_tft = np.array(results_tft)
results_lstm = np.array(results_lstm)

time_tft = np.array(time_tft)
time_lstm = np.array(time_lstm)

# ==============================
# SUMMARY
# ==============================
summary = pd.DataFrame({
    "Model": ["TFT", "LSTM"],

    "Accuracy Mean": [
        results_tft[:,0].mean(),
        results_lstm[:,0].mean()
    ],
    "Accuracy SD": [
        results_tft[:,0].std(),
        results_lstm[:,0].std()
    ],

    "F1 Mean": [
        results_tft[:,1].mean(),
        results_lstm[:,1].mean()
    ],
    "F1 SD": [
        results_tft[:,1].std(),
        results_lstm[:,1].std()
    ],

    "Time Mean (sec)": [
        time_tft.mean(),
        time_lstm.mean()
    ],
    "Time SD (sec)": [
        time_tft.std(),
        time_lstm.std()
    ]
})

summary
