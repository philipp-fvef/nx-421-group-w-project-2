# Part 1


```python
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.ndimage import convolve1d
from sklearn.model_selection import KFold

import seaborn as sns
sns.set()


plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["figure.dpi"] = 120
```

## 1) Visualize and preprocess the data for subject 2 (use only the first exercise set:  S2_A1_E1 with 12 actions). Does the data look reasonable? Did you need to remove any trials?

a) load the data


```python
# Path to your parquet file
path = "data/s2/S2_A1_E1.parquet"

df = pd.read_parquet(path)

print(df.shape)
df.head()
```

    (100686, 36)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emg_0</th>
      <th>emg_1</th>
      <th>emg_2</th>
      <th>emg_3</th>
      <th>emg_4</th>
      <th>emg_5</th>
      <th>emg_6</th>
      <th>emg_7</th>
      <th>emg_8</th>
      <th>emg_9</th>
      <th>...</th>
      <th>glove_16</th>
      <th>glove_17</th>
      <th>glove_18</th>
      <th>glove_19</th>
      <th>glove_20</th>
      <th>glove_21</th>
      <th>repetition_0</th>
      <th>rerepetition_0</th>
      <th>stimulus_0</th>
      <th>restimulus_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0269</td>
      <td>0.1001</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>...</td>
      <td>108.0</td>
      <td>94.0</td>
      <td>153.0</td>
      <td>136.0</td>
      <td>145.0</td>
      <td>107.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0269</td>
      <td>0.0757</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>...</td>
      <td>108.0</td>
      <td>94.0</td>
      <td>153.0</td>
      <td>136.0</td>
      <td>145.0</td>
      <td>107.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0317</td>
      <td>0.0586</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>...</td>
      <td>108.0</td>
      <td>94.0</td>
      <td>153.0</td>
      <td>136.0</td>
      <td>145.0</td>
      <td>107.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0293</td>
      <td>0.0391</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>...</td>
      <td>108.0</td>
      <td>94.0</td>
      <td>153.0</td>
      <td>136.0</td>
      <td>145.0</td>
      <td>107.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0293</td>
      <td>0.0269</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0049</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>0.0024</td>
      <td>...</td>
      <td>108.0</td>
      <td>94.0</td>
      <td>153.0</td>
      <td>136.0</td>
      <td>145.0</td>
      <td>107.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
print(df.columns.tolist())
```

    ['emg_0', 'emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6', 'emg_7', 'emg_8', 'emg_9', 'glove_0', 'glove_1', 'glove_2', 'glove_3', 'glove_4', 'glove_5', 'glove_6', 'glove_7', 'glove_8', 'glove_9', 'glove_10', 'glove_11', 'glove_12', 'glove_13', 'glove_14', 'glove_15', 'glove_16', 'glove_17', 'glove_18', 'glove_19', 'glove_20', 'glove_21', 'repetition_0', 'rerepetition_0', 'stimulus_0', 'restimulus_0']
    

b) Preprocess data


```python
emg_cols   = [c for c in df.columns if c.startswith("emg_")]
glove_cols = [c for c in df.columns if c.startswith("glove_")]

emg        = df[emg_cols].to_numpy()
glove      = df[glove_cols].to_numpy()

stimulus      = df["stimulus_0"].to_numpy().astype(int)
restimulus    = df["restimulus_0"].to_numpy().astype(int)
repetition    = df["repetition_0"].to_numpy().astype(int)
rerepetition  = df["rerepetition_0"].to_numpy().astype(int)

n_samples, n_channels = emg.shape
print(f"EMG shape: {emg.shape}, glove shape: {glove.shape}")

```

    EMG shape: (100686, 10), glove shape: (100686, 22)
    


```python
fs = 100 
t = np.arange(n_samples) / fs
print(f"Approx recording duration: {t[-1]:.1f} s")
```

    Approx recording duration: 1006.9 s
    

## 1. Quick sanity plots (raw EMG + glove)

### 1a. EMG channels over a short window


```python
# pick a 5 s window somewhere in the middle
window_duration = 5  # seconds
start_time = 20      # seconds (you can change this)
start_idx = int(start_time * fs)
end_idx   = int((start_time + window_duration) * fs)
end_idx   = min(end_idx, n_samples)

segment_t   = t[start_idx:end_idx]
segment_emg = emg[start_idx:end_idx, :]

fig, ax = plt.subplots(5, 2, sharex=True, figsize=(12, 8), constrained_layout=True)
ax = ax.ravel()

for ch in range(n_channels):
    ax[ch].plot(segment_t, segment_emg[:, ch])
    ax[ch].set_title(f"Raw EMG channel {ch}")
    if ch >= 8:
        ax[ch].set_xlabel("Time [s]")

fig.suptitle("Raw EMG, 5 s window", y=1.02)

```




    Text(0.5, 1.02, 'Raw EMG, 5 s window')




    
![png](output_10_1.png)
    


### 1b. A few glove sensors in the same window


```python
segment_glove = glove[start_idx:end_idx, :]

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(12, 6), constrained_layout=True)

for i, gch in enumerate([0, 5, 10, 15]):
    ax[i].plot(segment_t, segment_glove[:, gch])
    ax[i].set_title(f"Glove sensor {gch}")
ax[-1].set_xlabel("Time [s]")

fig.suptitle("Glove signals, same 5 s window", y=1.02)

```




    Text(0.5, 1.02, 'Glove signals, same 5 s window')




    
![png](output_12_1.png)
    


## 2. What preprocessing is already done for this dataset?

Otto Bock electrodes:

1) amplified (14'000)
2) band-pass filtered (range: TODO)
3) RMS-recitfied EMG 
4) internal shielding & filtering against 50-60 Hz power supply, mobile phones, etc.  

### 2a. Check for 50 Hz artifacts


```python
from scipy.signal import welch

channel_for_psd = 0  # you can change this

f, Pxx = welch(emg[:, channel_for_psd], fs=fs, nperseg=len(emg))

fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# Full band 0–50 Hz (fs/2)
ax[0].semilogy(f, Pxx)
ax[0].set_xlim(0, fs / 2)
ax[0].set_xlabel("Frequency [Hz]")
ax[0].set_ylabel("PSD (log)")
ax[0].set_title(f"PSD EMG channel {channel_for_psd}")

# Zoom around 50 Hz
ax[1].semilogy(f, Pxx)
ax[1].set_xlim(40, 60)
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel("PSD (log)")
ax[1].set_title("Zoom around 50 Hz (powerline)")

fig.suptitle("Power spectral density – check for 50 Hz artifact", y=1.05)

```




    Text(0.5, 1.05, 'Power spectral density – check for 50 Hz artifact')




    
![png](output_15_1.png)
    


### 2b. Check that it's rectified (can already see above)


```python
print("Global min / max per channel:")
print(np.min(emg, axis=0))
print(np.max(emg, axis=0))

plt.figure(figsize=(6,4))
plt.hist(emg[:, 0], bins=50)
plt.title("Histogram of EMG channel 0")
plt.xlabel("Amplitude")
plt.ylabel("Count")
plt.show()

```

    Global min / max per channel:
    [0.0024 0.0024 0.0024 0.0024 0.0024 0.     0.     0.0024 0.     0.0024]
    [3.8647 2.1606 1.3599 2.9883 0.5298 0.6421 0.5298 2.8076 4.6606 2.0508]
    


    
![png](output_17_1.png)
    


### 3. Understand trial structure


```python
unique_restim = np.unique(restimulus)
print("Unique restimulus labels:", unique_restim)

actions = unique_restim[unique_restim > 0]  # exclude rest (0)
print(f"Number of actions (excluding rest): {len(actions)}")

unique_reps = np.unique(rerepetition)
unique_reps = unique_reps[unique_reps > 0]
print("Unique rerepetition labels:", unique_reps)
print(f"Number of repetitions (excluding 0): {len(unique_reps)}")

n_stimuli     = len(actions)
n_repetitions = len(unique_reps)
samples_per_trial = np.zeros((n_stimuli, n_repetitions), dtype=int)

for i, a in enumerate(actions):
    for j, r in enumerate(unique_reps):
        idx = (restimulus == a) & (rerepetition == r)
        samples_per_trial[i, j] = idx.sum()

samples_per_trial


```

    Unique restimulus labels: [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
    Number of actions (excluding rest): 12
    Unique rerepetition labels: [ 1  2  3  4  5  6  7  8  9 10]
    Number of repetitions (excluding 0): 10
    




    array([[307, 390, 262, 226, 418, 204, 196, 270, 306, 489],
           [334, 323, 344, 309, 333, 376, 362, 307, 348, 339],
           [465, 352, 450, 460, 436, 389, 323, 388, 383, 316],
           [310, 402, 357, 396, 255, 333, 263, 369, 438, 471],
           [516, 382, 425, 379, 463, 411, 387, 343, 286, 377],
           [337, 319, 337, 368, 338, 336, 268, 354, 319, 313],
           [428, 392, 341, 393, 421, 445, 342, 328, 329, 358],
           [289, 372, 406, 397, 498, 336, 415, 328, 472, 405],
           [384, 424, 371, 316, 280, 314, 342, 284, 307, 421],
           [311, 296, 299, 296, 301, 246, 213, 317, 493, 221],
           [352, 403, 338, 416, 330, 355, 313, 322, 329, 219],
           [270, 330, 300, 387, 295, 204, 363, 267, 281, 329]])



Looks correct (trial length do differ a lot but I hope that's fine)

### 4a. Visual trial inspection


```python
def plot_trial_emg(action_id, rep_id, signal=None, title_prefix="EMG"):
    if signal is None:
        signal = emg
        
    idx = (restimulus == action_id) & (rerepetition == rep_id)
    trial = signal[idx, :]
    if trial.size == 0:
        print("No samples for this (action, repetition).")
        return
    
    n_ch = trial.shape[1]
    rows, cols = 5, 2  # for 10 channels
    
    fig, ax = plt.subplots(rows, cols, figsize=(12, 8), constrained_layout=True, sharex=True)
    ax = ax.ravel()
    
    for ch in range(n_ch):
        ax[ch].plot(trial[:, ch])
        ax[ch].set_title(f"ch {ch}")
    for k in range(n_ch, rows * cols):
        ax[k].axis("off")
    
    fig.suptitle(f"{title_prefix} – action {action_id}, repetition {rep_id}", y=1.02)

```

Manually check a few

actions[0]: first non-zero restimulus label (first of 12 movements)

unique_reps[0]: repetition 1


```python
# Example: first action, first repetition
plot_trial_emg(action_id=actions[7], rep_id=unique_reps[5])

```


    
![png](output_24_0.png)
    


### 4b. Visual trial inspection (same scale)


```python
def plot_trial_emg_same_scale(action_id, rep_id, signal=None, title_prefix="EMG"):
    if signal is None:
        signal = emg

    # Select trial samples
    idx = (restimulus == action_id) & (rerepetition == rep_id)
    trial = signal[idx, :]
    if trial.size == 0:
        print("No samples for this (action, repetition).")
        return

    n_ch = trial.shape[1]
    rows, cols = 5, 2

    # Compute global y-limits across all channels
    y_min = trial.min()
    y_max = trial.max()

    fig, ax = plt.subplots(rows, cols, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()

    for ch in range(n_ch):
        ax[ch].plot(trial[:, ch])
        ax[ch].set_title(f"ch {ch}")
        ax[ch].set_ylim(y_min, y_max)  # same scale for all plots

    # Hide unused subplots (for generality)
    for k in range(n_ch, rows * cols):
        ax[k].axis("off")

    fig.suptitle(f"{title_prefix} (same y-scale) – action {action_id}, repetition {rep_id}", y=1.02)

```


```python
plot_trial_emg_same_scale(action_id=actions[1], rep_id=unique_reps[0])
```


    
![png](output_27_0.png)
    


Check the last one (maybe some sensor lost contact)


```python
plot_trial_emg_same_scale(action_id=actions[11], rep_id=unique_reps[9])
```


    
![png](output_29_0.png)
    


Doing this manually is kind of tedious and I ain't doing that

12 actions, 10 repetitions --> 120 plots 

### 5a. Automatic detection


```python
def find_bad_trials(
    signal, 
    restimulus, 
    rerepetition, 
    actions, 
    reps, 
    rms_drop_fraction=0.1, 
    min_std=1e-7
):
    """
    Flag (action, repetition) trials where one or more channels are basically 'dead':
      - very low RMS compared to typical (global RMS), and/or
      - almost zero variance.

    Parameters
    ----------
    signal : np.ndarray
        EMG array of shape (N_samples, N_channels).
    restimulus, rerepetition : np.ndarray
        Label vectors.
    actions, reps : array-like
        Unique action and repetition labels (exclude 0).
    rms_drop_fraction : float
        Trial RMS must be at least this fraction of global RMS, otherwise it's suspicious.
    min_std : float
        Absolute minimum standard deviation to avoid being considered 'flat'.

    Returns
    -------
    bad_trials : list of dict
        Each dict has keys: 'action', 'repetition', 'bad_channels', 'trial_length'.
    """
    bad_trials = []

    # Global RMS per channel over the full recording
    global_rms = np.sqrt(np.mean(signal**2, axis=0))

    for a in actions:
        for r in reps:
            idx = (restimulus == a) & (rerepetition == r)
            trial = signal[idx, :]
            if trial.size == 0:
                continue

            # Trial-level RMS and std per channel
            trial_rms = np.sqrt(np.mean(trial**2, axis=0))
            trial_std = trial.std(axis=0)

            # Conditions for 'bad' channels
            low_amp = trial_rms < (rms_drop_fraction * global_rms)
            flat    = trial_std < min_std

            bad_ch = np.where(low_amp | flat)[0]

            if len(bad_ch) > 0:
                bad_trials.append({
                    "action": int(a),
                    "repetition": int(r),
                    "bad_channels": bad_ch.tolist(),
                    "trial_length": int(trial.shape[0])
                })

    return bad_trials

```


```python
bad_trials = find_bad_trials(
    signal=emg,
    restimulus=restimulus,
    rerepetition=rerepetition,
    actions=actions,
    reps=unique_reps,
    rms_drop_fraction=0.2,  # ~10% of normal RMS
    min_std=1e-7          # almost flat
)

print(f"Number of suspicious trials: {len(bad_trials)}")
bad_trials[:40]  # show first few

```

    Number of suspicious trials: 37
    




    [{'action': 2, 'repetition': 1, 'bad_channels': [4, 6], 'trial_length': 334},
     {'action': 2, 'repetition': 2, 'bad_channels': [4], 'trial_length': 323},
     {'action': 2,
      'repetition': 3,
      'bad_channels': [4, 5, 6],
      'trial_length': 344},
     {'action': 2,
      'repetition': 4,
      'bad_channels': [4, 5, 6],
      'trial_length': 309},
     {'action': 2, 'repetition': 5, 'bad_channels': [4, 6], 'trial_length': 333},
     {'action': 2,
      'repetition': 6,
      'bad_channels': [4, 5, 6],
      'trial_length': 376},
     {'action': 2,
      'repetition': 7,
      'bad_channels': [4, 5, 6],
      'trial_length': 362},
     {'action': 2,
      'repetition': 8,
      'bad_channels': [4, 5, 6],
      'trial_length': 307},
     {'action': 2,
      'repetition': 9,
      'bad_channels': [4, 5, 6],
      'trial_length': 348},
     {'action': 2,
      'repetition': 10,
      'bad_channels': [4, 5, 6],
      'trial_length': 339},
     {'action': 3, 'repetition': 8, 'bad_channels': [6], 'trial_length': 388},
     {'action': 3, 'repetition': 9, 'bad_channels': [6], 'trial_length': 383},
     {'action': 3, 'repetition': 10, 'bad_channels': [1, 6], 'trial_length': 316},
     {'action': 4, 'repetition': 5, 'bad_channels': [4], 'trial_length': 255},
     {'action': 4, 'repetition': 10, 'bad_channels': [4], 'trial_length': 471},
     {'action': 5, 'repetition': 6, 'bad_channels': [5, 9], 'trial_length': 411},
     {'action': 6, 'repetition': 3, 'bad_channels': [5], 'trial_length': 337},
     {'action': 6, 'repetition': 4, 'bad_channels': [5], 'trial_length': 368},
     {'action': 6, 'repetition': 6, 'bad_channels': [5], 'trial_length': 336},
     {'action': 8,
      'repetition': 1,
      'bad_channels': [2, 4, 5],
      'trial_length': 289},
     {'action': 8,
      'repetition': 2,
      'bad_channels': [3, 4, 5],
      'trial_length': 372},
     {'action': 8,
      'repetition': 3,
      'bad_channels': [3, 4, 5, 6, 8],
      'trial_length': 406},
     {'action': 8, 'repetition': 4, 'bad_channels': [4, 5], 'trial_length': 397},
     {'action': 8, 'repetition': 6, 'bad_channels': [4, 5], 'trial_length': 336},
     {'action': 8,
      'repetition': 7,
      'bad_channels': [2, 3, 5, 6, 8],
      'trial_length': 415},
     {'action': 8,
      'repetition': 8,
      'bad_channels': [3, 5, 6, 8],
      'trial_length': 328},
     {'action': 8, 'repetition': 9, 'bad_channels': [6], 'trial_length': 472},
     {'action': 8, 'repetition': 10, 'bad_channels': [4, 6], 'trial_length': 405},
     {'action': 10, 'repetition': 1, 'bad_channels': [5], 'trial_length': 311},
     {'action': 10, 'repetition': 2, 'bad_channels': [3], 'trial_length': 296},
     {'action': 10, 'repetition': 3, 'bad_channels': [3], 'trial_length': 299},
     {'action': 10, 'repetition': 4, 'bad_channels': [3], 'trial_length': 296},
     {'action': 10, 'repetition': 6, 'bad_channels': [3], 'trial_length': 246},
     {'action': 10, 'repetition': 7, 'bad_channels': [3], 'trial_length': 213},
     {'action': 10, 'repetition': 8, 'bad_channels': [3], 'trial_length': 317},
     {'action': 10,
      'repetition': 9,
      'bad_channels': [3, 4, 8],
      'trial_length': 493},
     {'action': 10, 'repetition': 10, 'bad_channels': [3, 4], 'trial_length': 221}]




```python
plot_trial_emg_same_scale(action_id=actions[7], rep_id=unique_reps[5])
```


    
![png](output_34_0.png)
    


### 5b. Exclude those bad channels (only that action of course)


```python
# 1) Turn bad_trials into a set of (action, repetition) pairs
trials_to_drop = {(b["action"], b["repetition"]) for b in bad_trials}
print("Trials to drop (action, repetition):", sorted(trials_to_drop))

# 2) Build a sample-wise mask: keep everything *except* those trials
def build_keep_mask(restimulus, rerepetition, trials_to_drop):
    mask = np.ones_like(restimulus, dtype=bool)
    for (a, r) in trials_to_drop:
        mask &= ~((restimulus == a) & (rerepetition == r))
    return mask

keep_mask = build_keep_mask(restimulus, rerepetition, trials_to_drop)

print("Original samples:", len(df))
print("Samples after dropping:", keep_mask.sum())

# 3) New cleaned DataFrame
df_clean = df[keep_mask].reset_index(drop=True)

# 4) Re-derive cleaned arrays from df_clean
emg_clean   = df_clean[emg_cols].to_numpy()
glove_clean = df_clean[glove_cols].to_numpy()

restimulus_clean   = df_clean["restimulus_0"].to_numpy().astype(int)
rerepetition_clean = df_clean["rerepetition_0"].to_numpy().astype(int)

print("Clean EMG shape:", emg_clean.shape)
print("Clean restimulus unique:", np.unique(restimulus_clean))
print("Clean rerepetition unique:", np.unique(rerepetition_clean))


```

    Trials to drop (action, repetition): [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 8), (3, 9), (3, 10), (4, 5), (4, 10), (5, 6), (6, 3), (6, 4), (6, 6), (8, 1), (8, 2), (8, 3), (8, 4), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (10, 1), (10, 2), (10, 3), (10, 4), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]
    Original samples: 100686
    Samples after dropping: 87934
    Clean EMG shape: (87934, 10)
    Clean restimulus unique: [ 0  1  3  4  5  6  7  8  9 10 11 12]
    Clean rerepetition unique: [ 0  1  2  3  4  5  6  7  8  9 10]
    

### 6. Envelopes


```python
# Number of stimuli and repetitions in the CLEANED labels
actions_clean = np.unique(restimulus_clean)
actions_clean = actions_clean[actions_clean > 0]  # exclude rest

reps_clean = np.unique(rerepetition_clean)
reps_clean = reps_clean[reps_clean > 0]          # exclude 0

n_stimuli_clean     = len(actions_clean)
n_repetitions_clean = len(reps_clean)
n_channels          = emg_clean.shape[1]

print("n_stimuli_clean:", n_stimuli_clean)
print("n_repetitions_clean:", n_repetitions_clean)

# Moving-average window (same as Exercise 10b)
mov_mean_length  = 25
mov_mean_weights = np.ones(mov_mean_length) / mov_mean_length

# Nested lists: [stimulus_idx][repetition_idx]
emg_windows_clean   = [[None for _ in range(n_repetitions_clean)] for _ in range(n_stimuli_clean)]
emg_envelopes_clean = [[None for _ in range(n_repetitions_clean)] for _ in range(n_stimuli_clean)]

for s_idx, a in enumerate(actions_clean):
    for r_idx, r in enumerate(reps_clean):
        # select this trial in the CLEANED data
        idx = (restimulus_clean == a) & (rerepetition_clean == r)
        trial = emg_clean[idx, :]
        if trial.size == 0:
            continue

        emg_windows_clean[s_idx][r_idx]   = trial
        emg_envelopes_clean[s_idx][r_idx] = convolve1d(trial, mov_mean_weights, axis=0)

```

    n_stimuli_clean: 11
    n_repetitions_clean: 10
    

Quickly look at envelope for one trial


```python
example_s_idx = 0  # first stimulus in actions_clean
example_r_idx = 0  # first repetition in reps_clean

trial_raw  = emg_windows_clean[example_s_idx][example_r_idx]
trial_env  = emg_envelopes_clean[example_s_idx][example_r_idx]

fig, ax = plt.subplots(2, 5, figsize=(12, 6), constrained_layout=True)
ax = ax.ravel()

for ch in range(n_channels):
    ax[ch].plot(trial_raw[:, ch])
    ax[ch].set_title(f"Raw ch {ch}")
plt.suptitle(f"Rectified EMG – stimulus {actions_clean[example_s_idx]}, repetition {reps_clean[example_r_idx]}")

fig, ax = plt.subplots(2, 5, figsize=(12, 6), constrained_layout=True)
ax = ax.ravel()
for ch in range(n_channels):
    ax[ch].plot(trial_env[:, ch])
    ax[ch].set_title(f"Env ch {ch}")
plt.suptitle("Envelopes of the EMG signal")

```




    Text(0.5, 0.98, 'Envelopes of the EMG signal')




    
![png](output_40_1.png)
    



    
![png](output_40_2.png)
    


### 7 Average activations + heatmaps


```python
import seaborn as sns
sns.set()

# (channels, stimuli, repetitions)
emg_average_activations_clean = np.full(
    (n_channels, n_stimuli_clean, n_repetitions_clean),
    np.nan
)

for s_idx in range(n_stimuli_clean):
    for r_idx in range(n_repetitions_clean):
        env = emg_envelopes_clean[s_idx][r_idx]
        if env is None:
            continue
        emg_average_activations_clean[:, s_idx, r_idx] = np.mean(env, axis=0)

```


```python
fig, ax = plt.subplots(4, 3, figsize=(10, 6), constrained_layout=True, sharex=True, sharey=True)
ax = ax.ravel()

for s_idx in range(n_stimuli_clean):
    sns.heatmap(
        emg_average_activations_clean[:, s_idx, :],
        ax=ax[s_idx],
        xticklabels=False,
        yticklabels=False,
        cbar=True
    )
    ax[s_idx].set_title(f"Stimulus {actions_clean[s_idx]}")
    ax[s_idx].set_xlabel("Repetition")
    ax[s_idx].set_ylabel("EMG channel")

```


    
![png](output_43_0.png)
    


TODO: only one repetition bad is weird, no?

## 2. Split the data into training, validation, and testing sets for the subject. Why do we need the different datasets? 

Things to keep in mind

- Stratified (The split should be stratified, meaning that all classes (movement labels) appear in train/validation/test in similar proportions.)

- We will fix a random seed to make the split reproducible.

- assign whole repetitions to each set, not individual windows (DATA LEAKAGE!!)

TODO: double check this (is that what they write in the paper?)

Let's split the following way:

2 repetitions (3 and 7) for testing --> 8 left for validation and training --> 4 fold cross validation

TODO do we have to be careful because we removed some channels for certain stimuli?


```python
# For reproducibility when we shuffle reps later
np.random.seed(42)

# Unique repetitions in the CLEANED data (excluding 0)
all_reps = np.unique(rerepetition_clean)
all_reps = all_reps[all_reps > 0]

print("All repetitions in cleaned data:", all_reps)

# --- Choose which repetitions are test ---
# You can change [3, 7] if you prefer different test reps,
# but keep them fixed once chosen.
test_reps = np.array([3, 7])

# Training/validation repetitions are all others
train_reps = np.array([r for r in all_reps if r not in test_reps])

print("Train repetitions:", train_reps)
print("Test repetitions: ", test_reps)

# --- Sample-wise masks (for later use) ---
train_mask = np.isin(rerepetition_clean, train_reps)
test_mask  = np.isin(rerepetition_clean, test_reps)

print("Samples in train reps:", train_mask.sum())
print("Samples in test reps: ", test_mask.sum())

```

    All repetitions in cleaned data: [ 1  2  3  4  5  6  7  8  9 10]
    Train repetitions: [ 1  2  4  5  6  8  9 10]
    Test repetitions:  [3 7]
    Samples in train reps: 23398
    Samples in test reps:  5641
    


```python
# Optional: train/test views of the raw signals (per sample)
emg_train = emg_clean[train_mask]
emg_test  = emg_clean[test_mask]

restimulus_train = restimulus_clean[train_mask]
restimulus_test  = restimulus_clean[test_mask]

rerepetition_train = rerepetition_clean[train_mask]
rerepetition_test  = rerepetition_clean[test_mask]

```


```python
unique_train_reps = np.unique(train_reps)
print("Unique train reps:", unique_train_reps)

# 4-fold CV over train repetitions
kf = KFold(n_splits=4, shuffle=True, random_state=42)

# We'll store folds as lists of repetition IDs (not sample indices yet)
cv_rep_splits = []

for fold_idx, (rep_tr_idx, rep_val_idx) in enumerate(kf.split(unique_train_reps)):
    reps_in_train_fold = unique_train_reps[rep_tr_idx]
    reps_in_val_fold   = unique_train_reps[rep_val_idx]

    cv_rep_splits.append((reps_in_train_fold, reps_in_val_fold))

    print(f"Fold {fold_idx}:")
    print("  train reps:", reps_in_train_fold)
    print("  val reps  :", reps_in_val_fold)

```

    Unique train reps: [ 1  2  4  5  6  8  9 10]
    Fold 0:
      train reps: [ 1  4  5  6  9 10]
      val reps  : [2 8]
    Fold 1:
      train reps: [2 4 5 6 8 9]
      val reps  : [ 1 10]
    Fold 2:
      train reps: [ 1  2  5  8  9 10]
      val reps  : [4 6]
    Fold 3:
      train reps: [ 1  2  4  6  8 10]
      val reps  : [5 9]
    

### Why we need all of that (ChatGPT) TODO

### ✅ Why do we need different datasets?

- **Training set** – used to fit the model.  
- **Validation set** – used for hyperparameter tuning and monitoring overfitting.  
- **Test set** – kept completely unseen; provides an unbiased estimate of generalization.

- **Repetition-wise splitting is essential for EMG:**  
  - Samples from the same repetition are highly correlated.  
  - Mixing repetitions across train/test causes information leakage and inflated accuracy.

- **A single split is not reliable:**  
  - Performance may vary depending on which repetitions land in each set.  
  - Cross-validation reduces this randomness.

---

### ✅ Why use cross-validation?

- Splits the training set into *k* folds.  
- Trains on *k−1* folds and validates on the remaining fold.  
- Repeats this *k* times and averages performance.

**Benefits:**

- More stable and trustworthy evaluation.  
- Helps detect overfitting.  
- Enables systematic hyperparameter tuning.  
- **Repetition-wise CV** ensures each repetition stays in exactly one fold — the correct approach for EMG data.


## 3. Extract features from the trials (at least 5 different ones), explain briefly why you picked these features. Visualize the typical values of  the features across the different trials. What do you see? Are the values similar  between repetitions and between channels? Explain the possible reasons for  similarity/discrepancies.  

### Challenge:

How to decide which features to use: see Feature importance/selection section in exercise 11

--> [mutual_info_classif](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_selection.mutual_info_classif.html) and [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) from sklearn

Mutual_info_classif evaluates the mutual information between each feature and the target variable, providing an insight into the relevance of each feature. 

Meanwhile, SelectKBest allows us to select a specified number of features that have the highest scores according to a given scoring function, in this case, the mutual information.  
By combining these two functions, we can effectively reduce the feature space to those most relevant.

In exercise 11 they suggest to use frequency domain features --> find explanation why that's a good idea TODO

For more information, see (from exercise 11): 
PDFs on github

1. [Real-Time EMG Based Pattern Recognition Control for Hand Prostheses: A Review on Existing Methods, Challenges and Future Implementation](https://doi.org/10.3390/s19204596)
2. [Evaluation of the forearm EMG signal features for the control of a prosthetic hand](https://doi.org/10.1088/0967-3334/24/2/307)


1. "The results indicate that the energy of wavelet coefficients in nine scales and cepstrum
coefficients, which were selected in the evaluation procedure, lead to the best features"

2. "Hudgins [34] proposed the four
 different time-domain features (MAV, WL, ZC, SSC) [35] for feature extraction from EMG, and it is
 the most adopted one to date in the field of myoelectric pattern recognition"

In exercise 11 they use:

- MAV (Mean absolute value)

- WL (Waveform length)

- SSC (Slope sign changes)

- RMS (root mean square)

- STD (standard deviation)

- MAXAV (maximum absolute value)

### For now only time domain features

Note: apparently the two features from the first study are more difficult to implement. Could try that later or first try total spectral power or mean/median frequency TODO


```python
# ==== Time-domain feature functions (per trial) ====
# Input:  trial array x with shape (T, n_channels)
# Output: 1D array of length n_channels (one value per channel)

import numpy as np

def feat_mav(x):
    """Mean Absolute Value (MAV)"""
    return np.mean(np.abs(x), axis=0)

def feat_rms(x):
    """Root Mean Square (RMS)"""
    return np.sqrt(np.mean(x**2, axis=0))

def feat_wl(x):
    """Waveform Length (WL)"""
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def feat_zc(x, thr=0.01):
    """
    Zero Crossings (ZC) with threshold.
    Counts sign changes where the amplitude change is larger than 'thr'.
    """
    x1 = x[:-1, :]
    x2 = x[1:, :]
    sign_change = (x1 * x2) < 0
    big_enough  = np.abs(x1 - x2) > thr
    return np.sum(sign_change & big_enough, axis=0)

def feat_ssc(x, thr=0.01):
    """
    Slope Sign Changes (SSC) with threshold.
    Counts changes in slope direction where at least one of the slopes is larger than 'thr'.
    """
    diff1 = x[1:-1, :] - x[:-2, :]
    diff2 = x[2:, :]  - x[1:-1, :]
    sign_change = (diff1 * diff2) < 0
    big_enough  = (np.abs(diff1) > thr) | (np.abs(diff2) > thr)
    return np.sum(sign_change & big_enough, axis=0)

def feat_std(x):
    """Standard Deviation (STD)"""
    return np.std(x, axis=0)

def feat_maxav(x):
    """Maximum Absolute Value (MAXAV)"""
    return np.max(np.abs(x), axis=0)


# === Collect all features in a list (order matters) ===
feature_fns = [
    feat_mav,
    feat_rms,
    feat_wl,
    feat_zc,
    feat_ssc,
    feat_std,
    feat_maxav,
]

feature_names = [
    "MAV",
    "RMS",
    "WL",
    "ZC",
    "SSC",
    "STD",
    "MAXAV",
]

n_features = len(feature_fns)
print("Using", n_features, "time-domain features:", feature_names)
```

    Using 7 time-domain features: ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'STD', 'MAXAV']
    


```python
# ==== Build feature dataset from emg_envelopes_clean ====

def build_feature_dataset(emg_trials, actions_clean, reps_clean, feature_fns, feature_names):
    """
    Build a feature matrix with one row per trial.

    Parameters
    ----------
    emg_trials    : nested list [stimulus_idx][repetition_idx] -> trial array (T, n_channels)
    actions_clean : 1D array of movement labels used in emg_trials
    reps_clean    : 1D array of repetition IDs used in emg_trials
    feature_fns   : list of functions, each f(trial) -> (n_channels,)
    feature_names : list of feature names (same length as feature_fns)

    Returns
    -------
    X_features        : (n_trials, n_features * n_channels)
    y_features        : (n_trials,) movement label per trial
    rep_ids_features  : (n_trials,) repetition ID per trial
    """
    rows = []
    y_list = []
    rep_list = []

    for s_idx, a in enumerate(actions_clean):
        for r_idx, r in enumerate(reps_clean):
            trial = emg_trials[s_idx][r_idx]
            if trial is None:
                continue  # some (a, r) were dropped

            # compute all features for this trial
            feat_per_trial = []
            for f in feature_fns:
                feat_vals = f(trial)                       # shape (n_channels,)
                feat_per_trial.append(feat_vals)

            feat_vec = np.concatenate(feat_per_trial)      # shape (n_features * n_channels,)
            rows.append(feat_vec)
            y_list.append(a)
            rep_list.append(r)

    X_features       = np.vstack(rows)
    y_features       = np.array(y_list)
    rep_ids_features = np.array(rep_list)

    return X_features, y_features, rep_ids_features


# --- Actually build the dataset using the ENVELOPES ---
X_features, y_features, rep_ids_features = build_feature_dataset(
    emg_trials=emg_envelopes_clean,
    actions_clean=actions_clean,
    reps_clean=reps_clean,
    feature_fns=feature_fns,
    feature_names=feature_names,
)

print("Feature matrix shape:", X_features.shape)  # (n_trials, n_features * n_channels)
print("Labels shape:", y_features.shape)
print("Repetition IDs shape:", rep_ids_features.shape)

```

    Feature matrix shape: (83, 70)
    Labels shape: (83,)
    Repetition IDs shape: (83,)
    

Visualize


```python
n_trials = X_features.shape[0]
n_channels = emg_clean.shape[1]
n_feat = len(feature_names)

# reshape: (trials, features, channels)
features_3d = X_features.reshape(n_trials, n_feat, n_channels)

# ---- 3a) Heatmaps: each feature separately ----
fig, axes = plt.subplots(1, n_feat, figsize=(4*n_feat, 4), constrained_layout=True)

if n_feat == 1:
    axes = [axes]

for fi, fname in enumerate(feature_names):
    # rows = channels, columns = trials
    data = features_3d[:, fi, :].T  # shape (channels, trials)
    sns.heatmap(data, ax=axes[fi], cmap="viridis", cbar=True)
    axes[fi].set_title(fname)
    axes[fi].set_xlabel("Trial index")
    axes[fi].set_ylabel("Channel")

plt.show()

# ---- 3b) Example boxplot: one feature, one channel, across repetitions ----
# Let's pick MAV of channel 0 as a simple example

# extract MAV block: first feature in feature_names -> index 0
mav_block = features_3d[:, feature_names.index("MAV"), :]  # shape (trials, channels)
mav_ch0   = mav_block[:, 0]

import pandas as pd

df_feat = pd.DataFrame({
    "MAV_ch0": mav_ch0,
    "action": y_features,
    "repetition": rep_ids_features,
})

plt.figure(figsize=(8, 4))
sns.boxplot(data=df_feat, x="repetition", y="MAV_ch0")
plt.title("MAV of channel 0 across repetitions")
plt.xlabel("Repetition")
plt.ylabel("MAV (channel 0)")
plt.show()

```


    
![png](output_60_0.png)
    



    
![png](output_60_1.png)
    


Note: ZC is only 0 of course as we don't have 0 crossings for a rectified signal

## 4. Perform classification on the data of subject 2. Predict the action of the subject based  on the EMG signals. Use hyperparameter optimization to increase your models’ performance (Gradient boosting) 



## 5. Evaluate the performance using a metric of your choice. Justify why the metric is  suitable for this task and whether the performance is satisfactory.

In exercise 11 the print accuracy and show the confusion matrix 

TODO: check papers for best metric

## 6. Perform feature selection / dimension reduction using 2 methods of your choice that you think might perform better or yield insights.  Evaluate the performance using the same metric as point 5. Is there an improvement  in the performance and why do you think this is the case?
