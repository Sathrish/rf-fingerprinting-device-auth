# RF Fingerprinting for Device Authentication

> **Status: work in progress.** These are my first experiments (built in Google Colab). The headline result comes from a random train/test split and has not yet been tested on unseen recording conditions. See [Limitations](#limitations-and-next-steps).

Passwords, certificates, keys and MAC addresses are all software. They can be copied or spoofed, so a cloned device can impersonate a real one. **RF fingerprinting** identifies a radio by the tiny, unavoidable imperfections in its hardware instead: IQ imbalance, carrier frequency offset, phase noise, DC offset and amplifier non-linearity. A cloned device with identical software still has different hardware, so it can be rejected.

This project trains deep-learning models to tell **16 USRP X310 transmitters** apart from their raw IQ signals.

## Data

- **Dataset:** ORACLE RF fingerprinting dataset, *KRI-16Devices-RawData* (Genesys Lab, Northeastern University): [genesys-lab.org/oracle](https://www.genesys-lab.org/oracle)
- **Contents:** 16 USRP X310 transmitters sending WiFi frames, recorded at 11 distances (2 ft to 62 ft), stored as complex128 IQ samples (`.sigmf-data`).
- The dataset (about 28 GB) is **not** stored in this repository. Download it and set `DATASET_PATH` in the notebooks.

## Method

1. Load raw IQ from each `.sigmf-data` file and cut it into fixed-length chunks.
2. **CNN input:** turn each chunk into a 2D spectrogram with the Short-Time Fourier Transform (STFT); shape 256 x 63 (frequency x time).
3. **LSTM input:** the normalised I and Q values as a time series.
4. Train a CNN and LSTM variants (PyTorch, Colab T4 GPU) to classify the 16 devices.
5. Stress-test with additive white Gaussian noise at different SNRs.

## Results so far

| Notebook | Data | Model | Test accuracy |
|---|---|---|---|
| `01_baseline_cnn_lstm_2ft` | 960 chunks, 2 ft only | CNN | 75.0% |
| | | LSTM | 6.25% (chance level) |
| `02_multi_distance_cnn_lstm` | 8,000 chunks, 5 distances (2–26 ft) | CNN | **98.3%** |
| | | LSTM (256 samples) | 26.9% |
| `03_improved_models_snr_robustness` | 8,000 chunks, 5 distances | CNN | 97.7% |
| | | LSTM (4,096 samples) | 48.1% |
| | | CNN + BiLSTM + attention | 53.4% |

Chance level for 16 devices is 6.25%. Test sets are a random, stratified 20% of chunks (1,600 chunks in notebooks 02 and 03).

**Noise robustness** (notebook 03, CNN trained on clean signals):

| SNR | 30 dB | 20 dB | 10 dB | 0 dB | -10 dB |
|---|---|---|---|---|---|
| CNN | 98.0% | 60.6% | 15.2% | 6.6% | 7.9% |
| LSTM | 36.6% | 35.9% | 28.3% | 8.9% | 6.5% |

The CNN is accurate on clean signals but **not robust to noise**: accuracy collapses below 20 dB SNR.

## Limitations and next steps

These are known weaknesses of the current results, written down so they can be fixed:

- **Optimistic split.** Chunks are split randomly, so pieces of the *same recording* appear in both training and test sets. This can inflate accuracy.
- **Model selection on the test set.** The "best" epoch was chosen using test accuracy; a separate validation set is needed.
- **No held-out condition yet.** Notebook 02 sets 62 ft aside as an unseen test distance, but that test has not been run.

Next steps:
1. Evaluate on a held-out distance (62 ft) and on held-out recording runs (train on run 1, test on run 2).
2. Add a validation split and report final numbers only on untouched test data.
3. Train with noise augmentation to improve robustness at low SNR.
4. Try an open-set / Siamese (similarity-based) model so the system can *reject unknown devices*, which is what real authentication needs, rather than only classifying 16 known ones.

## Repository layout

```
notebooks/
  01_baseline_cnn_lstm_2ft.ipynb             first CNN vs LSTM experiment (2 ft only)
  02_multi_distance_cnn_lstm.ipynb           CNN vs LSTM on 5 distances, full evaluation
  03_improved_models_snr_robustness.ipynb    improved models + noise (SNR) robustness test
docs/
  Project_proposal.pdf                       project motivation and design
```

## Running

The notebooks were written for Google Colab with the dataset on Google Drive. To run them, upload the dataset to Drive (or a local folder), update `DATASET_PATH`, and run the cells in order. Requirements: Python 3, NumPy, SciPy, PyTorch, scikit-learn, Matplotlib, Seaborn.

## Author

Sathrish S, B.Tech Computer and Communication Engineering, Amrita Vishwa Vidyapeetham, Coimbatore.
