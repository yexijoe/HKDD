# Towards Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification

## Datasets
In both datasets, the original bit sequence is chosen from 0 and 1 in a random manner to ensure that the probability of appearance for each symbol is equivalent. The length of each modulated signal is 1024 for dataset HKDD_AMC36 and 512 for dataset HKDD_AMC12. The oversampling rate is 8, so each sampled sequence in dataset HKDD_AMC36 contains 128 symbols and each sampled sequence in dataset HKDD_AMC12 contains 64 symbols. A root raised-cosine (RRC) filter with 6-symbols truncated length is employed as the pulse-shaping filter and the roll-off coefficient of RRC is randomly chosen within the range 0.2 to 0.7. The frequency offset is randomly chosen from -0.2 to 0.2 (normalized to the sampling frequency). The range of SNR is (-20 dB, 30 dB) for dataset HKDD_AMC36 and (-20 dB, 20 dB) for dataset HKDD_AMC12 with an interval of 2 dB. The number of training samples for each modulation type is 1000 in each SNR and the number of testing samples is half of the training samples. Both datasets contain both IQ signals and traditional features. HKDD_AMC36 is the dataset currently available that contains the most number of modulation categories.

We give an example of importing dataset through Python:
```
import h5py
train_data = h5py.File('HKDD_AMC12_train.mat')
data_raw = train_data['XTrainIQ']
data_feature = train_data['Feature']
```

- The shape of data_raw is (2, L, 1, N), 2 represents I and Q channels of each signal, L is the length of each modulated signal, N is the number of signal samples. The shape of data_feature is (228, N), 228 represents the number of different adopted traditional features.

## Access link of these two datasets
Coming soon.
