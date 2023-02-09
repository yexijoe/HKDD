# Towards Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification

## About paper
Title: Towards Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification.

Author: Shilian Zheng, Xiaoyu Zhou, Luxin Zhang, Peihan Qi, Kunfeng Qiu, Jiawei Zhu, and Xiaoniu Yang.

Journal: IEEE Transactions on Cognitive Communications and Networking.

Abstract: Automatic modulation classification (AMC) can generally be divided into knowledge-based methods and data-driven methods. In this paper, we explore combining the knowledgebased method and data-driven technology to take full advantage of both and propose a hybrid knowledge and data-driven deep learning framework (HKDD) for AMC. To make the handcrafted features more discriminative, various traditional features are adopted, including instantaneous features, statistical features, and spectral features. In the HKDD framework, a feature fusion mechanism is proposed to integrate the features learned from the original signal with those processed by a fully connected network from the handcrafted features. Besides, an attention mechanism is implemented on the fused features to neglect immature features and highlight important features. To evaluate the performance of the proposed method, we construct two modulation classification datasets containing both traditional features and raw IQ data. The bigger one contains 36 modulation categories, which is greater than the number of categories of any AMC dataset currently available. Simulation results show that our proposed method has significant performance gain in both adequate-sample classification scenario and few-shot classification scenario.

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

| feature | location | feature | location | feature | location | feature | location | feature | location | feature | location |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $K$ | 0 | $F_3^8$ | 19 | $r(n)-M_{12,2}$ | 38 | $r(n)-C_{6,2}$ | 57 | $z(n,2)-M_{4,0}$ | 76 | $z(n,2)-M_{16,0}$ | 95 | $z(n,2)-C_{8,3}$ | 114 | $z(n,4)-M_{6,2}$ | 133 | $z(n,4)-M_{16,5}$ | 152 | $z(n,4)-\\widehat C_{8,0}$ | 171 | $z(n,8)-M_{8,4}$ | 190 | $z(n,8)-C_{4,1}$ | 209 |

## Access link of these two datasets
[The dataset HKDD_AMC12](https://figshare.com/articles/dataset/The_dataset_HKDD_AMC12_of_paper_Towards_Next-Generation_Signal_Intelligence_A_Hybrid_Knowledge_and_Data-Driven_Deep_Learning_Framework_for_Radio_Signal_Classification_/22047170)

[The first part of the dataset HKDD_AMC36](https://figshare.com/articles/dataset/The_first_part_of_the_dataset_HKDD_AMC36_of_paper_Towards_Next-Generation_Signal_Intelligence_A_Hybrid_Knowledge_and_Data-Driven_Deep_Learning_Framework_for_Radio_Signal_Classification_/22047071)

[The second part of the dataset HKDD_AMC36](https://figshare.com/articles/dataset/The_second_part_of_the_dataset_HKDD_AMC36_of_paper_Towards_Next-Generation_Signal_Intelligence_A_Hybrid_Knowledge_and_Data-Driven_Deep_Learning_Framework_for_Radio_Signal_Classification_/22047245)

## Citation
If you find this repository helpful, please consider citing:
```
@Article{zheng2023hkdd,
  title   = {Towards Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification},
  author  = {Shilian Zheng, Xiaoyu Zhou, Luxin Zhang, Peihan Qi, Kunfeng Qiu, Jiawei Zhu, and Xiaoniu Yang},
  journal = {IEEE Transactions on Cognitive Communications & Networking},
  year    = {2023},
}
```
