# [Toward Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification](https://ieeexplore.ieee.org/document/10042021)

## About paper
**Title**: [Toward Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification.](https://ieeexplore.ieee.org/document/10042021)

**Author**: Shilian Zheng, Xiaoyu Zhou, Luxin Zhang, Peihan Qi, Kunfeng Qiu, Jiawei Zhu, and Xiaoniu Yang.

**Journal**: IEEE Transactions on Cognitive Communications and Networking.

**Abstract**: Automatic modulation classification (AMC) can generally be divided into knowledge-based methods and data-driven methods. In this paper, we explore combining the knowledgebased method and data-driven technology to take full advantage of both and propose a hybrid knowledge and data-driven deep learning framework (HKDD) for AMC. To make the handcrafted features more discriminative, various traditional features are adopted, including instantaneous features, statistical features, and spectral features. In the HKDD framework, a feature fusion mechanism is proposed to integrate the features learned from the original signal with those processed by a fully connected network from the handcrafted features. Besides, an attention mechanism is implemented on the fused features to neglect immature features and highlight important features. To evaluate the performance of the proposed method, we construct two modulation classification datasets containing both traditional features and raw IQ data. The bigger one contains 36 modulation categories, which is greater than the number of categories of any AMC dataset currently available. Simulation results show that our proposed method has significant performance gain in both adequate-sample classification scenario and few-shot classification scenario.

## Datasets
In both datasets, the original bit sequence is chosen from 0 and 1 in a random manner to ensure that the probability of appearance for each symbol is equivalent. The length of each modulated signal is 1024 for dataset HKDD_AMC36 and 512 for dataset HKDD_AMC12. The oversampling rate is 8, so each sampled sequence in dataset HKDD_AMC36 contains 128 symbols and each sampled sequence in dataset HKDD_AMC12 contains 64 symbols. A root raised-cosine (RRC) filter with 6-symbols truncated length is employed as the pulse-shaping filter and the roll-off coefficient of RRC is randomly chosen within the range 0.2 to 0.7. The frequency offset is randomly chosen from -0.2 to 0.2 (normalized to the sampling frequency). The range of SNR is (-20 dB, 30 dB) for dataset HKDD_AMC36 and (-20 dB, 20 dB) for dataset HKDD_AMC12 with an interval of 2 dB. The number of training samples for each modulation type is 1000 in each SNR and the number of testing samples is half of the training samples. Both datasets contain both IQ signals and traditional features. HKDD_AMC36 is the dataset currently available that contains the most number of modulation categories.

We give an example of importing dataset through Python:
```
import h5py
train_data = h5py.File('HKDD_AMC12_train.mat')
data_raw = train_data['XTrainIQ']
data_feature = train_data['Feature']
```

- The shape of **data_raw** is (2, L, 1, N), 2 represents I and Q channels of each signal, L is the length of each modulated signal, N is the number of signal samples. The shape of **data_feature** is (228, N), 228 represents the number of different adopted traditional features.
- The corresponding indexes of these 228 features in **data_feature** are shown in the following table, and the specific definitions of these 228 features can be found in the paper.

| feature | index | feature | index | feature | index | feature | index | feature | index | feature | index | feature | index | feature | index | feature | index | feature | index | feature | index | feature | index |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $K$ | 0 | $F_3^8$ | 19 | $r(n)@M_{12,2}$ | 38 | $r(n)@C_{6,2}$ | 57 | $z(n,2)@M_{4,0}$ | 76 | $z(n,2)@M_{16,0}$ | 95 | $z(n,2)@C_{8,3}$ | 114 | $z(n,4)@M_{6,2}$ | 133 | $z(n,4)@M_{16,5}$ | 152 | $z(n,4)@\\widehat C_{8,0}$ | 171 | $z(n,8)@M_{8,3}$ | 190 | $z(n,8)@C_{4,1}$ | 209 |
| $P$ | 1 | $r(n)@M_{2,0}$ | 20 | $r(n)@M_{12,3}$ | 39 | $r(n)@C_{6,3}$ | 58 | $z(n,2)@M_{4,1}$ | 77 | $z(n,2)@M_{16,1}$ | 96 | $z(n,2)@C_{8,4}$ | 115 | $z(n,4)@M_{6,3}$ | 134 | $z(n,4)@M_{16,6}$ | 153 | $z(n,4)@\\widehat C_{8,1}$ | 172 | $z(n,8)@M_{8,4}$ | 191 | $z(n,8)@M_{4,2}$ | 210 |
| $\\gamma_{\\max}$ | 2 | $r(n)@M_{2,1}$ | 21 | $r(n)@M_{12,4}$ | 40 | $r(n)@C_{8,0}$ | 59 | $z(n,2)@M_{4,2}$ | 78 | $z(n,2)@M_{16,2}$ | 97 | $z(n,2)@-\\widehat C_{6,0}$ | 116 | $z(n,4)@M_{8,0}$ | 135 | $z(n,4)@M_{16,7}$ | 154 | $z(n,4)@\\widehat C_{8,2}$ | 173 | $z(n,8)@M_{12,0}$ | 192 | $z(n,8)@M_{6,0}$ | 211 |
| $\\sigma_{aa}$ | 3 | $r(n)@M_{3,0}$ | 22 | $r(n)@M_{12,5}$ | 41 | $r(n)@C_{8,1}$ | 60 | $z(n,2)@M_{6,0}$ | 79 | $z(n,2)@M_{16,3}$ | 98 | $z(n,2)@\\widehat C_{6,1}$ | 117 | $z(n,4)@M_{8,1}$ | 136 | $z(n,4)@M_{16,8}$ | 155 | $z(n,4)@\\widehat C_{8,3}$ | 174 | $z(n,8)@M_{12,1}$ | 193 | $z(n,8)@M_{6,1}$ | 212 |
| $\\sigma_{af}$ | 4 | $r(n)@M_{3,1}$ | 23 | $r(n)@M_{12,6}$ | 42 | $r(n)@C_{8,2}$ | 61 | $z(n,2)@M_{6,1}$ | 80 | $z(n,2)@M_{16,4}$ | 99 | $z(n,2)@\\widehat C_{6,2}$ | 118 | $z(n,4)@M_{8,2}$ | 137 | $z(n,4)@C_{4,0}$ | 156 | $z(n,4)@\\widehat C_{4,2}$ | 175 | $z(n,8)@M_{12,2}$ | 194 | $z(n,8)@M_{6,2}$ | 213 |
| $\\sigma_a$ | 5 | $r(n)@M_{4,0}$ | 24 | $r(n)@M_{16,0}$ | 43 | $r(n)@C_{8,3}$ | 62 | $z(n,2)@M_{6,2}$ | 81 | $z(n,2)@M_{16,5}$ | 100 | $z(n,2)@\\widehat C_{8,0}$ | 119 | $z(n,4)@M_{8,3}$ | 138 | $z(n,4)@C_{4,1}$ | 157 | $z(n,8)@M_{2,0}$ | 176 | $z(n,8)@M_{12,3}$ | 195 | $z(n,8)@M_{6,3}$ | 214 |
| $\\mu_a$ | 6 | $r(n)@M_{4,1}$ | 25 | $r(n)@M_{16,1}$ | 44 | $r(n)@C_{8,4}$ | 63 | $z(n,2)@M_{6,3}$ | 82 | $z(n,2)@M_{16,6}$ | 101 | $z(n,2)@\\widehat C_{8,1}$ | 120 | $z(n,4)@M_{8,4}$ | 139 | $z(n,4)@C_{4,2}$ | 158 | $z(n,8)@M_{2,1}$ | 177 | $z(n,8)@M_{12,4}$ | 196 | $z(n,8)@M_{8,0}$ | 215 |
| $\\mu_f$ | 7 | $r(n)@M_{4,2}$ | 26 | $r(n)@M_{16,2}$ | 45 | $r(n)@\\widehat C_{6,0}$ | 64 | $z(n,2)@M_{8,0}$ | 83 | $z(n,2)@M_{16,7}$ | 102 | $z(n,2)@\\widehat C_{8,2}$ | 121 | $z(n,4)@M_{12,0}$ | 140 | $z(n,4)@C_{6,0}$ | 159 | $z(n,8)@M_{3,0}$ | 178 | $z(n,8)@M_{12,5}$ | 197 | $z(n,8)@M_{8,1}$ | 216 |
| $F_1^1$ | 8 | $r(n)@M_{6,0}$ | 27 | $r(n)@M_{16,3}$ | 46 | $r(n)@\\widehat C_{6,1}$ | 65 | $z(n,2)@M_{8,1}$ | 84 | $z(n,2)@M_{16,8}$ | 103 | $z(n,2)@\\widehat C_{8,3}$ | 122 | $z(n,4)@M_{12,1}$ | 141 | $z(n,4)@C_{6,1}$ | 160 | $z(n,8)@M_{3,1}$ | 179 | $z(n,8)@M_{12,6}$ | 198 | $z(n,8)@M_{8,2}$ | 217 |
| $F_2^1$ | 9 | $r(n)@M_{6,1}$ | 28 | $r(n)@M_{16,4}$ | 47 | $r(n)@\\widehat C_{6,2}$ | 66 | $z(n,2)@M_{8,2}$ | 85 | $z(n,2)@C_{4,0}$ | 104 | $z(n,2)@\\widehat C_{4,2}$ | 123 | $z(n,4)@M_{12,2}$ | 142 | $z(n,4)@C_{6,2}$ | 161 | $z(n,8)@M_{4,0}$ | 180 | $z(n,8)@M_{16,0}$ | 199 | $z(n,8)@M_{8,3}$ | 218 |
| $F_3^1$ | 10 | $r(n)@M_{6,2}$ | 29 | $r(n)@M_{16,5}$ | 48 | $r(n)@\\widehat C_{8,0}$ | 67 | $z(n,2)@M_{8,3}$ | 86 | $z(n,2)@C_{4,1}$ | 105 | $z(n,4)@M_{2,0}$ | 124 | $z(n,4)@M_{12,3}$ | 143 | $z(n,4)@C_{6,3}$ | 162 | $z(n,8)@M_{4,1}$ | 181 | $z(n,8)@M_{16,1}$ | 200 | $z(n,8)@M_{8,4}$ | 219 |
| $F_1^2$ | 11 | $r(n)@M_{6,3}$ | 30 | $r(n)@M_{16,6}$ | 49 | $r(n)@\\widehat C_{8,1}$ | 68 | $z(n,2)@M_{8,4}$ | 87 | $z(n,2)@C_{4,2}$ | 106 | $z(n,4)@M_{2,1}$ | 125 | $z(n,4)@M_{12,4}$ | 144 | $z(n,4)@C_{8,0}$ | 163 | $z(n,8)@M_{4,3}$ | 182 | $z(n,8)@M_{16,2}$ | 201 | $z(n,8)@\\widehat C_{6,0}$ | 220 |
| $F_2^2$ | 12 | $r(n)@M_{8,0}$ | 31 | $r(n)@M_{16,7}$ | 50 | $r(n)@\\widehat C_{8,2}$ | 69 | $z(n,2)@M_{12,0}$ | 88 | $z(n,2)@C_{6,0}$ | 107 | $z(n,4)@M_{3,0}$ | 126 | $z(n,4)@M_{12,5}$ | 145 | $z(n,4)@C_{8,1}$ | 164 | $z(n,8)@M_{6,0}$ | 183 | $z(n,8)@M_{16,3}$ | 202 | $z(n,8)@\\widehat C_{6,1}$ | 221 |
| $F_3^2$ | 13 | $r(n)@M_{8,1}$ | 32 | $r(n)@M_{16,8}$ | 51 | $r(n)@\\widehat C_{8,3}$ | 70 | $z(n,2)@M_{12,1}$ | 89 | $z(n,2)@C_{6,1}$ | 108 | $z(n,4)@M_{3,1}$ | 127 | $z(n,4)@M_{12,6}$ | 146 | $z(n,4)@C_{8,2}$ | 165 | $z(n,8)@M_{6,1}$ | 184 | $z(n,8)@M_{16,4}$ | 203 | $z(n,8)@\\widehat C_{6,2}$ | 222 |
| $F_1^4$ | 14 | $r(n)@M_{8,2}$ | 33 | $r(n)@C_{4,0}$ | 52 | $r(n)@\\widehat C_{4,2}$ | 71 | $z(n,2)@M_{12,2}$ | 90 | $z(n,2)@C_{6,2}$ | 109 | $z(n,4)@M_{4,0}$ | 128 | $z(n,4)@M_{16,0}$ | 147 | $z(n,4)@C_{8,3}$ | 166 | $z(n,8)@M_{6,2}$ | 185 | $z(n,8)@M_{16,5}$ | 204 | $z(n,8)@\\widehat C_{8,0}$ | 223 |
| $F_2^4$ | 15 | $r(n)@M_{8,3}$ | 34 | $r(n)@C_{4,1}$ | 53 | $z(n,2)@M_{2,0}$ | 72 | $z(n,2)@M_{12,3}$ | 91 | $z(n,2)@C_{6,3}$ | 110 | $z(n,4)@M_{4,1}$ | 129 | $z(n,4)@M_{16,1}$ | 148 | $z(n,4)@C_{8,4}$ | 167 | $z(n,8)@M_{6,3}$ | 186 | $z(n,8)@M_{16,6}$ | 205 | $z(n,8)@\\widehat C_{8,1}$ | 224 |
| $F_3^4$ | 16 | $r(n)@M_{8,4}$ | 35 | $r(n)@C_{4,2}$ | 54 | $z(n,2)@M_{2,1}$ | 73 | $z(n,2)@M_{12,4}$ | 92 | $z(n,2)@C_{8,1}$ | 111 | $z(n,4)@M_{4,2}$ | 130 | $z(n,4)@M_{16,2}$ | 149 | $z(n,4)@\\widehat C_{6,0}$ | 168 | $z(n,8)@M_{8,0}$ | 187 | $z(n,8)@M_{16,7}$ | 206 | $z(n,8)@\\widehat C_{8,2}$ | 225 |
| $F_1^8$ | 17 | $r(n)@M_{12,0}$ | 36 | $r(n)@C_{6,0}$ | 55 | $z(n,2)@M_{3,0}$ | 74 | $z(n,2)@M_{12,5}$ | 93 | $z(n,2)@C_{8,1}$ | 112 | $z(n,4)@M_{6,0}$ | 131 | $z(n,4)@M_{16,3}$ | 150 | $z(n,4)@\\widehat C_{6,1}$ | 169 | $z(n,8)@M_{8,1}$ | 188 | $z(n,8)@M_{16,8}$ | 207 | $z(n,8)@\\widehat C_{8,3}$ | 226 |
| $F_2^8$ | 18 | $r(n)@M_{12,1}$ | 37 | $r(n)@C_{6,1}$ | 56 | $z(n,2)@M_{3,1}$ | 75 | $z(n,2)@M_{12,6}$ | 94 | $z(n,2)@C_{8,2}$ | 113 | $z(n,4)@M_{6,1}$ | 132 | $z(n,4)@M_{16,4}$ | 151 | $z(n,4)@\\widehat C_{6,2}$ | 170 | $z(n,8)@M_{8,2}$ | 189 | $z(n,8)@M_{4,0}$ | 208 | $z(n,8)@\\widehat C_{4,2}$ | 227 |

## Access link of these two datasets
[The dataset HKDD_AMC12](https://figshare.com/articles/dataset/The_dataset_HKDD_AMC12_of_paper_Towards_Next-Generation_Signal_Intelligence_A_Hybrid_Knowledge_and_Data-Driven_Deep_Learning_Framework_for_Radio_Signal_Classification_/22047170)

[The first part of the dataset HKDD_AMC36](https://figshare.com/articles/dataset/The_first_part_of_the_dataset_HKDD_AMC36_of_paper_Towards_Next-Generation_Signal_Intelligence_A_Hybrid_Knowledge_and_Data-Driven_Deep_Learning_Framework_for_Radio_Signal_Classification_/22047071)

[The second part of the dataset HKDD_AMC36](https://figshare.com/articles/dataset/The_second_part_of_the_dataset_HKDD_AMC36_of_paper_Towards_Next-Generation_Signal_Intelligence_A_Hybrid_Knowledge_and_Data-Driven_Deep_Learning_Framework_for_Radio_Signal_Classification_/22047245)

## Citation
If you find this repository helpful, please consider citing:
```
S. Zheng et al., "Toward Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification," in IEEE Transactions on Cognitive Communications and Networking, doi: 10.1109/TCCN.2023.3243899.
```
or
```
@ARTICLE{10042021,
  author={Zheng, Shilian and Zhou, Xiaoyu and Zhang, Luxin and Qi, Peihan and Qiu, Kunfeng and Zhu, Jiawei and Yang, Xiaoniu},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={Toward Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCCN.2023.3243899}}
```
