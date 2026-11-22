# Phase 2 Model Selection: What We Use, What We Do Not, and Why

**Team:** TensorTitans, Section C
**Owner:** Nitanshu
**Used in:** `reports/phase2/sections/related_work.tex` and `methodology.tex`

Our task: given today's 13 input layers on a 64x64 map, predict for every pixel whether it burns tomorrow. The output is a map, not one number.

---

## 1. The model we use: U-Net

| Part | What it does |
|---|---|
| Encoder (3 levels, 16, 32, 64 filters) | Looks at larger and larger areas of the map |
| Bottleneck (128 filters, dropout 0.2) | The smallest, most compressed view |
| Decoder (3 levels) | Brings the map back to 64x64 |
| Skip connections | Copy the sharp detail from each encoder level to the matching decoder level |
| Output | One sigmoid per pixel: the probability of fire tomorrow |

**Why it fits:** the output has the same shape as the input; fire edges need fine detail; it works with a small amount of data; and it is small enough to train on a laptop CPU (491,553 parameters, about 31 minutes).

---

## 2. Other models we could have used

| Model | Idea | Why it could work here | Status |
|---|---|---|---|
| Convolutional autoencoder | Squeeze then expand, with no skip connections | This is the model in the NDWS dataset paper (test PR-AUC about 0.284) | A fair baseline; our U-Net adds skip connections |
| Attention U-Net | U-Net plus attention gates on the skips | Focuses on the area near the fire | Best next upgrade (future work) |
| Residual U-Net | Residual connections inside each block | Trains deeper networks more steadily | Possible, but our network is shallow |
| U-Net with extra input channels | Add the Phase 1 features (for example, the burning-neighbour count) as more layers | The neighbour rule alone scored 0.173 on test | Cheap to try, not run |
| U-Net + LightGBM stacking | The U-Net output becomes input to LightGBM | Combines spatial and tabular strengths | We tried a quick draft version before the final notebooks (the script is not in this repo). It scored higher on validation (0.320 vs 0.310) but lower on test (0.272 vs 0.360), so it is not in the final submission |

---

## 3. Models we are not using, and why

| Model | Why not |
|---|---|
| **ConvLSTM / time-series models** (Shi et al., 2015) | These need a sequence of days. NDWS gives only one input day, so there is nothing for the memory part to learn. They suit datasets like WildfireSpreadTS. |
| **Vision Transformers and Swin U-Net** (Dosovitskiy et al., 2021; Cao et al., 2022) | They usually need much more data and a GPU. Our maps are only 64x64, and they are likely to overfit. |
| **Mamba-based U-Nets** | Very new and heavy. Not suitable for our hardware and deadline. |
| **DeepLabV3+** | Designed for large photos. At 64x64 its wide convolutions add little and it is heavier. |
| **Single-pixel MLP / tabular models** | They cannot see neighbours. This is exactly why the Phase 1 model failed. |
| **GANs and diffusion models** | They generate images. We need one probability per pixel, and they are unstable to train. |
| **Graph neural networks** | Our data is already a regular grid, which CNNs handle naturally. |
| **ImageNet-pretrained encoders** | Pretrained on colour photos. Our 13 layers are weather and terrain, so the benefit is small. |
| **Physics fire simulators** | Not machine learning, and they need fuel maps we do not have. |

---

## 4. Loss function choices

| Loss | Idea | Our test (validation PR-AUC, 4,000 maps, 10 epochs) |
|---|---|---|
| Plain cross-entropy | Every pixel counts the same | 0.233 |
| **Weighted cross-entropy (our choice)** | Fire pixels count 3 times; unknown pixels count 0 | **0.260** |
| Focal loss (Lin et al., 2017) | Down-weights easy pixels | 0.231 |
| Dice loss (Milletari et al., 2016) | Rewards overlap | Not tested |

---

## 5. One-line answers for the viva

- **Why U-Net?** The answer is a map the same size as the input, and skip connections keep the fire edges sharp.
- **Why not ConvLSTM?** We only have one day of input, so there is no sequence.
- **Why not a transformer?** Not enough data or compute for a 64x64 map.
- **Why not the autoencoder from the dataset paper?** It is a good baseline, but a U-Net adds skip connections, which help with sharp edges.
- **Did you try the hybrid?** We tried a quick one. It won on validation and lost on test, because the second model only had half of the validation maps to learn from. The correct fix is out-of-fold training, which we list as future work.
