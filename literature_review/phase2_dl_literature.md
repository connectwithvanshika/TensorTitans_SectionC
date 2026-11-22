# Phase 2 Literature Review: Deep Learning for Next-Day Wildfire Spread

**Team:** TensorTitans, Section C
**Owners:** Vanshika and Nitanshu
**Used in:** `reports/phase2/sections/introduction.tex` and `related_work.tex`

How to read this file: read the papers in the order below. Each one has the same five parts: what it is, what they did, the key result, how we use it, and its limit.

---

## 1. Huot et al. (2022): the dataset paper

**Cite as:** `huot2022nextday` (IEEE Transactions on Geoscience and Remote Sensing, vol. 60, DOI 10.1109/TGRS.2022.3192974)

- **What it is:** The paper that created the Next Day Wildfire Spread (NDWS) dataset we use.
- **What they did:** Built 64x64 maps (1 km per pixel) over the US from 2012 to 2020. Each map has 12 input layers (weather, wind, terrain, vegetation, drought, population and yesterday's fire mask) and one label: where the fire is tomorrow. They trained a convolutional autoencoder, logistic regression and random forest.
- **Key result:** The convolutional autoencoder reached a test PR-AUC of about 0.284. The persistence rule ("tomorrow looks like today") scored about 0.115. Logistic regression and random forest were close to each other and at least 6 points lower than the neural network. These numbers come from the paper and from later papers that quote it. Precision about 0.336 and recall about 0.431 are quoted in benchmark summaries, so check them in the paper before using them on a slide.
- **How we use it:** This is our dataset and our reference score. Our persistence rule gets 0.107 on our test split, close to their 0.115, which shows our evaluation is set up fairly. Our U-Net gets 0.356.
- **Limit:** Their training pipeline (for example, how they crop maps) is not identical to ours, so we say "comparable or better", not "we beat the paper".

## 2. Ronneberger, Fischer and Brox (2015): U-Net

**Cite as:** `ronneberger2015unet` (MICCAI 2015, DOI 10.1007/978-3-319-24574-4_28)

- **What it is:** The original U-Net, built to mark every pixel of a medical image.
- **What they did:** An encoder squeezes the image to find what is in it. A decoder expands it back to full size to find where. Skip connections copy detail from the encoder to the decoder. They trained on very few images and relied on heavy augmentation (flips, shifts, deformations).
- **Key result:** It won several biomedical segmentation challenges in 2015 and became the standard design for pixel-level prediction.
- **How we use it:** Our main model. Tomorrow's fire map has the same shape as the input map, so a per-pixel output is exactly what we need. Skip connections keep the sharp edges of the fire.
- **Limit:** Built for medical images. Fire spreads with wind and terrain, so we add wind vectors as input and flip the wind sign whenever we flip a map.

## 3. Oktay et al. (2018): Attention U-Net

**Cite as:** `oktay2018attention` (MIDL 2018, arXiv 1804.03999)

- **What it is:** A U-Net with "attention gates" on the skip connections.
- **What they did:** The gate learns which parts of the encoder features matter and quiets the rest before they reach the decoder. They tested on CT scans of the pancreas.
- **Key result:** Better results than a plain U-Net on that task, with only a small increase in size.
- **How we use it:** The most natural upgrade to our U-Net, and a model we name as future work. We did not train it, because the deadline was close and the plain U-Net already beats every baseline.
- **Limit:** More code and more training time. The gain on 64x64 fire maps is unknown until tested.

## 4. Lin et al. (2017): Focal Loss

**Cite as:** `lin2017focal` (ICCV 2017, pp. 2980 to 2988)

- **What it is:** A loss function for tasks where almost everything is "negative".
- **What they did:** Standard cross-entropy gets swamped by the easy negatives. Focal loss shrinks the loss of easy pixels by a factor of (1 - p)^gamma, so the model spends its effort on hard pixels. They used gamma = 2.
- **Key result:** Large gains for object detection, where background dominates.
- **How we use it:** Fire is only about 1.1% of known pixels, so this is a natural choice. We tested it in our ablation (notebook 12).
- **Our finding:** On our data, focal loss scored 0.231 on validation, worse than our weighted cross-entropy at 0.260. So the simpler loss won, and we kept it.

## 5. Milletari, Navab and Ahmadi (2016): V-Net and the Dice loss

**Cite as:** `milletari2016vnet` (3DV 2016)

- **What it is:** A 3D medical segmentation network that introduced a loss based on the overlap between the predicted and true shape (the Dice loss).
- **Why it matters:** Overlap scores ignore the huge empty background, so the loss does not reward "predict nothing". Many fire papers use it or a mix of Dice and cross-entropy.
- **How we use it:** We list it as an alternative loss. We did not test it, because the weighted cross-entropy was already strong and we ran out of time.
- **Limit:** Dice can be unstable when a map has no fire at all, which is common in this data.

## 6. Fitzgerald et al. (2023): attention U-Net on NDWS

**Cite as:** `fitzgerald2023paying` (ICMI 2023, DOI 10.1145/3577190.3614116)

- **What it is:** A U-Net with attention blocks, built for this same dataset.
- **What they did:** Used the 12 NDWS input layers to predict where fire will be in 24 hours. The model was designed to train efficiently.
- **How we use it:** It shows that U-Net style models are a sensible direction for NDWS, and it is the closest paper to our approach.
- **Limit:** We did not copy their numbers into our report, because we have not checked their exact evaluation setup. Compare only after reading their results table.

## 7. Kumar et al. (2025): has the network learned the fire process?

**Cite as:** `kumar2025stochastic` (ICLR 2025, arXiv 2402.15163)

- **What it is:** A paper about how we should evaluate fire-spread models, using this dataset.
- **What they did:** They argue that matching tomorrow's fire pixels is not the same as learning the random process behind fire. They propose a check called Fidelity to Stochastic Process and show that calibration error is a good tool for it.
- **How we use it:** It supports our view that plain accuracy is not enough. It also tells us something we have not done: we never measured calibration. A calibration plot is a natural next step.
- **Limit:** It is about evaluation, not a new model.

## 8. Gerard, Zhao and Sullivan (2023): WildfireSpreadTS

**Cite as:** `gerard2023wildfirespreadts` (NeurIPS 2023, Datasets and Benchmarks)

- **What it is:** A newer dataset with a full daily time series for each fire: 13,607 images across 607 US fire events, from 2018 to 2021, at 375 m resolution.
- **Why it matters:** Our NDWS data gives only one day of input. WildfireSpreadTS gives many days, so models that read sequences (like ConvLSTM) make sense there.
- **How we use it:** It is our answer to "why not ConvLSTM?". With one input day there is no sequence to learn from. It is also a possible future dataset.
- **Limit:** Large (about 48 GB), so it is not suitable for our laptop deadline.

## 9. Wolpert (1992): stacked generalization

**Cite as:** `wolpert1992stacked` (Neural Networks, vol. 5, no. 2, pp. 241 to 259)

- **What it is:** The theory behind stacking, where one model learns from the outputs of other models.
- **The key rule:** The second-level model must learn from predictions on data the first-level models never saw. Otherwise it learns from over-confident predictions and the result is fake.
- **How we use it:** We tried a quick draft hybrid (U-Net output fed into LightGBM; the script is not part of this repo) and found it scored higher on validation but lower on test. The rule above is the likely reason: only half of the validation set was left for the second model. We did not take the hybrid forward in the final submission.
- **Limit:** A proper version needs out-of-fold training, which costs several times the compute.

## 10. Saito and Rehmsmeier (2015): PR curves for rare events

**Cite as:** `saito2015precision` (PLoS ONE, DOI 10.1371/journal.pone.0118432)

- **What it is:** A study of why the precision-recall curve is better than the ROC curve when positives are rare.
- **How we use it:** The reason our main metric is PR-AUC. In Phase 1, the ROC-AUC looked acceptable while no fires were caught. In Phase 2, the U-Net's PR-AUC of 0.356 can be compared directly with the fire rate of 0.0125 (28 times higher).

---

## Supporting techniques we use in the model

| Technique | Paper | Where it is in our code |
|---|---|---|
| Batch normalisation | Ioffe and Szegedy, ICML 2015 | after every convolution in `build_unet` |
| Dropout | Srivastava et al., JMLR 2014 | rate 0.2 in the bottleneck |
| AdamW (weight decay) | Loshchilov and Hutter, ICLR 2019 | the optimizer in `train_unet` |

---

## The gap our project fills

Earlier work shows that deep models can predict next-day fire spread, but many projects stop at "the model scored X". We add three things that a course project can show honestly:

1. A Phase 1 result that failed (62 fire pixels, PR-AUC equal to random), and a clear explanation of why.
2. A U-Net trained on all 14,979 maps (658,270 fire pixels) that beats both simple rules and the Phase 1 model on the same held-out test.
3. An ablation that shows which choices mattered: yesterday's fire (-0.084), the loss weighting, flips and weather.
