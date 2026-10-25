# Interview Study Guide: Cat vs. Dog CNN Classifier

> **Honesty note:** this repo contains only `README.md` (no `.py`/`.ipynb`). Everything below
> marked ✅ comes from the README. Items marked ⚠️ are standard concepts or inferences you
> **must verify against your own code** before claiming them in an interview.

## 1. 30-second pitch
Binary image classifier (cat vs. dog) built as a CNN in PyTorch. Images are preprocessed with
torchvision `datasets`/`transforms`. The model is 4 conv blocks (Conv → BatchNorm → ReLU → MaxPool)
followed by a small fully connected head with dropout. Evaluated on a validation set with
accuracy, precision and recall; all had to exceed 0.6. ✅

## 2. Pipeline
1. **Data loading/preprocessing** ✅ torchvision `datasets` (likely `ImageFolder`⚠️) + `transforms`
   (⚠️ know your exact transforms: resize size, `ToTensor`, normalization, any augmentation).
2. **Model** ✅ see §3.
3. **Training** ✅ lr = 0.001, batch size 8, 8 epochs. ⚠️ Know your optimizer (Adam?), loss
   (almost certainly `CrossEntropyLoss` with 2 logits), and whether you call `model.train()` / `model.eval()`.
4. **Evaluation** ✅ accuracy, precision, recall on the validation set.

## 2b. Be ready to explain the training loop (⚠️ confirm in your code)
For each batch: `optimizer.zero_grad()` → forward pass → loss → `loss.backward()` → `optimizer.step()`.
Validation: `model.eval()` + `torch.no_grad()` so BatchNorm uses running stats and dropout is off.

## 3. Architecture ✅
| Block | Conv (k=3) | Then |
|---|---|---|
| 1 | 3 → 64 | BatchNorm, ReLU, MaxPool(2) |
| 2 | 64 → 128 | BatchNorm, ReLU, MaxPool(2) |
| 3 | 128 → 256 | BatchNorm, ReLU, MaxPool(2) |
| 4 | 256 → 512 | BatchNorm, ReLU, MaxPool(2) |
| Head | Linear 512 → 512, ReLU, Dropout(0.5), Linear 512 → 2 | logits for 2 classes |

Why this shape: channels double as spatial size halves — early layers learn edges/textures, later
layers learn parts (ears, snouts); more channels compensate for lost resolution.

⚠️ **Known gap:** the head takes 512 features, so something must reduce the 4th block's output to
1×1 (global/adaptive average pooling, or the input size happens to work out). Look up exactly
what you did; a senior will ask how you got from feature maps to a 512-vector.
(The README's Layer 5 title also says "Linear→ReLU→Linear→ReLU→Linear" but lists only two linears.)

## 4. Regularization / training stabilizers
- **Batch Normalization** ✅ after every conv. Normalizes activations per channel over the mini-batch,
  then applies learned scale/shift. Benefits: more stable gradients, tolerates higher learning rates,
  faster convergence, mild regularization from batch noise. Train vs eval: batch stats vs running averages.
  Caveat: batch size 8 is small, so batch statistics are noisy (good follow-up question).
- **Dropout (p=0.5)** ✅ in the FC head. Randomly zeroes activations in training (scaled by 1/(1-p)),
  preventing co-adaptation of units. Disabled at eval. Placed in the head because that's where most
  parameters (and overfitting risk) are.
- **Short training (8 epochs)** ✅ a crude form of early stopping to limit overfitting.
- **Low LR (0.001)** ✅ avoids overshooting minima.
- Not used (per README): weight decay, LR scheduler, augmentation. Be ready to say these are next steps.

## 5. Metrics (know the definitions cold)
- **Accuracy** = correct / total.
- **Precision** = TP / (TP + FP): of predicted "dog", how many truly are dogs.
- **Recall** = TP / (TP + FN): of real dogs, how many we found.
- Cat/dog is roughly balanced so accuracy is meaningful; precision/recall show per-class error trade-offs.
- Threshold of 0.6 is a project bar, not a state-of-the-art claim. Your actual numbers are in `images/Results.jpg`; memorize them.

## 6. Likely interview questions
1. **Why BatchNorm before ReLU?** Keeps pre-activations well-scaled; common convention.
2. **Why 3×3 kernels?** Stack of small kernels gives a large receptive field with fewer parameters.
3. **What does MaxPool do?** Downsamples, adds small translation invariance, reduces compute.
4. **How would you detect overfitting?** Train accuracy ≫ validation accuracy / validation loss rising.
5. **Why CrossEntropy with 2 outputs rather than sigmoid with 1?** Equivalent in effect; 2-logit softmax generalizes to more classes. ⚠️ confirm what you used.
6. **How would you improve it?** Data augmentation (flips, crops, color jitter), transfer learning (e.g. pretrained ResNet),
   weight decay, LR scheduling, larger batch size, early stopping on validation loss, k-fold / test set separate from validation.
7. **Limitations?** Evaluated only on validation set (no held-out test set), small batch size, small/simple model.
8. **How many parameters / what is the output shape after each block?** Practice computing by hand.

## 7. Pre-interview checklist
- [ ] Open your code; fill in optimizer, loss, transforms, input size, how 512 features are produced.
- [ ] Memorize final metrics from `images/Results.jpg`.
- [ ] Be able to draw the architecture and trace tensor shapes from input to output.
- [ ] Be able to explain train vs. eval mode differences.
- [ ] Prepare an honest answer on what you'd do differently.
