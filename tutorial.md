# Tutorial: MNIST denoising with RCPS

This tutorial accompanies [`rcps_toy_demo.ipynb`](rcps_toy_demo.ipynb). It explains the notebook as a small image-to-image uncertainty demo based on Angelopoulos et al., [*Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging*](https://arxiv.org/abs/2202.05265).

The task uses real handwritten digit images. Clean MNIST digits are the targets; noisy versions are the inputs. A convolutional neural network predicts a denoised image and a pixelwise uncertainty shape. Risk-Controlling Prediction Sets (RCPS) then calibrates the size of the pixel intervals on separate calibration images.

This is a teaching example, not an exact reproduction of the paper's FastMRI or microscopy experiments. The corruption is simulated, and the network is intentionally small.

## What you will see

For each noisy input digit, the notebook produces:

1. A **median prediction**: the model's central denoised image.
2. A **prediction interval at every pixel**: lower and upper grayscale values around the median prediction.
3. An **interval-width image**: wider/brighter pixels indicate greater estimated uncertainty; narrower/darker pixels indicate less.

The main visualization shows noisy inputs, clean targets, predicted medians, and interval widths. It does not draw the lower and upper bound images separately, though those bounds are computed by the notebook. The clean target is used for training and evaluation; at prediction time the model receives only the noisy image.

## 1. Set up and run the notebook

The notebook uses PyTorch, torchvision, NumPy, and Matplotlib. PyTorch and torchvision are included in this repository's `environment.yml`.

From the repository root, create and activate the project environment if you have not already done so:

```bash
conda env create -f environment.yml
conda activate im2im-uq
```

Open `experiments/rcps_toy_demo.ipynb` in Jupyter and run the cells from top to bottom. The first run downloads MNIST into the notebook's `data/` folder, so it needs an internet connection and a few hundred megabytes of disk space. Training and the 300-trial repeated calibration section take longer than the earlier demonstration cells; you can stop after the held-out evaluation if you only need the main prediction figure.

The notebook selects CUDA when available and otherwise runs on CPU. CPU training and the repeated simulation can take a while.

## 2. Understand the paired images

Each MNIST example gives a clean grayscale target image (Y), with pixel values in ([0,1]). The notebook creates a matching noisy input (X) by adding centered lognormal noise. The noise scale grows with the clean pixel intensity, so it is heteroscedastic; its distribution is asymmetric, so it is right-skewed. The noisy result is clipped to ([0,1]).

The model learns the mapping

\[
X\;\text{(noisy digit)} \longrightarrow Y\;\text{(clean digit)}.
\]

The data are split by MNIST image. The model sees 15,000 training images. A separate set of 2,500 images is used for RCPS calibration. Another 2,500 are reserved for validation but are not used by this simple walkthrough. The official MNIST test set is held out for the final example and repeated check.

## 3. What the network learns

The small convolutional network outputs three grayscale images for each input:

- the estimated **conditional median** at each pixel;
- the estimated 5th percentile at each pixel;
- the estimated 95th percentile at each pixel.

The median output is trained with pinball loss at quantile 0.5. The lower and upper outputs use pinball losses at 0.05 and 0.95. Those quantiles are only a first estimate of the uncertainty: their intervals are not assumed to have the desired risk on new images.

The notebook converts the estimates into lower and upper radii around the median. If the center is \(\hat f(X)\), and the radii are \(\tilde l(X)\) and \(\tilde u(X)\), a candidate interval at pixel \(p\) is

\[
\left[\hat f(X)_p-\lambda\tilde l(X)_p,\;\hat f(X)_p+\lambda\tilde u(X)_p\right].
\]

The interval endpoints are clipped to the valid grayscale range. The same nonnegative scale \(\lambda\) is applied to every pixel. Increasing \(\lambda\) makes all intervals wider, so these prediction sets are nested.

## 4. What “risk” means here

For one image, the loss is the fraction of its pixels whose clean target values fall outside their intervals:

\[
L_i(\lambda)=\frac{\#\{p:Y_{i,p}\text{ is outside its interval at }\lambda\}}{28\times 28}.
\]

This image-level loss lies between 0 and 1. The calibration calculation averages one such loss per image. It does not treat the 784 pixels in one image as 784 independent calibration examples.

The goal with \(\alpha=0.10\) is to control the expected fraction of missed pixels on a fresh image to at most 10%, with the risk-control statement failing for at most a \(\delta=0.10\) fraction of fresh calibration samples. This is an average-over-pixels, average-over-new-images guarantee; it does not promise that every image has at most 10% misses, nor that every pixel is covered.

## 5. How RCPS chooses \(\lambda\)

For a candidate scale, the notebook computes the mean calibration loss and adds the one-sided Hoeffding adjustment:

\[
R^+(\lambda)=\frac{1}{n}\sum_{i=1}^n L_i(\lambda)+\sqrt{\frac{\log(1/\delta)}{2n}}.
\]

Here \(n\) is the number of calibration images. With 2,500 images and \(\delta=0.10\), the additive adjustment is about 0.021. RCPS searches a grid from wide intervals toward narrow intervals and returns the smallest grid scale whose upper bound is no greater than \(\alpha\). If no grid value passes, the notebook raises an error instead of claiming the requested control.

The notebook computes the model's output once per calibration image and pixel, then derives the minimum scale each pixel would need to contain its target. This makes checking the lambda grid much faster; it is equivalent to testing the same nested intervals over and over for that fixed input/target pair.

### Paper terminology

- **Definition 1:** specifies the RCPS guarantee in terms of a chosen bounded loss. Here that loss is the image-level fraction of pixels missed.
- **Algorithm 1:** train a predictor and uncertainty heuristic, calibrate the heuristic, then return the resulting prediction sets for new inputs.
- **Algorithm 2:** select the interval scale using an upper confidence bound on calibration risk. This notebook uses the Hoeffding bound shown above.

The neural network proposes the interval shape. The calibration data determine how much to scale it. The bound is the mechanism that connects the observed calibration misses to a risk guarantee.

## 6. Read the main visualization

In the held-out figure, each column is one digit and the rows are:

1. **Noisy input (X)** — the only image supplied to the model at inference time.
2. **Clean target (Y)** — the answer used to assess the prediction.
3. **Predicted median** — the network's central denoised estimate.
4. **Interval width** — upper bound minus lower bound at each pixel.

The interval-width row is not itself the prediction set: it shows only how wide each set is. The lower and upper interval images together with the predicted center specify the set at every pixel. A useful extension for a presentation is an outside-interval mask that highlights pixels where the clean target fell outside the bounds.

## 7. Interpret the test risk and interval width

The notebook reports the mean image-level risk on the held-out test images and the average width of the intervals. Lower risk means fewer missed pixels on average. Narrower intervals are more informative, provided they still meet the risk target. The calibration guarantee concerns risk; interval width is a utility measure, not part of the guarantee.

A particular finite test run can have risk above \(\alpha\). The guarantee is probabilistic over calibration samples and controls expected risk on a fresh image; it is not a promise about every finite batch or every individual image.

## 8. Why repeat calibration 300 times?

The final section keeps the trained model fixed and repeats the calibration step on fresh samples. Each trial chooses a new \(\lambda\) from 1,000 calibration images, then estimates risk using 500 independent test images. The calibration and test draws use separate pools of held-out MNIST images. A histogram shows how the estimated test risk varies, and the notebook reports the fraction of trials whose estimated risk exceeds \(\alpha\).

This is an empirical check of the calibration behavior, not a proof. The MNIST test pools are finite and each test risk is estimated from a finite batch, so the observed fraction can fluctuate around \(\delta\). The formal RCPS result comes from the assumptions and bound, not from requiring every simulation to land below 10%.

## 9. RCPS versus Bayesian uncertainty

This notebook is an RCPS example; it is not a Bayesian neural network. The CNN's quantile heads supply heuristic per-pixel interval widths, and RCPS scales them using calibration data. A Bayesian or Monte Carlo-dropout model could instead propose intervals from variation among posterior predictions. That would be a separate uncertainty-estimation method. To obtain the paper-style risk-control guarantee, its proposed sets would still need an appropriate RCPS calibration step.

## Troubleshooting

- **MNIST download fails:** check the internet connection, then rerun the dataset-loading cell.
- **Out of memory:** lower `BATCH_SIZE` (for example, to 64) and rerun from the data/model cells.
- **No feasible lambda:** increase the calibration set size or `lam_max`, then rerun calibration. Do not report a calibrated guarantee when the bound did not pass.
- **NameError after changing a notebook cell:** run the notebook from the top, or at least rerun the cell defining the missing function and every dependent cell below it. The kernel retains old definitions until cells are rerun.

## Reference

Angelopoulos, A. N. et al. (2022). [Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging](https://arxiv.org/abs/2202.05265). See Section 2, Definition 1, and Algorithms 1–2.
