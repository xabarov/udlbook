When you predict multiple continuous targets (here: height in meters and weight in kg) with a single regression loss (e.g., MSE), **different units/scales** can cause:

## What problems does this cause?

- **Loss domination by the larger-scale target.**  
  With an MSE like
  $$
  \mathcal{L}=\frac{1}{N}\sum_{i=1}^N\left[(\hat h_i-h_i)^2+(\hat w_i-w_i)^2\right],
  $$
  typical absolute errors in kg are numerically much larger than errors in meters, so gradients from weight dominate. The model may effectively “care” much more about weight than height.

- **Imbalanced optimization / poor conditioning.**  
  The shared network parameters get gradient signals of very different magnitudes across tasks, which can slow training, make learning-rate tuning harder, and yield a suboptimal trade-off (one target improves while the other stagnates).

## Two solutions

1) **Normalize/standardize the targets (train in normalized space).**  
   Transform each target to comparable scale, e.g.
   $$
   h'_i=\frac{h_i-\mu_h}{\sigma_h},\quad w'_i=\frac{w_i-\mu_w}{\sigma_w},
   $$
   train the network to predict $(h',w')$ with a standard loss, then invert the transform at inference.

2) **Use a weighted loss to balance units/scales.**  
   Define
   $$
   \mathcal{L}=\frac{1}{N}\sum_{i=1}^N\left[\alpha(\hat h_i-h_i)^2+\beta(\hat w_i-w_i)^2\right],
   $$
   where weights can be set using domain knowledge or statistics (common choice: $\alpha=\frac{1}{\sigma_h^2}$, $\beta=\frac{1}{\sigma_w^2}$), so each dimension contributes comparably.  
   (Variant: make $\alpha,\beta$ learnable, e.g., uncertainty-based weighting.)

If you tell me what loss you’re currently using (MSE/Huber/etc.) and whether the model shares most layers between height and weight, I can suggest a concrete weighting/normalization setup.