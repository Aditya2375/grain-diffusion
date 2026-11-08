# Grain / 02

[Open the guided pixel laboratory](https://aditya2375.github.io/grain-diffusion/)

A real conditional pixel diffusion experiment: class-dropout fine-tuning, classifier-free guidance, deterministic DDIM, same-noise comparisons, noise interpolation, scrubbed trajectories and local PNG export. **14 × 14 grayscale MNIST digits only. Not general text-to-image.** No pretrained generator, hidden image gallery, remote inference or quality guarantee. Rough, ambiguous and wrong digits remain possible.

## Two real checkpoints

The original `2-model.json` is preserved: seed 431, 30,000 AdamW updates, 173,252 parameters. The second `1-guided-model.json` **starts from the original local checkpoint**, not from random initialization. Its fixed additional budget is 12,000 updates, seed 731, AdamW learning rate 0.0002, weight decay 0.0001, batch 128, norm clipping 1. The final checkpoint is used, not the minimum validation checkpoint. With probability 0.15 per example, all ten class features are zeroed, teaching the same network a null-conditioned noise estimate. Other training settings, data split and schedule stay the same.

Architecture: 196 pixels + 16 sinusoidal timestep features + 10 class features → 256 SiLU → 256 SiLU → 196 noise outputs. No U-Net, text encoder, latent autoencoder or classifier. The learned null condition exists only in the new checkpoint. No outside pretrained model or additional dataset is used.

## A same-noise experiment

The two panes use one checkpoint, the same initial Gaussian array, same requested class and same DDIM time grid. Left: scale 1. Right: selected scale `s`, with

```
epsilon_guided = epsilon_null + s * (epsilon_class - epsilon_null)
```

Scale 0 is null-conditioned, 1 is ordinary conditional sampling, values above 1 extrapolate toward the condition. More guidance is not a guarantee of better samples, class correctness or diversity. Original checkpoint mode forces both panes to 1 because its null condition was not trained. Both panes then match exactly. This pair isolates guidance, not training improvement.

Sampling is deterministic DDIM with eta=0. The grid has 8, 16, 32 or 64 distinct timesteps uniformly spanning indices 63 to 0, rounded to nearest integer. After the last index, alpha-bar at the clean endpoint is 1. At each step:

1. Predict noise with the guided formula.
2. Estimate clean pixels `(x - sqrt(1-a)*epsilon) / sqrt(a)`, clipped to [-1,1].
3. Recompute the implied epsilon after clipping: `(x - sqrt(a)*clipped_clean)/sqrt(1-a)`.
4. Set the next state to `sqrt(a_next)*clipped_clean + sqrt(1-a_next)*corrected_epsilon`.

No fresh Gaussian noise is added between DDIM steps. Clipping without consistently recomputing epsilon can produce a very different process; both the browser and independent NumPy implementation use the above correction. The original ancestral DDPM implementation remains in `engine.js`, with its tests, but the new comparison interface uses DDIM for both checkpoints. DDIM 64 is not the original stochastic DDPM trajectory.

Each step uses one network call for ordinary conditioning, two for a non-1 guidance scale. At scale 1 the right pane reuses the left state instead of doing redundant work. Elapsed time includes both paths, message passing and yields. It is a local observation, not a general speed benchmark. Fewer time steps mean fewer network calls, not an established quality/speed optimum.

## Noise interpolation and actual frames

Two 32-bit seeds generate two arrays with the original LCG/Box-Muller implementation. The blend at u is `cos(pi*u/2)*A + sin(pi*u/2)*B`. Endpoints use A/B respectively, up to floating-point rounding. This has unit marginal variance if A and B are independent standard Gaussian arrays; equal or correlated seed arrays do not satisfy that assumption. This is a path in noise space, not guaranteed semantic interpolation or a distribution-preserving guarantee for fixed deterministic seeds.

All displayed stages are arrays computed in the sampling worker, not slideshow images. Changing controls does not relabel an existing result: its actual checkpoint/class/seeds/blend/scale remain attached. Stop invalidates the request ID, retains only received stages and explicitly labels the result partial. New jobs supersede old jobs. Scrubbing/export always uses the displayed stage. Export is nearest-neighbor enlarged 280×280 grayscale PNG, never claimed to add detail. Saved PNGs may be intermediate if the displayed stage is intermediate.

## Evidence, without an image-quality claim

2,048 fixed shared-noise probes, seed 1281, from 1,691 distinct source images in the existing last-5,000 validation partition:

- Original conditional noise MSE: **0.23226032**.
- Fine-tuned conditional noise MSE: **0.21367668**.
- Paired reduction: **0.01858365**.
- 95% source-image-cluster percentile bootstrap interval: **[0.01794329, 0.01927109]**, 3,000 draws, seed 321.

Probes may reuse a source image. Bootstrap resamples unique source images with replacement, keeping all associated probes together and weighting by the resulting number of probes. The pairing compares identical image/noise/timestep/class inputs. This is a fixed validation probe set, monitored every 1,000 fine-tuning updates, not an untouched new test or new-population estimate. The checkpoint was a fixed final budget, not selected from those measurements.

**Noise MSE is not an image-quality score.** These numbers do not establish better-looking digits, class accuracy, diversity, novelty or freedom from memorization. No FID, perceptual metric, digit-classifier accuracy, diversity or nearest-neighbor/memorization study was done. The official MNIST test set was not used. The UI keeps these limitations beside the numbers. Validation losses include conditional and null modes; the original checkpoint's untrained null-mode loss is not a claimed fair null baseline.

## Data and provenance

[MNIST via CVDF](https://github.com/cvdfoundation/mnist), by Yann LeCun, Corinna Cortes and Christopher J.C. Burges. Original training files from https://storage.googleapis.com/cvdf-datasets/mnist/ . First 55,000 official training images train; last 5,000 validate. Images are 2×2 average pooled from 28×28 to 14×14, then scaled to [-1,1]. The original cosine schedule has 64 steps, offset 0.008, beta capped at 0.999. Raw source SHA-256 hashes are in both checkpoints and checked before fine-tuning. Data are not bundled in the browser app.

Methods: [DDPM, Ho et al. (2020)](https://arxiv.org/abs/2006.11239), [DDIM, Song et al. (2020)](https://arxiv.org/abs/2010.02502), [Classifier-Free Diffusion Guidance, Ho & Salimans (2022)](https://arxiv.org/abs/2207.12598). The guidance convention here is null+s*(class-null), so s=1 is conditional. This experiment does not reproduce these papers' architectures, scale, benchmarks or image quality.

## Reproduce

The live site needs no installation. Local app: serve this folder with `python -m http.server 8000` and open localhost:8000. To recreate training/evidence:

```sh
python -m pip install -r requirements.txt
python download.py
python train.py
python train_guided.py
python evaluate_guided.py
OPENBLAS_NUM_THREADS=1 python export_sampling.py
python -m unittest -v
node test_engine.cjs
node test_sampling.cjs
```

Tested with PyTorch 2.14.1+cpu, two training threads. Cross-CPU/library kernels can cause weight and score drift; exact training-byte identity is not promised. Independent NumPy sampler fixtures use exported weights in float64 and pure Python replication of the sampling RNG. `noise-probes.npz` is a local numeric review artifact created by `evaluate_guided.py`, not a browser asset. Models are static JSON, never pickle.

Seven Python tests cover original source hashes, schedule, baseline fixtures/loss, guided fixtures/null conditioning, fixed compute budget and full paired cluster bootstrap recomputation. JS tests cover original stochastic reverse process, seeded Gaussian moments/replay, all guided/null PyTorch fixtures, all four full independent NumPy DDIM trajectories with different count/scale/blend, clipping/finite endpoints, exact replay, s=0/s=1 identities, time grids and invalid inputs. Maximum guided PyTorch/double drift 1.69e-6; independent float64 complete sampler drift 1.64e-13 in the tested environment. Neither is a promise of cross-browser image-byte identity.

Browser harness: install `npm install --no-save playwright`, run `node browser-check.cjs`. Chrome defaults to `/usr/bin/google-chrome`; override `CHROME_PATH`. `BASE_URL` tests a live site; `SCREENSHOT_DIR` changes `/tmp` output. Tests cover real generation, scale1 identical pair, scale3 differing pair, exact replay, scrubbing, PNG download, invalid/empty seeds, stop/restart, original-mode restriction, blend changes, 320/390px overflow and runtime errors.

## Privacy and security

No accounts, API keys, upload forms, cookies, storage, analytics or remote inference. Same-origin scripts/model/worker under a restrictive CSP. Dynamic text uses safe DOM setters. Request IDs prevent stale messages overwriting new results. Seed/range/class/budget validation exists both in page and worker/math where relevant. Local PNG Blob URLs are revoked. Reload discards outputs. No external runtime dependencies, generated-image library, hidden result selection or authorship claim.
