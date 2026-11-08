# Grain

Conditional pixel diffusion, trained from scratch on MNIST and sampled locally in the browser. A 14 x 14 grayscale digit image generator, not a general text-to-image service. It can produce rough, ambiguous or wrong digits. No hidden image gallery, pretrained model, remote inference or upscaling service.

## Run and reproduce

`python -m http.server 8000`, open localhost:8000. To reproduce training: `python -m pip install -r requirements.txt`, `python download.py`, `python train.py`, `python -m unittest -v`, `node test_engine.cjs`. Tested with PyTorch 2.14.1+cpu. CPU/library differences may affect exact training replay. Model source hashes, fixtures, weights and all loss measurements are included.

Data: [MNIST via CVDF](https://github.com/cvdfoundation/mnist), by Yann LeCun, Corinna Cortes and Christopher J.C. Burges. Training images and labels come from the CVDF mirror at https://storage.googleapis.com/cvdf-datasets/mnist/ . Original data are not bundled with the website. First 55,000 of the official 60,000 training images are training, last 5,000 validation. No test-set or perceptual-quality score is reported.

2 x 2 average pooling reduces 28 x 28 images to 14 x 14; scale to [-1,1]. Denoiser: 196 pixels + 16 sinusoidal timestep features + 10 one-hot digit features, then 256 SiLU, 256 SiLU, 196 predicted-noise outputs. Cosine schedule with 64 steps, offset 0.008, beta max 0.999. Conditional DDPM objective is noise prediction MSE. No U-Net, text encoder, latent autoencoder or classifier-free guidance.

Seed 431. AdamW 30,000 updates, batch 128, learning rate 0.0007, weight decay 0.0001, gradient clip 1. Validation loss is the mean of 12 batches of 256 held-out images with fresh noise from a separate seeded random generator. Final checkpoint used. Training loss is the rolling mean over 500 updates. Noise-prediction MSE is not an image-quality score. No FID or digit classifier accuracy was measured.

Sampling starts with 196 Gaussian values. The network predicts noise at each timestep. The predicted clean image is clipped to [-1,1], then the DDPM posterior mean/variance update produces the next image. Fresh seeded Gaussian noise is added except at step zero. A 32-bit LCG plus Box-Muller produces local sampling noise. Display clips values to [-1,1]. Same seed/checkpoint/class and numerical behavior replay the image. Small floating-point differences across browsers may affect exact bytes. Export is the current displayed stage, including a partial stage if stopped, enlarged to 280 x 280 pixels with nearest-neighbor sampling.

## Interaction, checks and security

Choose digit and seed. Generate, stop, scrub the computed trajectory, inspect snapshot buttons, save the displayed PNG. The worker emits actual intermediate arrays, not pre-rendered images. No account, API key, upload, analytics, storage, cookies or remote inference. Same-origin model, script and worker with restrictive CSP; dynamic strings use textContent. PNG export uses a local Blob URL and revokes it. Invalid seeds are rejected, not silently changed.

Python tests cover source hashes, schedule, PyTorch JSON parity and measured training improvement. JS tests cover denoiser parity, all finite reverse states, clean-state clipping, exact seeded replay and empirical Gaussian mean/variance. Browser tests cover loading, generation, replay, stop, reset sampling, scrubbing, invalid seeds, mobile layout and errors; the optional harness is included.

Reference: [Ho, Jain & Abbeel, Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239). This small MLP experiment does not reproduce their architecture, benchmarks or image quality.

Browser harness: install Node.js and `npm install --no-save playwright`, then `node browser-check.cjs`. Chrome defaults to `/usr/bin/google-chrome`; set `CHROME_PATH` to override. Screenshots save in `/tmp`; the server binds to localhost.
