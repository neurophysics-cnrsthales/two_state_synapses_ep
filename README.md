# Two-State Dynamical Synapses as a Physical Substrate for Equilibrium Propagation

This code runs the simulations for the proposed two-state dynamical synapses framework. We consider two variants: Variant A, the energy-coupled formulation, where the effective weight is the sum of the short-term and long-term states; and Variant B, the hardware-oriented decoupled formulation, where only the long-term state is used for neuron updates.

For each variant, we evaluate two long-term update rules: phase-based, corresponding to standard phase-based EqProp, and continual, corresponding to continual EqProp.

Experiments are performed on MNIST using an MLP with one hidden layer of 512 units. The code reproduces the results reported in the article.

## Getting started

### 1. Requirements

* Python 3 (tested with 3.10; other versions are untested) and about 100 MB of disk space (MNIST is ~80 MB once extracted).
* GPU is optional but recommended.
* Packages (`requirements.txt`): `torch`, `torchvision`, `numpy`, `pandas`, `tqdm`.
  Tested with torch 2.6.0 (CUDA 11.8), torchvision 0.21.0, numpy 1.23.5, pandas 1.5.3, tqdm 4.64.1.

### 2. Installation

Install PyTorch first. Pick the command for your system on https://pytorch.org/get-started/locally/, for example

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # NVIDIA GPU, CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # no GPU
```

then the remaining packages:

```bash
pip install -r requirements.txt
```

finally, check the installation:

```bash
python -c "import torch, torchvision; print(torch.__version__, 'GPU available:', torch.cuda.is_available())"
```

## Files

| File | Content |
|---|---|
| `main.py` | entry point: reads a `config.json`, trains, writes `results.csv` and model weights |
| `network.py` | `MacDynamic` implements Variant A (`mac = total`) and Variant B (`mac = partial`); `InstarD` is the dynamic weight update |
| `train_test.py` | training / evaluation loops (phased update `train_dyn_ep`, continuous update `train_dyn_sequential_ep`) |
| `tool.py`, `data.py` | activation function, optimiser, results table; MNIST loader (data kept on the device) |
| `config/best/` | the four configurations used for expriments: phased / continuous x Variant A / B (784-512-10) |
| `METHODS_AND_HYPERPARAMETERS.md` | methods, hyper-parameters, selection protocol and results (file paths in it refer to the original `model/` folder) |

## Notation

M = static (trained) weights (code: `net.W`), w = dynamic weights (code: `net.dyn`, reset to 0 for every mini-batch),
g = `w_scale`. 

Variant A: the neurons use the effective weight M + g*w; Variant B: M only (w is only used for the update).


In a configuration `"lr": [a, b]` the first entry is the hidden layer and the second the output layer
(for two hidden layers: `[first hidden, second hidden, output]`); `fcLayers` sets the architecture.

## Run

One command per method; each command trains the method once (seed 0) and writes its results to `./results/<method>/`.
The output folder must be empty or new (`results.csv` is appended to, not overwritten); `--device -1` runs on the CPU.

```bash
# phased update (30 epochs)
python main.py --json_path ./config/best/phased_A --seed 0 --device 0 --out_dir ./results/phased_A
python main.py --json_path ./config/best/phased_B --seed 0 --device 0 --out_dir ./results/phased_B

# continual update (300 epochs)
python main.py --json_path ./config/best/continuous_A --seed 0 --device 0 --out_dir ./results/continuous_A
python main.py --json_path ./config/best/continuous_B --seed 0 --device 0 --out_dir ./results/continuous_B
```

Each run writes `results.csv` (one row per epoch; columns `Train_Error`, `Train_Loss`, `Weight_Norm`, `Test_Error`,
`Min_Test_Error`, as fractions) and the weights `model_state_dict*.pt`. The reported number is the test error after the last
epoch, i.e. the last line of `results.csv`:

```bash
tail -n 1 ./results/phased_A/results.csv
```

## Expected results (last-epoch test error %)

| Method | this command (seed 0) | mean ± SD over seeds 0-9 |
|---|---|---|
| phased A (30 epochs) | 2.17 | 2.01 ± 0.09 |
| phased B (30 epochs) | 1.99 | 1.98 ± 0.07 |
| continual A (300 epochs) | 2.14 | 2.24 ± 0.11 |
| continual B (300 epochs) | 2.24 | 2.30 ± 0.09 |

A single run differs from the 10-seed mean by about one standard deviation, so do not expect exactly the mean. Runs are
deterministic for a fixed seed on the same software and hardware. About 2 min per phased run and 20-30 min per continual run on
one RTX 4090.

## Options

Command line: `--json_path` (folder with `config.json`), `--device` (GPU index, -1 = CPU), `--seed`, `--out_dir`.

Keys of a `config.json` (all of them are read; nothing else is): `device`, `dataset` (`mnist`),
`epochs`, `batchSize`, `test_batchSize`, `fcLayers` (e.g. `[784, 512, 10]` or `[784, 512, 512, 10]`), `lr` (one value per weight
layer, input side first), `activation_function` (`smooth_hardsigm`), `optimizer` (`SGD` or `Adam`), `continuous_update`
(`false` = phased update, `true` = continuous update), `seed`, `mac` (`total` = Variant A, `partial` = Variant B), `w_scale` (g),
`dt`, `T` (free-phase steps), `Kmax` (nudge-phase steps), `beta`.
Optional: `lr_scheduler` (`cosine` or `null`) with `lr_min`, `grad_clip` (per-layer norm clipping), `grad_clip_value`
(per-layer element-wise clipping), `momentum`, `weight_decay`, `val_split` (hold out that many training images as a validation
set, reported in the `Test_Error` column).


