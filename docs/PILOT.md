# Pilot

Author: Huda Mueen. Pilot seeds 1000 to 1009 only, reference environment only. Evaluation seeds are never used here. The rule and the grid are in configs/pilot_grid.json and docs/PREDICTIONS.md, and every change to them is in the amendment log. This file is written by scripts/render_pilot.py from the files in results/pilot/.

## pendulum

Chosen budget: {"steps": 7000} with pilot rate 0.60.

Grid entries with the final settings:

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| pendulum.steps3000 | 10 | 0.00 | -1440 | -1535 | -1313 |
| pendulum.steps4500 | 10 | 0.00 | -1069 | -1187 | -867 |
| pendulum.steps6000 | 10 | 0.00 | -423 | -916 | -286 |
| pendulum.steps7000 | 10 | 0.60 | -213 | -678 | -196 |
| pendulum.steps9000 | 10 | 1.00 | -181 | -214 | -178 |

## cartpole

Chosen budget: {"steps": 40000} with pilot rate 0.30.

Grid entries with the final settings:

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| cartpole.steps15000 | 10 | 0.10 | 114 | 90 | 459 |
| cartpole.steps20000 | 10 | 0.20 | 155 | 96 | 500 |
| cartpole.steps25000 | 10 | 0.20 | 144 | 64 | 500 |
| cartpole.steps30000 | 10 | 0.20 | 143 | 34 | 500 |
| cartpole.steps40000 | 10 | 0.30 | 162 | 43 | 500 |

Entries of the original grid, superseded by an amendment (earlier settings, partial entries included):

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| cartpole.steps10000 | 10 | 0.00 | 74 | 13 | 266 |
| cartpole.steps18000 | 10 | 0.00 | 129 | 17 | 253 |
| cartpole.steps28000 | 4 | 0.00 | 80 | 13 | 225 |

Exploration runs (a few pilot seeds each):

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| cartpole.obs_noise0.02_steps20000 | 4 | 0.00 | 134 | 19 | 159 |
| cartpole.obs_noise0.05_steps20000 | 4 | 0.00 | 173 | 119 | 241 |
| cartpole.obs_noise0.0_steps20000 | 4 | 0.25 | 230 | 147 | 500 |

## invpend

Chosen budget: {"steps": 20000} with pilot rate 0.50.

Grid entries with the final settings:

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| invpend.steps12000 | 10 | 0.10 | 64 | 30 | 1000 |
| invpend.steps16000 | 10 | 0.10 | 100 | 40 | 1000 |
| invpend.steps20000 | 10 | 0.50 | 669 | 47 | 1000 |

Entries of the original grid, superseded by an amendment (earlier settings, partial entries included):

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| invpend.steps10000 | 10 | 0.00 | 56 | 34 | 70 |
| invpend.steps15000 | 1 | 0.00 | 63 | 63 | 63 |
| invpend.steps4000 | 10 | 0.00 | 57 | 43 | 78 |
| invpend.steps7000 | 10 | 0.00 | 51 | 38 | 89 |

Exploration runs (a few pilot seeds each):

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| invpend.batch256_lr0.0003_steps10000 | 3 | 0.00 | 52 | 49 | 59 |
| invpend.batch256_lr0.001_steps10000 | 3 | 0.00 | 68 | 43 | 336 |

## grok

Chosen budget: {"train_frac": 0.5, "steps": 3000} with pilot rate 0.30.

Grid entries with the final settings:

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| grok.train_frac0.45_steps3000 | 10 | 0.10 | 0.529 | 0.296 | 0.928 |
| grok.train_frac0.5_steps3000 | 10 | 0.30 | 0.553 | 0.347 | 1.000 |

Entries of the original grid, superseded by an amendment (earlier settings, partial entries included):

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| grok.train_frac0.4_steps2000 | 10 | 0.00 | 0.380 | 0.287 | 0.404 |
| grok.train_frac0.4_steps4000 | 10 | 0.00 | 0.382 | 0.347 | 0.404 |
| grok.train_frac0.4_steps8000 | 2 | 0.00 | 0.393 | 0.388 | 0.397 |
| grok.train_frac0.5_steps2000 | 10 | 0.00 | 0.477 | 0.423 | 0.521 |
| grok.train_frac0.5_steps4000 | 10 | 0.00 | 0.483 | 0.438 | 0.528 |
| grok.train_frac0.6_steps2000 | 10 | 0.00 | 0.568 | 0.547 | 0.608 |
| grok.train_frac0.6_steps4000 | 10 | 0.00 | 0.578 | 0.547 | 0.627 |

Exploration runs (a few pilot seeds each):

| File | Seeds | Success rate | Median score | Min | Max |
|---|---|---|---|---|---|
| grok.lr0.003_p13_steps3000_train_frac0.5 | 4 | 0.00 | 0.388 | 0.212 | 0.494 |
| grok.lr0.003_p13_steps3000_train_frac0.7 | 4 | 1.00 | 0.990 | 0.961 | 1.000 |
| grok.lr0.003_p23_steps3000_train_frac0.5 | 4 | 0.25 | 0.553 | 0.472 | 1.000 |
