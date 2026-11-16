# Subject-Invariant EEG Representation Learning for Emotion Recognition

A TensorFlow 2 / Keras 3 re-implementation of **SI-EEG** and **SIDA-EEG** (Rayatdoost, Yin, Rudrauf and Soleymani, ICASSP 2021) and of the baselines compared in the paper: EEG CNN, SINN, DANN and SIDANN.
The original Keras code was lost, so this code was rewritten from the paper.
Every detail the paper leaves open is a documented, configurable choice (see [Re-implementation decisions](#re-implementation-decisions) and [docs/DESIGN.md](docs/DESIGN.md)).

## What the paper does

EEG varies so much between people that it can identify them, and emotion recognizers trained on some participants, or on one database, generalise poorly to others.
The paper turns the log power spectral density of the EEG in six frequency bands into 32x32 spectral topographic maps and learns a CNN representation of them that predicts valence and arousal while hiding who the participant is.
A discriminator learns to tell in-group (training) participants from out-group (test) participants, and a gradient reversal layer (GRL) trains the encoder to defeat it (SI-EEG, Eq. 1); only the test participants' unlabeled EEG is needed.
SIDA-EEG (Eq. 3) adds a DANN database discriminator so that a model trained on DAI-EF transfers to MAHNOB-HCI. In the paper, SI-EEG gave the best results within DAI-EF and SIDA-EEG the best from DAI-EF to MAHNOB-HCI (Table 1).

| `--method` | Paper name | Adversaries attached to the representation (each through a GRL) | Lambda | Protocols |
| --- | --- | --- | --- | --- |
| `cnn` | EEG CNN [7] | none | - | within, cross |
| `sinn` | SINN [8] | subject identity (one class per subject, each subject weighted equally) | 0.2 | within, cross |
| `si_eeg` | **SI-EEG** (Eq. 1) | in-group vs out-group subjects (across databases: the test subject vs the source and the other target subjects) | 0.2 | within, cross |
| `dann` | DANN [9] | source vs target database | 0.1 | cross |
| `sidann` | SIDANN [8] | database + subject identity | 0.1 database, 0.15 subject | cross |
| `sida_eeg` | **SIDA-EEG** (Eq. 3) | database + in-group vs out-group subjects (as for `si_eeg`) | 0.1 database, 0.15 subject | cross |

The lambdas of SI-EEG and SIDA-EEG are the best values reported in the paper; the baselines reuse them, so that all methods are compared at the same adversarial strength (the paper does not give their lambdas).
Shared network (paper Sec. 5.1): 2x conv16, max-pool, 2x conv32, max-pool, FC256, FC128 (relu convolutions, tanh fully connected layers); an FC64 (tanh) emotion classifier and FC64 (elu) discriminators; Nadam with learning rate 0.002, beta_1 0.9, beta_2 0.999; 100 epochs; the best epoch is chosen on five validation subjects.

## Install

```bash
git clone https://github.com/soheilrayatdoost/Subject-Independent-EEG-Rep.git
cd Subject-Independent-EEG-Rep
pip install -e ".[dev]"       # Python >= 3.10, TensorFlow >= 2.16 (Keras 3), MNE, SciPy, scikit-learn, pandas
pytest -m "not slow"          # unit tests, synthetic data only
sieeg --help                  # or: python -m sieeg --help
```

For an NVIDIA GPU on Linux, install TensorFlow's GPU build as well: `pip install "tensorflow[and-cuda]"`.

## Quick check on synthetic data

The commands below run the whole workflow on synthetic EEG in a minute or two on a laptop CPU.
`configs/smoke.yaml` shrinks the network and trains for 3 epochs, so the scores mean nothing; it only shows that everything works.

```bash
sieeg extract synthetic --out features/syn_a.npz --dataset syn_a --n-subjects 8
sieeg extract synthetic --out features/syn_b.npz --dataset syn_b --n-subjects 6 --domain-shift 0.5 --seed 1
sieeg info features/syn_a.npz
sieeg bias features/syn_a.npz --other features/syn_b.npz --n-estimators 100 --n-splits 5
sieeg within features/syn_a.npz --method cnn,si_eeg --label valence --config configs/smoke.yaml --out runs-smoke
sieeg cross features/syn_a.npz features/syn_b.npz --method dann,sida_eeg --label valence \
    --config configs/smoke.yaml --out runs-smoke
sieeg report runs-smoke --reference si_eeg,sida_eeg
```

`sieeg extract synthetic --raw` renders raw 32-channel signals instead and runs them through the real extraction pipeline (referencing, resampling, Welch, band power).

## Data preparation

`sieeg extract` reads the raw EEG once and writes one `.npz` feature file per database: the log10 band power of every trial (32 channels x 6 bands, one sample per trial), with subject ids, trial ids and ratings.
All experiments start from these files.
Each trial is re-referenced to the average of the 32 channels and resampled to 256 Hz; its Welch PSD (15 s Hann segments, 50 % overlap) is averaged within delta (1-4 Hz), theta (4-8 Hz), slow alpha (8-10 Hz), alpha (8-12 Hz), beta (12-30 Hz) and gamma (30-45 Hz), and the log10 of each band mean is the feature.
Every band must lie below the Nyquist frequency of the resampled signal and of the recording: a `features.resample_sfreq` below twice the highest band edge (90 Hz for gamma at 30-45 Hz) is an error, and a trial recorded below that rate is skipped with a warning and listed as skipped in the feature file, because upsampling cannot restore frequencies the recording lacks.
Trials the loaders drop (missing file, crop outside the recording, unreadable BDF, rated MAHNOB session without a BDF) are recorded as skipped in the feature file together with the extractor's own skips (see `sieeg info`).
`sieeg extract` does not replace an existing feature file unless `--overwrite` is given.

### MAHNOB-HCI

MAHNOB-HCI (Soleymani et al., 2012) has 30 participants who rated 20 emotional clips for valence and arousal on 1-9 scales (SAM).
It is distributed by its maintainers through the official portal (<https://mahnob-db.eu/hci-tagging/>), which currently returns HTTP 404, so access may require contacting the maintainers.
Keep the folder layout of the download:

```text
MAHNOB-HCI/
  Sessions/
    <session id>/
      session.xml                          # subject id and the feltVlnc / feltArsl ratings
      Part_<p>_S_Trial<k>_emotion.bdf      # 32 EEG channels, peripheral signals and Status
    ...
```

```bash
sieeg extract mahnob /data/MAHNOB-HCI --out features/mahnob.npz     # --subjects 1,2 extracts a subset
```

Only rated emotion sessions (`session.xml` has `feltVlnc` and `feltArsl`) are used; sessions without a BDF file, and sessions whose `session.xml` cannot be read, are skipped with a warning and listed as skipped in the feature file, so one damaged session does not stop the extraction.
Each trial is cropped to the stimulus using the `Status` channel (rising edges of the trigger bit with value 16); if those markers are missing or implausible, the 30 s recorded before and after each clip are cut off instead (`--pad-sec`).

### DAI-EF, or any other dataset, through a manifest

DAI-EF (Rayatdoost et al., ICASSP 2020) has 60 participants who watched 40 clips and rated valence and arousal on 1-9 SAM scales.
DAI-EF is a private database: it is not publicly available, and only the code is published here.
Without it, the same experiments run on other data: leave-one-subject-out within MAHNOB-HCI (`sieeg within features/mahnob.npz ...`), or between MAHNOB-HCI and any other emotion database you prepare as below (`sieeg cross SOURCE TARGET ...`).
DAI-EF's recordings, like those of any other dataset, are described by a CSV manifest with one row per trial:

```csv
subject,trial,path,valence,arousal,tmin,tmax
P01,1,raw/P01.bdf,7,3,12.0,72.0
P01,2,raw/P01.bdf,2,6,95.5,155.5
P02,1,arrays/P02_t1.npy,5,5,,
```

| Column | Required | Meaning |
| --- | --- | --- |
| `subject` | yes | participant id, kept as text (`01` stays `01`) |
| `trial` | yes | trial or clip id |
| `path` | yes | EEG file, absolute or relative to the manifest's folder: anything MNE reads (`.bdf .edf .gdf .fif .set .vhdr .cnt` ...) or an array file (`.npy .npz .mat`) |
| `valence`, `arousal` | yes | self-reported ratings (1-9); leave empty if missing |
| `tmin`, `tmax` | no | crop `[tmin, tmax)` in seconds from the start of the file, e.g. one clip of a continuous recording; several rows may share a file |
| `sfreq` | no | sampling rate of an array file |
| `ch_names` | no | channel names of an array file, separated by `;` |
| `array_key` | no | variable holding the EEG in a `.npz` / `.mat` file (default: `data`, else the only 2-D array) |
| `transpose` | no | `true` if the array is (n_times, n_channels) |
| `scale` | no | multiplier to volts, e.g. `1e-6` for microvolts |

Column names must be spelled exactly as above, and each only once; spaces around them are ignored, so `tmin` with a space before it is still the column `tmin` (and a second `tmin`).
A known column in another case or spelling (`Tmin`, `Scale`, `array key`) or a common alias (`onset`/`start`, `offset`/`end`/`stop`, `fs`/`srate`/`sampling_rate`, `arraykey`/`key`, `channels`) stops `sieeg extract` with an error naming the expected column, because a column that is not found silently falls back to its default (the whole file, no scaling).
Other columns, such as notes, are ignored with a warning.
Spaces around ids and paths are removed, so `S01` followed by a space is `S01`.
Text cells (ids, paths, `ch_names`, `array_key`) are otherwise read as written, so `NA` or `None` is a value, not a missing one; leave a cell empty when it has no value (from R: `write.csv(..., na = "")`).
Leave a rating empty when it is missing; any other non-numeric rating (`7.5.`, `-`) is an error.
Rows that read the same samples as an earlier row (same file, variable and crop, e.g. forgotten `tmin`/`tmax` for a shared recording) are reported in a warning, since their features would be identical.

```bash
sieeg extract manifest daief.csv --dataset daief --out features/daief.npz
# array files that store neither the sampling rate nor the channel names (example values):
sieeg extract manifest daief.csv --dataset daief --out features/daief.npz \
    --sfreq 256 --ch-names Fp1,AF3,F3,F7,... --scale 1e-6
```

Channels are matched by name (case-insensitive) to the 32 BioSemi-32 channels of `features.channels` in `configs/paper.yaml`, which both databases contain.
MNE-readable files keep only their EEG channels; if they use other labels (for example BioSemi's `A1`-`B32`), save the data as arrays with the 10-20 names in `ch_names`.
`--dataset` names the database in every later output (run directories, the database label of Experiment 1), so use the same name throughout.
It may not contain `:`, which separates database and subject in the dataset-qualified subject ids (`daief:S01`) that keep the subjects of two databases apart; subject ids may contain it.

**The ten excluded DAI-EF participants.**
The paper analyses 50 of the 60 DAI-EF participants: ten were excluded for high noise and incomplete recordings, and the paper states that their exact IDs were published in this repository.
That list was lost with the original code.
If you know which participants to exclude, leave them out of the manifest, or pass their ids to `bias` and `within` with `--exclude-subjects` and to `cross` with `--exclude-source`, as written in the manifest's `subject` column:

```bash
EXCLUDED="id1,id2,id3,id4,id5,id6,id7,id8,id9,id10"     # replace with the ten DAI-EF subject ids
# or keep one id per line in a file: EXCLUDED=$(paste -sd, daief_excluded.txt)
```

An id that is not in the feature file is reported with a warning that lists the file's subject ids, so a misspelled id (`7` vs `P07`) does not go unnoticed.

### Check the features

```bash
sieeg info features/daief.npz
```

prints the number of trials and subjects, channels, bands, the range of the log10 band power, extraction settings, skipped trials, the share of high valence and arousal ratings (rating > 5), and the trials and high ratings of every subject.
For EEG in volts the log10 band power is strongly negative (PSD in V^2/Hz); values around 0 suggest microvolts read as volts (set `scale` in the manifest or `--scale 1e-6`).
The per-subject z-score of the deep models removes such constant factors, but the unnormalised analysis of Experiment 1 does not.
`sieeg bias` warns when the median log10 band power of the two databases differs by more than 2 decades, or when they were extracted with different settings.

## Reproduce the paper

The defaults are the paper's settings (`configs/paper.yaml`): 100 epochs, Nadam(0.002, 0.9, 0.999), batch size 32, the best epoch selected on five validation subjects, lambda 0.2 for SI-EEG and 0.15 (subject) / 0.1 (database) for SIDA-EEG.
Ratings above 5 form the high-arousal / positive-valence class; a rating of 5 is low / negative.
Add `--jobs N` to train N folds in parallel processes.

**Experiment 1: subject and database information in the features.**
Random forests (500 trees, 10-fold cross-validation) on the 192 log band-power features identify the participant among the 50 DAI-EF subjects and the database (DAI-EF vs MAHNOB-HCI), without and with per-subject z-scoring (fitted inside every fold on each participant's training trials, or on all of its trials when the fold holds the participant out entirely).
The larger database is undersampled so that both have the same number of trials (`--no-balance` keeps every trial).
Database identification is reported twice.
With trial-level folds, as in the paper, every participant has trials in both the training and the test folds and belongs to one database, so the forest can name the database by recognising the person: this row is comparable with the paper's 75 %, but it is close to 100 % even for two databases that do not differ.
With subject-grouped folds no participant is in both, so only this row measures the database bias itself, with chance at 50 % (its balanced accuracy is the more robust number).
Per-subject z-scoring removes any offset that all participants of a database share, including a units mismatch, so the z-scored rows show what is left after the paper's normalisation.
Scores are printed and written to the CSV in percent (columns `accuracy_pct`, `balanced_accuracy_pct`, `chance_uniform_pct`, ...; `cv` is `trial` or `subject`).

```bash
sieeg bias features/daief.npz --other features/mahnob.npz --exclude-subjects "$EXCLUDED" --out results/bias.csv
```

**Experiment 2, within DAI-EF: leave one subject out** (3 methods x 2 labels = 6 runs of 50 folds).
The test subject's unlabeled EEG is SI-EEG's out-group and one of SINN's identity classes; its labels only score the predictions.
The test subject fills half of every identity batch, so SINN weights those rows by 1 / (number of training subjects): it counts like one training subject.

```bash
sieeg within features/daief.npz --method cnn,sinn,si_eeg --label both --exclude-subjects "$EXCLUDED" --out runs
```

**Experiment 2, between databases: DAI-EF (labeled source) to MAHNOB-HCI (unlabeled target).**
Validation subjects come from DAI-EF.
`si_eeg` and `sida_eeg` train one model per MAHNOB-HCI subject: that subject's trials are the out-group, while DAI-EF and the other MAHNOB-HCI subjects are in-group, because Eq. 3 sums the subject loss over source and target data.
`cnn`, `dann` and `sidann` train one model for the whole target database.
Source and target must be two different databases: `sieeg cross` stops if both have the same dataset name or share a trial, and warns if the extraction settings recorded in the two feature files differ.

```bash
sieeg cross features/daief.npz features/mahnob.npz --method cnn,si_eeg,dann,sidann,sida_eeg --label both \
    --exclude-source "$EXCLUDED" --out runs
```

**Table 1 with t-tests.**

```bash
sieeg report runs --reference sida_eeg,si_eeg --out results/table1.md
```

The report has a Table-1 style results table (CR and weighted F1 of valence and arousal per protocol, test and training database, method, tag and lambdas) and paired tests across test subjects (t-test and Wilcoxon signed-rank test) of each reference method against every other method on the same protocol, databases, tag and label.
`--metric cr` tests the classification rate instead of F1; `--source subject_mean` averages per-subject scores instead of pooling all test trials.
A row averages only the runs that share all of these, normally different seeds: `n_runs` is the most runs behind one of its scores and `n_subjects` the number of test subjects of its runs.
Runs with other lambdas or another source database get their own rows, and a warning names the run directories when two runs of a row have the same seed (e.g. fold shards saved under two `--out` folders) or test different subjects (e.g. a `--folds` check next to a full run).
Paired tests compare runs of the same `--tag` only: run the baselines with the same tag to test a tagged variant against them.
With `--probe` runs, the report also lists the subject probes (see [Configuration](#configuration)).

**Lambda sweep.** Each value gets its own run directory through `--tag`, and its own row in the results table:

```bash
for lam in 0.05 0.1 0.15 0.2 0.3 0.5; do
  sieeg within features/daief.npz --method si_eeg --label both --exclude-subjects "$EXCLUDED" \
      --lambda-subject "$lam" --tag "lam$lam" --out runs-sweep
done
sieeg report runs-sweep
```

The sweep has no paired tests (each tag holds only `si_eeg`); compare its rows, then test the chosen lambda against the baselines with an untagged run or with baselines of the same tag.
For SIDA-EEG, nest a loop over `--lambda-domain` inside one over `--lambda-subject` with `sieeg cross ... --method sida_eeg --tag "ls$ls-ld$ld"`.
Choosing lambda by test scores is optimistic: `folds.csv` records each fold's validation score (`best_val`), which allows choosing without test labels.

**Seeds.** `--seed` changes the validation subjects, the initialisation and the batch order; every seed has its own run directory (`seed<N>`).

## Outputs

Every (method, label) pair of `within` and `cross` is one run directory `<out>/<protocol>-<test dataset>/<method>[-<tag>]/<label>/seed<seed>/`:

```text
runs/
  within-daief/si_eeg/valence/seed0/
  within-daief/si_eeg-lam0.1/valence/seed0/       # --tag lam0.1
  cross-mahnob/sida_eeg/arousal/seed0/
    summary.json          protocol, method, tag, label, datasets, lambdas, seed, numbers of folds, subjects
                          and trials, CR and F1 pooled over test trials and as mean +- sd over test subjects,
                          subject-probe means (null unless protocol.subject_probe)
    predictions.csv       one row per test trial: fold, dataset, subject, trial, rating, y_true, y_pred, p1
    folds.csv             one row per fold: test and validation subjects, set sizes, best epoch,
                          best validation score (best_val, in the units of history/), CR, F1, seconds,
                          subject-probe scores probe_subject_acc, probe_subject_chance, probe_group_bacc
                          (fractions; empty unless protocol.subject_probe)
    history/<fold>.csv    per epoch: lambdas, losses and accuracies of the classifier and each adversary,
                          validation loss, accuracy and F1 (accuracies and F1 as fractions, not %)
    config.yaml           the resolved configuration of the command (reusable with --config); its features.*
                          section does not describe the feature file (see features.json)
    features.json         each input feature file: path, SHA-256, dataset, excluded and remaining subjects,
                          channels, bands and the file's metadata, including its extraction settings
    command.json          the command line, parsed arguments, working directory and sieeg version
```

`summary.json` is written last, so a directory that has one holds a complete run.
A command checks all its runs before training anything: a run directory that already has a `summary.json` stops it unless `--overwrite` is given, and so do `--folds` subjects without a rating for one of the labels, or too few subjects for the validation draw.
While a command trains, each of its run directories holds a `.running` file (pid, host, command line); a second `sieeg` process that would write the same directory, e.g. a cluster job started twice, stops instead of overwriting it, unless `--overwrite` is given.
If a process was killed, delete its `.running` file before running the same command again.
Each run prints one line with its scores and its directory; progress and warnings are logged to stderr (`-q` for warnings only, `-v` for debug output and tracebacks, before or after the command: `sieeg -q within ...`).
Errors such as a missing file or an invalid configuration value are reported in one line (`sieeg within: error: ...`).

## Configuration

The built-in defaults are the paper's values, as in `configs/paper.yaml`.
Precedence, from lowest to highest: defaults, `--config FILE.yaml`, `--set section.key=value` (repeatable), and the convenience flags `--epochs`, `--seed`, `--jobs`, `--lambda-subject`, `--lambda-domain`, `--probe`.

```bash
--set train.batch_size=64 --set protocol.n_val_subjects=3
--set train.lambda_schedule=dann           # DANN ramp 2 / (1 + exp(-10 p)) - 1 instead of a constant lambda
--set train.class_weight=balanced          # weight the emotion loss by inverse class frequency
--set train.optimizer=nadam                # Keras 3 Nadam instead of the Keras-2 one
--set protocol.cross_per_subject=never     # SI-EEG / SIDA-EEG with all target subjects as one out-group (the subject
                                           # adversary then duplicates the database adversary; a warning says so)
--set protocol.subject_probe=true          # or --probe: linear probes of subject information left in each fold's
                                           # representation (not in the paper)
```

Every value is checked when it is read, including the convenience flags: a wrong type (`train.epochs=1O`), a number that is not finite (`protocol.label_threshold=.nan`, `--lambda-subject nan`), a value out of range (`train.epochs=0`, `train.beta_2=1`, `features.welch_overlap=1`) or an unknown option (`protocol.normalization=zscore`) stops the command with an error that names the key, before any data is loaded.

**Subject probes** (`--probe`, not in the paper): after each fold, a cross-validated logistic regression decodes the subject of the validation trials from the 128-d representation (`probe_subject_acc` vs the majority rate `probe_subject_chance`), and another separates the in-group trials (training and untested target trials) from the test subject's (`probe_group_bacc`, chance 0.5).
Lower values mean less subject information; no emotion label is used.
`sieeg report` lists their means per method and label.

Feature settings (`features.*`) take effect when features are extracted, except `grid_size`, `interpolation` and `montage`, which shape the topographic maps built by `within` and `cross`.
The `features` section of a run's `config.yaml` therefore describes the command, not the feature file; the file's own extraction settings, hash and subjects are in the run's `features.json`.
The Python API underneath the command line:

```python
from sieeg.config import load_config
from sieeg.evaluation.protocols import run_within_database
from sieeg.evaluation.report import save_run
from sieeg.features import FeatureSet

cfg = load_config("configs/paper.yaml", ["train.epochs=50"])
fs = FeatureSet.load("features/daief.npz").drop_subjects(["id1", "id2"])
result = run_within_database(fs, "si_eeg", "valence", cfg)
save_run(result, "runs/api/si_eeg/valence", config=cfg)
print(result.summary["pooled"])           # {'cr': ..., 'f1': ...}
```

## Re-implementation decisions

The main choices, each a configuration key (all of them, with reasons, in [docs/DESIGN.md](docs/DESIGN.md), Sec. 2):

- **One sample per trial (D1).** The paper's "15 seconds time window with a 50% overlap through the Welch method" is read as Welch parameters: one PSD per trial over the whole stimulus, from 15 s Hann segments with 50 % overlap. This matches its single-trial classification "to avoid mixing samples from training and testing sets".
- **Gamma is 30-45 Hz (D2).** The paper gives gamma as ">30 Hz"; 45 Hz stays clear of 50 Hz mains noise. Band power is log10 of the mean PSD in each band (D3).
- **Common montage (D4, D5).** The 32 BioSemi-32 channels that both databases have, common average reference, 256 Hz.
- **Topographic maps (D6, D7)** as in EEGLearn (Bashivan et al., ICLR 2016): azimuthal-equidistant projection of the electrodes and Clough-Tocher interpolation onto 32x32 pixels, one image channel per band. Every feature is z-scored per participant first; test participants are normalised with their own unlabeled statistics, as in the paper.
- **Keras-2 Nadam, re-implemented for Keras 3 (D12).** The learning rate 0.002 is the default of the standalone Keras-2 Nadam, which warms up its momentum with `schedule_decay=0.004`. Keras 3's Nadam has no `schedule_decay` and its warm-up is about 250x faster, so `LegacyNadam` reproduces the Keras 2.2.4 update.
- **Out-group per target subject (D14).** Within a database the out-group is the held-out test subject. Across databases SI-EEG and SIDA-EEG train one model per target subject, whose trials form the out-group; the subject discriminator also sees the other target subjects, labeled in-group, because Eq. 3 sums the subject loss over source and target samples. Contrasting the source with the test subject alone would make its labels the database labels, so SI-EEG would be a DANN on that subject; `--set protocol.cross_per_subject=never` (whole target as out-group) has that effect and logs a warning.
- **Identity adversary (D15).** SINN and SIDANN classify the training subjects and the pool subjects (the target subjects across databases, the test subject within a database). Pool samples are down-weighted by (number of pool subjects) / (number of training subjects), so every subject has the same influence on the identity loss. Within DAI-EF, for example, the test subject fills half of every identity batch but counts as much as a single training subject.
- **Validation-based model selection (D16).** Five validation subjects are drawn (seeded) from the non-test training subjects, and the epoch with the best validation weighted F1 is kept. Test and target labels are never used for training, batch sampling, model selection or class weights; they only score the predictions.
- **Constant lambda inside the GRL (D11).** Lambda only scales the reversed gradient; the total loss is the unweighted sum of the classifier and discriminator losses.

## Expected compute

A training step of the full-size model takes about 87 ms on a 4-core laptop CPU.
Within DAI-EF, a fold trains on 44 subjects x 40 clips, about 1,760 trials or 55 steps per epoch; 100 epochs are about 5,500 steps, roughly 8 minutes.
A full leave-one-subject-out run (50 folds) is therefore roughly 7 hours per method and label.
Between databases, `cnn`, `dann` and `sidann` train a single model, while `si_eeg` and `sida_eeg` train one per MAHNOB-HCI subject.
Each SI-EEG step also encodes a batch from the target database, as SIDA-EEG does.
Full LOSO runs are best done on a GPU, or with `--jobs N` to train N folds in parallel processes; `--folds` and `configs/smoke.yaml` give quick checks.

## Results reported in the paper

Table 1 of the paper (CR: classification rate, F1: weighted F1, both in %):

| Protocol | Method | Test data | Valence CR | Valence F1 | Arousal CR | Arousal F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Within database | EEG CNN [7] | DAI-EF | 68.0 | 66.3 | 60.4 | 60.3 |
| Within database | SINN [8] | DAI-EF | 71.1 | 69.4 | 63.2 | 63.0 |
| Within database | SI-EEG | DAI-EF | 72.8 | 71.2 | 65.0 | 64.8 |
| Between databases | EEG CNN [7] | MAHNOB | 64.8 | 62.9 | 50.3 | 50.0 |
| Between databases | SI-EEG | MAHNOB | 65.2 | 64.2 | 52.1 | 52.0 |
| Between databases | DANN [9] | MAHNOB | 68.3 | 66.9 | 55.3 | 55.0 |
| Between databases | SIDANN [8] | MAHNOB | 69.0 | 67.8 | 56.8 | 56.5 |
| Between databases | SIDA-EEG | MAHNOB | 69.8 | 68.5 | 57.6 | 57.2 |

Experiment 1 of the paper: 52 % subject identification among the 50 DAI-EF participants (chance 2 %) and 75 % database identification, DAI-EF vs MAHNOB-HCI (chance 50 %; compare it with the trial-level database rows of `sieeg bias`).
The paper reports that SIDA-EEG significantly outperforms SIDANN (t-test, p < .05).

Results of this re-implementation will differ.
The original code, the list of excluded DAI-EF participants and several details are lost or unspecified (see the decisions above), and deep models trained on a few thousand EEG trials vary with the seed.
Compare methods on your own runs, ideally over several seeds.

References of the baselines: [7] S. Rayatdoost and M. Soleymani, "Cross-corpus EEG-based emotion recognition", MLSP 2018. [8] Y. Yin et al., "Speaker-invariant adversarial domain adaptation for emotion recognition", ICMI 2020. [9] Y. Ganin et al., "Domain-adversarial training of neural networks", JMLR 2016.

## Related code

The PyTorch implementation of SIDANN by our colleagues uses almost the same adversarial scheme:
<https://github.com/intelligent-human-perception-laboratory/Speaker-Invariant-Domain-Adversarial-Neural-Networks>.
It is the official code of the SIDANN baseline [8], not of SI-EEG.

## Citation

```bibtex
@inproceedings{rayatdoost2021subject,
  title     = {Subject-Invariant {EEG} Representation Learning for Emotion Recognition},
  author    = {Rayatdoost, Soheil and Yin, Yufeng and Rudrauf, David and Soleymani, Mohammad},
  booktitle = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2021}
}
```

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (GPL-3.0-or-later); see [LICENSE](LICENSE).
