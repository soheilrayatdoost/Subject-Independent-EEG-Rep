# Design: TensorFlow 2 re-implementation of SI-EEG / SIDA-EEG (ICASSP 2021)

This document is the contract for the code in `src/sieeg/`. Section 1 maps the
paper to code. Section 2 lists every decision the paper leaves open. Sections 3-8
specify each module precisely enough that it can be implemented and tested on its own.

Stack: Python >= 3.10, TensorFlow >= 2.16 with **Keras 3** (TensorFlow backend), MNE,
SciPy, scikit-learn, pandas. Training is a TF2 custom loop (`tf.GradientTape` +
`tf.function`). The gradient reversal layer is a `tf.custom_gradient`.

## 1. Paper -> code map

| Paper | Code |
|---|---|
| Sec. 3 MAHNOB-HCI (30 participants, 20 clips, SAM 1-9) | `data/mahnob.py` |
| Sec. 3 DAI-EF (50 of 60 participants, 40 clips) | `data/manifest.py` (generic CSV manifest) |
| Sec. 4 PSD by Welch, 15 s windows, 50 % overlap; log power in delta, theta, slow alpha, alpha, beta, gamma | `features/psd.py`, `features/extract.py` |
| Sec. 5.1 PSD -> 32x32 spectral topographic maps per band | `features/topomap.py` |
| Sec. 5.1 encoder: 2x conv16, maxpool, 2x conv32, maxpool, FC256, FC128 (relu convs, tanh FCs) | `models/networks.py` |
| Sec. 5.1 gradient reversal layer (GRL) | `models/grl.py` |
| Eq. 1 SI-EEG: emotion loss - lambda * in-group/out-group subject loss | `training/methods.py` (`si_eeg`), `training/trainer.py` |
| Eq. 3 SIDA-EEG: + lambda_d * database loss (DANN) | `training/methods.py` (`sida_eeg`) |
| Sec. 5.3 labels: rating > 5 is high/positive, 5 is low/negative | `FeatureSet.labels` |
| Sec. 5.3 per-participant z-score of features | `FeatureSet.normalized` |
| Sec. 5.3 / 6 Exp. 1: random forest, 10-fold CV, subject (50 classes) and domain bias | `analysis/bias.py` |
| Sec. 5.3 Exp. 2: leave-one-subject-out within DAI-EF; DAI-EF -> MAHNOB across databases; validation = 5 subjects; 100 epochs; Nadam(0.002, 0.9, 0.999) | `evaluation/protocols.py`, `training/optimizers.py` |
| Table 1: CR and weighted F1; t-test | `evaluation/metrics.py`, `evaluation/stats.py`, `evaluation/report.py` |
| Baselines EEG CNN [7], SINN / SIDANN [8], DANN [9] | `training/methods.py` (`cnn`, `sinn`, `sidann`, `dann`) |

## 2. Decisions the paper leaves open

The paper's best lambdas are used (SI-EEG 0.2; SIDA-EEG 0.15 subject, 0.1 domain).
Everything below is a re-implementation choice. Each one is a config key, so it can be changed.

| # | Topic | Choice | Why | Key |
|---|---|---|---|---|
| D1 | Welch "15 s window, 50 % overlap" | One PSD per trial: Welch over the whole stimulus with 15 s Hann segments and 50 % overlap. One sample per trial. | The paper's "single-trial classification ... to avoid mixing samples" means one sample per trial, and these are Welch parameters. | `features.welch_segment_sec`, `welch_overlap` |
| D2 | Gamma upper edge (">30 Hz") | 45 Hz | Stays clear of 50 Hz mains (London, Geneva). | `features.bands` |
| D3 | Band power | log10 of the mean PSD over [lo, hi) | "logarithms of PSD from six frequency bands". | - |
| D4 | Reference / rate | Common average over the 32 channels; resample to 256 Hz. | BioSemi CMS/DRL data have no reference; the MAHNOB authors re-reference to average. | `features.reference`, `resample_sfreq` |
| D5 | Channels | The 32 BioSemi-32 channels, which MAHNOB has and DAI-EF's 64 channels include. | Identical image geometry across databases. | `features.channels` |
| D6 | Topomap | EEGLearn (Bashivan et al. 2016): azimuthal-equidistant projection, grid over the electrodes' bounding box, Clough-Tocher cubic interpolation, 0 outside the hull. Rows: anterior to posterior. Columns: left to right. | Standard spectral-topography recipe; one triangulation is reused for all maps. | `features.grid_size`, `interpolation`, `montage` |
| D7 | Normalisation | Per-participant z-score of every (channel, band) feature *before* interpolation. The test subject is normalised with its own unlabeled statistics. Experiment 1 instead fits the z-score inside every cross-validation fold (Sec. 9), because there the subject is the class (subject identification) or nested in it (database identification); statistics over held-out trials would push the accuracy below chance. | Paper Sec. 5.3. Uses no labels; it is transductive, as in the paper. | `protocol.normalization` |
| D8 | Conv details | 3x3 kernels, `same` padding, no batch-norm or dropout. | VGG/EEGLearn style; not in the paper. | `model.kernel_size`, `padding`, `dropout` |
| D9 | Emotion head | FC64 (tanh) -> 2 logits. | Text: "a fully connected layer with 64 neurons for subject discriminator and emotion classifier". | `model.classifier_hidden` |
| D10 | Discriminators | GRL -> FC64 (elu) -> logits (1 for binary, n_subjects for identity). | Paper Sec. 5.3. | `model.discriminator_*` |
| D11 | Lambda | Constant, only inside the GRL; the total loss is the unweighted sum. A DANN ramp is optional. | Paper reports fixed best values; the SIDANN reference code does the same. | `train.lambda_schedule` |
| D12 | Optimiser | Keras-2 Nadam (lr 0.002, schedule_decay 0.004, eps 1e-7) re-implemented for Keras 3. | lr=0.002 is the standalone Keras-2 default. Keras 3 Nadam removed schedule_decay, which changes the momentum warm-up about 250x. | `train.optimizer` |
| D13 | Batch composition | Each step takes b labeled source samples S, plus b out-group samples O (group adversary, or identity without a target) and/or b target samples T (domain adversary, or a group or identity adversary when a target is given), drawn uniformly with replacement. One epoch is one pass over the labeled training set. | Balanced discriminator batches (as in DANN/SIDANN), even when the out-group is a single subject. With a target pool the group head sees S, O and T (D14). The multi-class identity head gets pool-row weights instead of balance (D15). | `train.batch_size` |
| D14 | Out-group | Within database: the held-out test subject; the group head contrasts S (in-group 0) with O (out-group 1). Across databases: the target test subject t (one model per target subject); the group head sees S, O and the whole target pool T, and a target sample is out-group (1) if it belongs to an out-group subject (t), in-group (0) otherwise. | Eq. 3 sums the subject loss over all M source and target samples, each with its in-group/out-group label. Contrasting S with t's trials alone would make the group label equal the domain label on every sample, so SI-EEG across databases would be exactly a DANN on t at lambda 0.2. `cross_per_subject=never` (out-group = every target trial) still makes the group labels equal the domain labels; that mode logs a warning. | `protocol.cross_per_subject`, `outgroup_includes_val` |
| D15 | Identity adversary (SINN/SIDANN) | Multi-class over labeled training subjects plus pool subjects: the target domain's subjects across databases, the test subject within a database. Uses pool subject IDs, never pool emotion labels. Pool rows weigh `alpha = n_pool_subjects / n_train_subjects` (distinct subjects) in the identity cross-entropy, `(sum_S CE + alpha * sum_P CE) / (n_S + alpha * n_P)`, so every subject counts equally in expectation; without a pool it is the plain mean over S. Within a database alpha = 1 / n_train_subjects; across databases alpha = n_target / n_train_source subjects (DAI-EF -> MAHNOB: 30/45). | Mirrors the official SIDANN code: the speaker head covers source and target speakers. Without the weight, D13's balanced batch makes the single within-database test subject half of every identity batch (44x a training subject at DAI-EF size), and SINN's loss would contain SI-EEG's balanced in-group/out-group loss. | - |
| D16 | Validation / model selection | 5 subjects drawn at random (seeded) from the non-test source subjects. Keep the epoch with the best validation weighted F1 (ties go to the earlier epoch). Never select on test or target labels. | Paper: "best model selected on the validation set (data from five subjects)". | `protocol.n_val_subjects`, `train.selection_metric` |
| D17 | Metrics | CR (accuracy) and weighted F1, pooled over all test trials *and* as mean ± sd over test subjects. Paired t-test (and Wilcoxon) across subjects. | Paper reports CR / weighted F1 and a t-test. | - |
| D18 | Exp. 1 domain chance level | Balance the two databases by undersampling the larger one, so both classes are equally frequent. Accuracy and balanced accuracy are reported for two cross-validations: trial-level stratified folds (the paper's "10-fold cross-validation", comparable with its 75 %) and subject-grouped folds (StratifiedGroupKFold over dataset-qualified subjects). 50 % is the null level only for subject-grouped folds: with trial-level folds every participant has trials in training and test and belongs to one database, so the forest names the database by recognising the person. Per-subject z-scoring removes any offset shared by a database's subjects, so the z-scored rows show what is left after the paper's normalisation. | Without balancing, the majority baseline is about 79 %, not 50 %. On synthetic databases with no database effect, trial-level folds give about 100 % and subject-grouped folds about 50 %. | `bias --no-balance` |

Test and target emotion labels never influence training, batch sampling, model
selection or class weights; they only score the predictions. Every loss is computed
from logits.

## 3. Data flow

```
raw files --loader--> Trial (volts, cropped to stimulus)
          --extract--> FeatureSet (.npz): log10 band power (n_trials, 32, 6) + subject/trial/ratings
          --topomap--> images (n_trials, 32, 32, 6) float32 NHWC   (per-subject z-score first)
          --protocol--> folds -> AdversarialTrainer -> predictions -> metrics -> run directory
```

## 4. Shared contracts (already implemented)

- `sieeg.config`: `ExperimentConfig(seed, features: FeatureConfig, model: ModelConfig, train: TrainConfig, protocol: ProtocolConfig)`,
  `load_config(path, overrides)`, `apply_overrides(cfg, ["train.epochs=5"])`, `save_config`, `dump_config`, `config_to_dict`,
  `config_from_dict`, `validate_config`, `CHOICES`, `MINIMUMS`, `POSITIVE`, `FRACTIONS`, `BIOSEMI32_CHANNELS`, `PAPER_BANDS`.
  Every value read from YAML or an override is checked: its field type (an `X | None` field also takes `null`;
  lists become tuples, integers floats; a number must be finite, so `.nan` and `.inf` are errors; a number field
  also takes a numeric string such as `1e-3`, which PyYAML reads as text), `MINIMUMS` (>= 0: `seed`,
  `features.l_freq`, `model.classifier_hidden`, `model.discriminator_hidden`, `train.schedule_decay`,
  `protocol.n_val_subjects`, `protocol.lambda_subject`, `protocol.lambda_domain`; >= 1: `model.kernel_size`, every
  filter count of `model.conv_blocks` and unit count of `model.dense_units`, `train.epochs`, `train.batch_size`,
  `protocol.jobs`), `POSITIVE` (> 0: `features.resample_sfreq`, `features.h_freq`, `features.welch_segment_sec`,
  `train.learning_rate`, `train.epsilon`), `FRACTIONS` (in [0, 1): `features.welch_overlap`, `model.dropout`,
  `train.beta_1`, `train.beta_2`) and `CHOICES` (optimizer, lambda_schedule, selection_metric, class_weight, reference,
  interpolation, normalization, cross_per_subject, padding); `NON_EMPTY` lists (`features.channels`,
  `features.bands`) need at least one entry, `features.channels` must not repeat a name ignoring case, `features.bands` must not repeat a band name (also
  checked by `FeatureSet`), every band needs `0 <= low < high`, and
  `features.grid_size` must leave the encoder a non-empty feature map (each conv block ends with a 2x2 max-pooling,
  so the paper's two blocks need >= 4). `validate_config` applies the same checks to a configuration built in
  Python, where a number given as a string (`welch_segment_sec="15"`) is an error; the CLI re-validates after its
  convenience flags (`--epochs`, `--seed`, `--jobs`, `--lambda-subject`, `--lambda-domain`). A wrong value or a
  section given as a scalar is a `ValueError` naming `section.key`, an unknown key a `KeyError`, and a YAML syntax
  error a `ValueError` naming the file, line and column.
- `sieeg.data.trial.Trial(dataset, subject, trial, data (n_ch, n_times) volts, sfreq, ch_names, valence, arousal, meta)`.
  A dataset name containing `:` is a `ValueError` here and in `FeatureSet` (`utils.check_dataset_name`).
  So are channel names that are not unique ignoring case (`utils.check_channel_names`); `FeatureSet.pick_channels`
  likewise rejects a channel requested twice.
- `sieeg.features.featureset.FeatureSet(dataset, band_power (n, n_ch, n_bands) log10, subjects, trials, valence, arousal, ch_names, bands, meta)`
  with `subset`, `select_subjects`, `drop_subjects`, `pick_channels`, `subject_ids` (natural order), `qualified_subjects`,
  `labels(label, threshold)`, `has_rating`, `normalized('subject'|'none')`, `flat_features`, `save`, `load`; `zscore_by_group`.
- `sieeg.data.synthetic`: `make_synthetic_featureset(...)`, `make_synthetic_trials(...)`. Every channel x band
  value is drawn from a stream keyed by the channel and band names, and the ratings from a stream of their own, so a
  reordered or reduced montage gives every named channel the same data and the same ratings. Raw trials are scaled
  by one gain set by `amplitude_volts` (the RMS of the 1/f background), `sfreq` and `duration`, so the rendered
  signals of a named channel are the same in every montage too.
- `sieeg.training.methods`: `AdversarySpec(kind in {group, identity, domain}, lam)`,
  `MethodSpec(name, description, adversaries)`, `get_method(name, lambda_subject=None, lambda_domain=None)`,
  `available_methods()`.
- `sieeg.utils`: `natural_key` (only decimal digits are numbers: "²" sorts as text), `natural_sorted`,
  `check_dataset_name(dataset)` (`ValueError` if it contains `:`), `check_channel_names(names, what)` (`ValueError`
  listing names equal under `str.lower()`, the folding every channel lookup uses), `qualify(dataset, subject)` = `"<dataset>:<subject>"`. Dataset names never contain `:` while subject ids may, so
  the first colon ends the dataset name and distinct (dataset, subject) pairs never collide.
- `tests/conftest.py`: fixtures `tiny_config`, `small_featureset`, `source_target_featuresets`, `raw_trials`, `rng`.

Rules for all modules: no TensorFlow import at module level outside `models/` and
`training/{trainer,optimizers}.py`. MNE is imported only inside the functions that
need it. Use the `logging` module (`logging.getLogger(__name__)`) rather than `print`
in library code. Use type hints, and docstrings that cite the paper section where relevant.

## 5. Data loaders

### 5.1 `sieeg.data.mahnob`

Layout: `<root>/Sessions/<sessionId>/session.xml` plus an optional `*.bdf`. `root` may
also be the `Sessions` folder itself. Rated emotion trials are sessions whose root
`<session>` element has the attributes `feltVlnc` and `feltArsl`. The subject id is
`<subject id="..">`. The BDF has 47 channels; channels 1-32 are EEG (names =
`BIOSEMI32_CHANNELS`) and `Status` holds the markers.

Stimulus timing: every file has 30 s before and after the stimulus, zero-padded
where nothing was recorded. A rising edge of bit value 16 in `Status` (mask the
channel with `0xFFFF`) marks stimulus start and stop. Read `sfreq` from the file;
the documentation disagrees (256 vs 512 Hz).

```python
def parse_session_xml(path) -> dict | None
    # rated emotion session -> {"session", "subject", "valence", "arousal", "media_file", "exp_type", "cut_nr", "xml"}
    # unrated / tagging session -> None. Values from attributes; ratings as float.
def find_rated_sessions(root, *, skipped=None, on_error="raise") -> list[dict]
    # adds "bdf" (Path; prefer *_emotion.bdf, else the single *.bdf). Skips folders without a BDF
    # (log one summary warning with the count). Sorted by (natural subject, natural session).
    # Rated sessions without a BDF are appended to skipped as {"subject", "trial": session, "reason"}.
    # A session whose metadata cannot be read (malformed XML, rated without <subject id>, non-numeric rating,
    # several candidate BDFs) raises with on_error="raise"; with "warn" it is logged and appended to skipped,
    # with subject UNKNOWN_SUBJECT ("unknown") if the XML gave none. A missing sessions folder always raises.
def stimulus_window(status, sfreq, n_times, *, pad_sec=30.0, tol_sec=2.0) -> tuple[int, int, str]
    # (start, stop, method). method "status": the first two 0->1 transitions of (status & 0xFFFF & 16),
    # accepted only if start/sfreq is within tol_sec of pad_sec AND (n_times - stop)/sfreq is within
    # 2*tol_sec of pad_sec AND stop > start. Otherwise method "fixed": [round(pad*sfreq), n_times - round(pad*sfreq)).
    # Raise ValueError if the fixed window is empty.
def load_session(session, *, channels=BIOSEMI32_CHANNELS, pad_sec=30.0, reader=None) -> Trial
    # reader(path) -> mne.io.BaseRaw; default mne.io.read_raw_bdf(path, preload=True, verbose="error").
    # Data = raw.get_data(picks=channels) [volts], cropped to [start, stop). Channel lookup is case-insensitive.
    # A file whose channel names collide ignoring case is a ValueError (the session is skipped under on_error="warn").
    # Trial(dataset="mahnob", subject=session["subject"], trial=session["session"], ...,
    #       meta={"bdf": str, "media_file", "crop": method, "onset_sec", "offset_sec", "sfreq_file"}).
def iter_mahnob_trials(root, *, channels=BIOSEMI32_CHANNELS, subjects=None, pad_sec=30.0,
                       reader=None, on_error="warn", skipped=None) -> Iterator[Trial]
    # on_error: "warn" (log and skip the session) or "raise"; it applies to reading the metadata too.
    # skipped (list): gets {"subject", "trial": session id, "reason"} for every dropped rated session of the
    # selected subjects (no BDF, unreadable metadata, load error) and for every unreadable session of an unknown
    # subject; complete once the iterator is exhausted.
```

Tests must not need real data. Build a fake `Sessions/` tree with `session.xml` files
(rated, unrated, tagging, and a folder without a BDF), and use an injected `reader` that
returns an `mne.io.RawArray` with 32 EEG channels, a few extra channels, and a
`Status` stim channel containing two 16-pulses.

### 5.2 `sieeg.data.manifest` (for DAI-EF or any other dataset)

A CSV with one row per trial. Required columns: `subject, trial, path, valence, arousal`.
Optional columns: `tmin, tmax` (seconds from file start, for cropping), `sfreq` and
`ch_names` (`;`-separated, for array files), `array_key` (variable name in .npz/.mat),
`transpose` (1/true/yes or 0/false/no/empty: the array is (n_times, n_ch)), and `scale` (multiplier to volts,
e.g. 1e-6 for µV). Relative paths are resolved against the manifest's directory.
Several rows may point to the same file with different `tmin`/`tmax`. Cache the most
recently opened file. Column names are exact (whitespace around them is removed). A known
column in another case or spelling, or an alias (onset/start -> tmin, offset/end/stop -> tmax,
fs/srate/sampling_rate -> sfreq, arraykey/key -> array_key, channels -> ch_names), is a
`ValueError` naming the expected column. Duplicate names, including names that differ only in
surrounding whitespace, are a `ValueError`; they are checked on the header row as written, since
pandas would read a repeated `tmin` as `tmin.1`. Any other column is ignored with one warning.
Every cell is read as written (`dtype=str, keep_default_na=False`) and normalised by one rule in
`_check_manifest`, the same for a CSV file and a DataFrame. Text cells (subject, trial, path, ch_names,
array_key) are stripped and otherwise literal: `NA` or `None` is a value, and only a blank cell is
missing. In the other known columns (ratings, tmin, tmax, sfreq, scale, transpose) a blank cell or one
of pandas' default missing-value tokens, in any letter case, is missing. A missing rating means
unrated; any other non-numeric rating is a `ValueError` naming the rows. A malformed tmin / tmax /
scale, or a malformed sfreq / transpose of an array file (MNE-readable files do not read these two),
fails only its row. Rows that read the same samples as an earlier row (same file, array variable and
crop) are reported in one warning.

```python
REQUIRED_COLUMNS = ("subject", "trial", "path", "valence", "arousal")
OPTIONAL_COLUMNS = ("tmin", "tmax", "sfreq", "ch_names", "array_key", "transpose", "scale")
COLUMN_ALIASES = {"onset": "tmin", "start": "tmin", "offset": "tmax", ...}   # rejected with the expected name
def read_manifest(path) -> pandas.DataFrame          # validates columns, resolves paths, keeps row order
def load_array_file(path, *, array_key=None) -> tuple[np.ndarray, dict]
    # .npy -> (array, {}); .npz -> (z[array_key or "data" or the only 2-D array], extras {"sfreq", "ch_names"} if stored);
    # .mat (scipy.io.loadmat) -> same rules. Raise ValueError if ambiguous.
def iter_manifest_trials(manifest, *, dataset="daief", sfreq=None, ch_names=None, scale=None,
                         reader=None, on_error="warn", skipped=None) -> Iterator[Trial]
    # MNE-readable files (.bdf .edf .gdf .fif .set .vhdr .cnt ...): reader(path) default
    # mne.io.read_raw(path, preload=True, verbose="error"); keep EEG channels only (drop stim/misc).
    # Arrays: sfreq / ch_names from the row, else from the file extras, else from the keyword defaults; error if unknown.
    # Precedence for scale: row > keyword > 1.0.
    # skipped (list): gets {"subject", "trial", "reason"} for every row skipped with on_error="warn".
    # ValueError before any file is read if dataset contains ":", ch_names collide ignoring case, sfreq is not a
    # positive number or scale is not a finite number other than 0 (a row's own scale cell: that row fails).
```

## 6. Features

### 6.1 `sieeg.features.psd`

```python
def welch_psd(data, sfreq, *, segment_sec=15.0, overlap=0.5, window="hann") -> tuple[np.ndarray, np.ndarray]
    # scipy.signal.welch on (n_ch, n_times): nperseg = min(round(segment_sec*sfreq), n_times),
    # noverlap = round(nperseg*overlap), detrend="constant", scaling="density", average="mean".
    # Returns (freqs, psd (n_ch, n_freqs)).
def band_log_power(freqs, psd, bands, *, floor=1e-30) -> np.ndarray
    # (n_ch, n_bands): log10(max(mean(psd[:, lo <= f < hi]), floor)). ValueError if a band has no bins.
def check_band_edges(bands, sfreq) -> None   # ValueError naming every band whose upper edge is above sfreq / 2 (Nyquist)
```

### 6.2 `sieeg.features.extract`

```python
def preprocess(trial, cfg: FeatureConfig) -> Trial
    # 1) pick cfg.channels (case-insensitive, in cfg order; KeyError lists the missing ones)
    # 2) reference == "average": subtract the mean over the picked channels
    # 3) optional mne.filter.filter_data(l_freq, h_freq)
    # 4) resample to cfg.resample_sfreq with scipy.signal.resample_poly (rational up/down) if it differs
def trial_band_power(trial, cfg) -> np.ndarray
    # ValueError if a band edge is above trial.sfreq / 2 (upsampling cannot restore what the recording lacks) ->
    # preprocess -> check_band_edges(cfg.bands, resampled sfreq) -> welch_psd -> band_log_power, (n_ch, n_bands).
    # So every band lies below min(trial.sfreq, resampled sfreq) / 2.
def extract_features(trials, cfg, *, dataset=None, on_error="warn", progress=True, loader_skipped=None) -> FeatureSet
    # Skip (warn) trials shorter than cfg.min_duration_sec and trials that raise (if on_error="warn").
    # meta = {"feature_config": config_to_dict-like dict, "skipped": [...], "n_input_trials", "sieeg_version"}.
    # All trials must come from one dataset (subject ids are unique only within one): a trial of another dataset
    # than the first is a ValueError whatever on_error. The name is that dataset's, or dataset if given (a rename).
    # ValueError up front if cfg.channels or cfg.bands is empty, cfg.channels repeats a name ignoring case,
    # cfg.bands repeats a band name, dataset contains ":" or a band edge is above cfg.resample_sfreq / 2. loader_skipped: the
    # skipped list given to the loader, read after trials is exhausted; its records come first in meta["skipped"] and
    # count in meta["n_input_trials"].
```

### 6.3 `sieeg.features.topomap`

```python
def montage_positions(ch_names, montage="standard_1005") -> np.ndarray     # (n, 3) metres, case-insensitive
def montage_center(montage="standard_1005") -> np.ndarray                  # least-squares sphere fit to ALL montage points
def azimuthal_equidistant(pos3d, center) -> np.ndarray
    # EEGLearn azim_proj on centred coordinates: colatitude theta = arccos(z/r), azimuth phi = atan2(y, x);
    # (X, Y) = theta * (cos phi, sin phi). X follows 3-D x (right), Y follows 3-D y (anterior).
class TopomapProjector:
    def __init__(self, ch_names, *, grid_size=32, method="cubic", montage="standard_1005", fill_value=0.0)
    # positions_2d (n, 2); grid columns x = linspace(min X, max X, G); rows y = linspace(max Y, min Y, G)
    # (row 0 = anterior). One scipy Delaunay triangulation; method "cubic" -> CloughTocher2DInterpolator,
    # "linear" -> LinearNDInterpolator, both on the precomputed triangulation.
    # ValueError if two channels share an electrode position (a repeated name, or aliases such as T3 / T7).
    mask: np.ndarray                               # (G, G) bool, True inside the electrodes' convex hull
    def transform(self, values) -> np.ndarray      # (..., n_ch) -> (..., G, G) float32; NaN -> fill_value.
                                                   # Vectorised: values reshaped to (n_ch, k), interpolated in chunks.
def build_images(fs, *, normalization="subject", grid_size=32, method="cubic", montage="standard_1005",
                 projector=None) -> np.ndarray     # (n, G, G, n_bands) float32, NHWC
def images_from_config(fs, cfg: ExperimentConfig) -> np.ndarray   # build_images with cfg.features + cfg.protocol.normalization
```

## 7. Models and training

### 7.1 `sieeg.models.grl`

```python
def gradient_reversal(x, lam) -> tf.Tensor
    # tf.custom_gradient: forward identity; backward (-lam * dy, zeros_like(lam)). lam is a scalar tensor.
class GradientReversal(keras.layers.Layer):     # holds lam in a non-trainable scalar weight; .lam property + setter
```

### 7.2 `sieeg.models.networks`

```python
def build_encoder(input_shape, cfg: ModelConfig, name="encoder") -> keras.Model
    # for block in cfg.conv_blocks: Conv2D(f, kernel_size, padding, activation=conv_activation) per filter count,
    # then MaxPooling2D(2). Flatten, Dropout(cfg.dropout) if > 0, then Dense(u, activation=dense_activation)
    # per cfg.dense_units. Default param count 574,528 for input (32, 32, 6). Every layer of the encoder and
    # the heads, and AdversarialNet itself, is dtype float32, and the image layers are
    # data_format="channels_last" (NHWC), whatever keras.config (image_data_format, floatx, dtype policy) says.
    # The optimiser is the exception: Keras stores its learning rate in floatx, so floatx "float16" trains
    # with 0.0020008 instead of 0.002 (float32, float64 and mixed_float16 are exact).
def build_head(input_dim, n_outputs, *, hidden, activation, dropout=0.0, name) -> keras.Model
    # Dense(hidden, activation) [+ Dropout] -> Dense(n_outputs) logits; hidden == 0 -> just Dense(n_outputs).
class AdversarialNet(keras.Model):
    def __init__(self, input_shape, n_classes, adversaries: dict[str, int], cfg: ModelConfig)
    # adversaries maps kind -> number of output logits ("group": 1, "domain": 1, "identity": n_subjects).
    # Attributes: encoder, classifier (FC classifier_hidden, classifier_activation), heads: dict[kind, keras.Model]
    # (FC discriminator_hidden, discriminator_activation). Every sub-model is built at construction.
    def call(self, x, training=False)            # emotion logits
    def embed(self, x, training=False)           # representation z
```

Weights: Keras defaults (glorot_uniform kernels, zero biases), the same as the original Keras code.

### 7.3 `sieeg.training.optimizers`

```python
class LegacyNadam(keras.optimizers.Optimizer)
    # Keras-2.x Nadam (Dozat 2016 with schedule_decay), per Keras 2.2.4 source; t = iterations + 1:
    # mu_t = b1*(1 - 0.5*0.96**(t*sd)); mu_t1 = b1*(1 - 0.5*0.96**((t+1)*sd)); prod_t = prod_{i<=t} mu_i (per variable)
    # m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g^2
    # g' = g/(1-prod_t); m' = m/(1-prod_t*mu_t1); v' = v/(1-b2**t); m_bar = (1-mu_t)*g' + mu_t1*m'
    # p -= lr * m_bar / (sqrt(v') + eps)
def make_optimizer(cfg: TrainConfig) -> keras.optimizers.Optimizer   # "nadam_legacy" | "nadam" | "adam"
```

### 7.4 `sieeg.training.schedules`

```python
def lambda_factor(progress, schedule="constant") -> float   # "constant" -> 1; "dann" -> 2/(1+exp(-10 p)) - 1
```

### 7.5 `sieeg.training.trainer`

```python
@dataclass class LabeledData:   x (n,H,W,C) float32; y (n,) int; subjects (n,) int (global index, -1 unknown)
@dataclass class UnlabeledData: x (n,H,W,C) float32; subjects (n,) int
@dataclass class FitResult:     history: pd.DataFrame; best_epoch: int (1-based; last epoch if no val); best_score: float | None; seconds: float;
                                identity_alpha: float | None = None (weight of identity-pool rows, D15; None without an identity pool)
class AdversarialTrainer:
    def __init__(self, method: MethodSpec, input_shape, *, n_classes=2, n_subjects=0,
                 model_cfg=ModelConfig(), train_cfg=TrainConfig(), seed=0)
        # keras.utils.set_random_seed(seed); build AdversarialNet with heads for method.adversaries
        # ("identity" needs n_subjects >= 2); make_optimizer; compile the train step with
        # tf.function(reduce_retracing=True, jit_compile=train_cfg.jit_compile) and a None batch dimension.
    def fit(self, train, val=None, *, outgroup=None, target=None) -> FitResult
    def predict_proba(self, x, batch_size=512) -> np.ndarray     # softmax probabilities (n, n_classes)
    def predict(self, x, batch_size=512) -> np.ndarray
    def embed(self, x, batch_size=512) -> np.ndarray
```

**Training step** (Eq. 1 / Eq. 3 by GRL). Per step, with numpy RNG `default_rng(seed)`:

```
S = next shuffled batch of the labeled training set (size b; the last batch may be smaller)
O = |S| samples drawn with replacement from `outgroup`   (if a "group" adversary exists, or "identity" uses it)
T = |S| samples drawn with replacement from `target`     (if a "domain" adversary exists, or a "group"/"identity" adversary and a target is given)
z = encoder(concat[S, O?, T?]); split into z_S, z_O, z_T   (a single forward pass; there is no batch-norm)
L = CE(classifier(z_S), y_S)                                              # emotion: source only, optional class weights
group:    L += BCE(head(GRL(z, lam_g)), [0]*|S| + [1]*|O| + g_T)   # every batch row (S, O, T?); g_T[j] = 1 iff subject(T_j) is an out-group subject
domain:   L += BCE(head(GRL(concat[z_S, z_T], lam_d)), [0]*|S| + [1]*|T|)   # source 0, target 1
identity: L += sum(w * CE(head(GRL(concat[z_S, z_P], lam_s)), concat[subj_S, subj_P])) / sum(w),
          w = [1]*|S| + [alpha]*|P|, alpha = n_subjects(pool) / n_subjects(train); P = T if target is given else O (else S only: plain mean)
one Nadam step on all trainable variables of encoder + classifier + heads
lam_k(step) = spec.lam * lambda_factor(step / total_steps, schedule)    # passed in as a tensor (no retracing)
```

`group_labels` (float32, one per batch row, computed in NumPy) and `identity_alpha` (float32
scalar) are inputs of the compiled step, so neither retraces it. The group head reads z
itself because it sees every block, and heads that read the same rows (SIDANN's domain and
identity heads) share one concat tensor; a third gradient path into z_S would make the float32
gradient sums depend on the trace (not bit-reproducible). With a target pool the group
adversary needs subject indices >= 0 for every out-group and target sample (`ValueError`).

Validation: if a needed pool is missing, raise `ValueError` naming it. After each
epoch, evaluate `val` (accuracy, weighted F1, CE loss). Snapshot `net.get_weights()`
whenever `selection_metric` strictly improves (`loss`: lower is better). With
`patience`, stop after that many epochs without improvement. Restore the best weights
at the end. History has one row per epoch: `epoch, lambda_<kind>, loss_cls, acc_cls,
loss_<kind>, acc_<kind>, val_loss, val_acc, val_f1, seconds`. `class_weight="balanced"`
weights the emotion CE by n/(K * count_k) computed on the training labels.

## 8. Evaluation

### 8.1 `sieeg.evaluation.metrics`

```python
def classification_rate(y_true, y_pred) -> float          # accuracy in %
def weighted_f1(y_true, y_pred) -> float                  # sklearn f1_score(average="weighted", zero_division=0) in %
def per_subject_metrics(pred: pd.DataFrame) -> pd.DataFrame   # columns subject, n, cr, f1 (natural subject order)
def summarize_predictions(pred) -> dict
    # {"n_trials", "n_subjects", "pooled": {"cr", "f1"}, "subject_mean": {"cr", "f1", "cr_std", "f1_std"}}
```

### 8.2 `sieeg.evaluation.stats`

```python
def paired_comparison(a: pd.Series, b: pd.Series) -> dict
    # a, b: per-subject scores (percent) indexed by subject; aligned on common subjects (n >= 2 required).
    # Both tests use the differences a - b rounded to 1e-9 percentage points, so float noise neither breaks
    # ties between equal differences nor makes a zero difference nonzero.
    # {"n", "mean_a", "mean_b", "mean_diff", "t", "p_ttest" (scipy ttest_1samp of the differences, as
    # ttest_rel), "p_wilcoxon" (scipy wilcoxon, default method, which depends on the SciPy version)}.
    # p_wilcoxon is NaN if all differences are zero; t and p_ttest are NaN if all differences are equal
    # (no spread: scipy would report t = inf, p = 0).
```

### 8.3 `sieeg.evaluation.report`

```python
@dataclass class RunResult: summary: dict; predictions: pd.DataFrame; folds: pd.DataFrame; histories: dict[str, pd.DataFrame];
                            run_dir: Path | None = None    # set by load_run; named in the tables' warnings
def save_run(result, out_dir, config=None) -> Path
    # summary.json (last), predictions.csv, folds.csv, history/<fold>.csv, config.yaml (if config)
def load_run(run_dir) -> RunResult                         # histories = {}
def find_runs(paths) -> list[Path]                         # every directory below paths that contains summary.json
def results_table(runs, *, source="pooled") -> pd.DataFrame
    # one row per (protocol, test_dataset, train_dataset, method, tag, lambdas as sorted JSON); columns: that key,
    # valence_cr, valence_f1, arousal_cr, arousal_f1, n_runs (most runs averaged into one score of the row),
    # n_subjects (distinct test subjects of the row's runs). Only runs of the same row and label are averaged.
def significance_table(runs, *, reference, metric="f1") -> pd.DataFrame
    # paired_comparison of per-subject `metric` between each (reference, lambdas) entry and every entry of another
    # method, per (protocol, test_dataset, train_dataset, tag, label); runs of one (method, lambdas) are averaged
    # per subject. Columns: that key, reference, reference_lambdas, method, method_lambdas, COMPARISON_FIELDS.
def probe_table(runs) -> pd.DataFrame
    # runs with summary["probe"]: one row per results-table row and label; n_runs, probe_subject_acc,
    # probe_subject_chance, probe_group_bacc (means over the runs, fractions)
def to_markdown(df, floatfmt=".1f") -> str                 # no tabulate dependency
```

`results_table` logs a warning that names the run directories when two runs of one row and
label have the same seed (e.g. fold shards saved under two `--out` roots), or when the runs of
a row test different subjects (e.g. a `--folds` check next to a full run, or a subject rated
for one label only). Paired tests are grouped by tag: a tagged variant is compared only with
runs of the same tag.

`summary.json` keys: `protocol` ("within" | "cross"), `method`, `tag`, `label`,
`train_dataset`, `test_dataset`, `lambdas` ({kind: lam}), `seed`, `n_folds`,
`n_subjects`, `n_trials`, `pooled`, `subject_mean`, `probe` ({subject_acc, subject_chance,
group_bacc}: means over folds of the subject-probe scores, fractions; null unless
`protocol.subject_probe`), `created` (ISO time), `sieeg_version`.
`predictions.csv` columns: `fold, dataset, subject, trial, rating, y_true, y_pred, p1`.
`folds.csv` columns: `fold, test_subjects, val_subjects, n_train, n_val, n_test, n_outgroup, n_target, best_epoch, best_val, cr, f1, seconds, probe_subject_acc, probe_subject_chance, probe_group_bacc`
(the probe columns are fractions, empty unless `protocol.subject_probe`).

### 8.4 `sieeg.evaluation.protocols`

```python
@dataclass class Split: test: list[str]; val: list[str]; train: list[str]
def loso_splits(subjects, n_val, seed) -> list[Split]
    # subjects in natural order; for test subject i: val = natural_sorted(default_rng([seed, i]).choice(others, n_val, replace=False))
def run_within_database(fs, method, label, cfg, *, folds=None, tag=None, progress=True) -> RunResult
def run_cross_database(source, target, method, label, cfg, *, folds=None, tag=None, progress=True) -> RunResult
```

Within database (any method without a "domain" adversary; otherwise `ValueError`):
1. Keep the rows with a finite `label` rating. `images_from_config` is computed once, and `y = fs.labels(label, cfg.protocol.label_threshold)`.
2. Subject index = position in `fs.subject_ids`. For each test subject (all of them, or `folds`): split via `loso_splits(..., cfg.protocol.n_val_subjects, cfg.seed)`.
   train = LabeledData(train subjects); val = LabeledData(val subjects);
   outgroup = UnlabeledData(test rows, plus val rows if `outgroup_includes_val`); target = None.
3. `keras.backend.clear_session()`; `AdversarialTrainer(method, (G, G, n_bands), n_classes=2, n_subjects=len(subject_ids), ..., seed=cfg.seed*1000 + fold_index)`; fit; predict the test rows.

Across databases (source labeled, target unlabeled):
1. Bands must match. Channels must match case-insensitively; otherwise use the intersection in source order and log a warning. Both sets keep only rows with a finite rating. Images are built per dataset.
   Source and target must have different dataset names (`ValueError`). A rated target band-power row (channels aligned to the source order) identical to a rated source row raises `ValueError` ("N of M target trials are identical to source trials: the target would be tested on labeled training data"). Extraction settings that differ between the two feature files (`features.featureset.feature_config_differences`) are logged in one warning.
2. val = `natural_sorted(default_rng(seed).choice(source subjects, n_val, replace=False))`; train = the remaining source subjects. target pool = all target rows, for every method with a group, domain or identity adversary. Subject index over `qualified_subjects` of source and target.
3. Per-subject mode (`cross_per_subject`: "auto" means per-subject iff the method has a "group" adversary, or "always" / "never"): for each target subject t (all, or `folds`), outgroup = rows of t (the group head labels t's target rows 1 and the other target rows 0, D14); fit; predict t. Single mode: outgroup = all target rows; fit once; predict every target subject (a group adversary then duplicates a domain adversary; logged as a warning). Fold name = the target subject, or "all".
4. Training seed: `cfg.seed*1000 + index` of the target subject (0 in single mode).

Each fold runs in try/finally: the fold's `CustomGradient-*` entries in TensorFlow's private
gradient registry (GRL, Keras optimiser) are removed at the end, because they pin the traced
train step (about 18 MiB per paper-size fold) for the rest of the process. This silently does
nothing if the private registry is unavailable. The trainer reference is dropped.

With `protocol.subject_probe` (not in the paper), each fold, after predicting, embeds (a) the
validation rows and runs `analysis.probe.subject_probe` on their subject indices (NaN with
fewer than 2 validation subjects) and (b) the in-group rows (training rows plus the
target-pool rows that are not tested) and the test rows, and runs `group_probe`; seed = the
fold seed; no emotion labels. Fold columns `probe_*`; `summary["probe"]` = means over folds
with finite values (None when off).

`cfg.protocol.jobs > 1` runs folds in a `ProcessPoolExecutor` (spawn context) of
`min(jobs, n_folds)` workers. Each worker enables memory growth on every GPU (by default TensorFlow
reserves nearly all GPU memory for one process and the other workers fail with `cudaSetDevice() ...
out of memory`), sets TF intra-op threads to `max(1, cpu_count // workers)` and receives the shared
arrays once, through the pool initializer. A run of one fold trains in the calling process, which
with `jobs > 1` also lets its GPU memory grow. A process that used the GPU before that (its own
TensorFlow code, or an earlier `jobs=1` run) keeps nearly all GPU memory unless
`TF_FORCE_GPU_ALLOW_GROWTH=true` is set, and leaves the workers the remainder. Splits, seeds, data
and fold order equal `jobs=1`. On a CPU the scores equal it up to float32 rounding, because
TensorFlow's CPU kernels sum in an order that depends on the intra-op thread count (about 1e-7 in
`p1` after a few epochs; such differences can grow over long training). On a GPU the default
kernels are not deterministic, so any two runs differ (about 1e-4 in `p1` after two epochs of a
tiny model); with `TF_DETERMINISTIC_OPS=1` in the environment, which the workers inherit
(TensorFlow settings made in the parent process are not inherited), runs are bit-identical and
`jobs=N` equals `jobs=1` exactly. TensorFlow has deprecated the variable in favour of
`tf.config.experimental.enable_op_determinism()`, which suffices for `jobs=1` if it is called before
the process trains anything (called later, it does not make the following runs bit-identical); the
variable is used because it is the only setting the workers inherit. This was checked with TensorFlow
2.18 only; on a GPU the bit-for-bit tests fail under a version that ignores the variable.
`summary` and `predictions` are built with `metrics.summarize_predictions`.

## 9. Experiment 1: `sieeg.analysis.bias`

```python
def subject_identification(fs, *, normalization="none", n_splits=10, n_estimators=500, seed=0, n_jobs=-1) -> dict
    # RandomForest on fs.flat_features("none"); StratifiedKFold(n_splits, shuffle, seed) over trials.
    # normalization "subject": in every fold each subject's mean/sd are fitted on its trials outside the test fold
    # (all its trials if none) and applied to all of them.
    # {"task": "subject", "normalization", "cv": "trial", "accuracy", "accuracy_std", "chance_uniform": 1/K,
    #  "chance_majority", "n_classes", "n_samples", "n_splits"}
def domain_identification(fs_a, fs_b, *, normalization="none", cv="trial", balance=True, n_splits=10,
                          n_estimators=500, seed=0, n_jobs=-1) -> dict
    # common channels; undersample the larger set (seeded) if balance; label = dataset; subjects qualified by dataset.
    # cv "trial": StratifiedKFold over trials (paper; also measures subject identity).
    # cv "subject": StratifiedGroupKFold over qualified subjects (seeded random group codes, shuffle=False),
    # at most min(#subjects per database) folds (warning when fewer than n_splits; ValueError below 2 subjects).
    # normalization as above, fitted per fold. normalization "none": WARNING if the median log10 band power differs by
    # more than 2 decades (UNITS_GAP_DECADES) or feature_config_differences(fs_a, fs_b) is non-empty.
    # {"task": "domain", "normalization", "cv", "accuracy", "accuracy_std", "balanced_accuracy", "chance_uniform": 0.5,
    #  "chance_majority", "n_classes", "n_samples", "n_splits" (folds used), "n_per_domain"}
def run_bias_analysis(fs, other=None, *, normalizations=("none", "subject"), domain_cv=("trial", "subject"),
                      balance=True, **kwargs) -> pd.DataFrame
    # rows: subject x normalizations, then domain x domain_cv x normalizations (trial rows first); fractions
```

Trial-level database identification is comparable with the paper but includes subject
identification; subject-grouped database identification has a 50 % null level. With
subject-grouped folds the class ratio varies a little between folds, so balanced accuracy is
the more robust figure. The library returns fractions; `sieeg bias` prints and writes percent.

## 10. CLI (`sieeg`, `python -m sieeg`)

```
sieeg extract mahnob ROOT --out FILE [--overwrite] [--subjects 1,2] [--pad-sec 30] [--config C] [--set k=v ...]
sieeg extract manifest CSV --dataset daief --out FILE [--overwrite] [--sfreq F] [--ch-names a,b,..] [--scale 1e-6]
             [--config C] [--set ...]
sieeg extract synthetic --out FILE [--overwrite] [--n-subjects 10] [--n-trials 20] [--dataset synthetic]
             [--domain-shift 0] [--seed 0] [--raw]
sieeg info FILE
sieeg bias FILE [--other FILE2] [--exclude-subjects ..] [--n-splits 10] [--n-estimators 500] [--seed 0]
             [--no-balance] [--jobs -1] [--out CSV]
sieeg within FILE --method M[,M2..] --label valence|arousal|both [--folds S1,S2] [--exclude-subjects ..]
             [--out runs] [--tag T] [--overwrite] [--config C] [--set k=v ..] [--epochs N] [--seed S] [--jobs J]
             [--lambda-subject X] [--probe]
sieeg cross SOURCE TARGET --method .. --label .. [--folds ..] [--exclude-source ..] [--exclude-target ..]
             [same options] [--lambda-subject X] [--lambda-domain X]
sieeg report DIR [DIR ..] [--reference sida_eeg] [--metric f1] [--source pooled|subject_mean] [--out FILE.md]
```

`-v` (debug logging, tracebacks) and `-q` (warnings only) go before or after the command.
Missing files, invalid configuration values (Sec. 4) and refused runs print one line,
`sieeg <command>: error: ...`, and exit with status 1 (OSError, KeyError, TypeError,
ValueError; `-v` re-raises them); invalid arguments exit with status 2.

`extract` refuses to replace an existing output file (with `.npz` appended when missing)
unless `--overwrite`, before reading any data. It passes a `skipped` list to the loader and
on to `extract_features(loader_skipped=...)`, so the feature file lists the trials the loader
dropped. `bias` prints and writes (`--out`) its scores in percent (`accuracy_pct`,
`accuracy_std_pct`, `balanced_accuracy_pct`, `chance_uniform_pct`, `chance_majority_pct`,
with `task`, `normalization` and `cv` first); the database rows name their folds
(trial-level or subject-grouped), followed by one line that explains the difference.

`within` / `cross` run directory: `<out>/<protocol>-<test_dataset>/<method>[-<tag>]/<label>/seed<seed>/`.
Before anything trains, the command refuses (one error, nothing created):

- methods with a domain adversary within a database ("needs a second, different target database (domain adversary): sieeg cross SOURCE TARGET");
- for every requested label: `--folds` subjects that are excluded, not in the (target) file, or not rated for the label, and
  `protocol.n_val_subjects` that leaves no training subject among the rated (source) subjects;
- run directories that already have a `summary.json`, unless `--overwrite`.

It then claims every run directory by creating `.running` with `O_CREAT | O_EXCL` (JSON: pid, host,
argv, start time). A directory claimed by another process is refused unless `--overwrite`,
which takes the claim over. After training, a run checks `summary.json` again (another
process may have saved one; refused unless `--overwrite`), removes the old `summary.json` and
`history/`, writes `command.json`, `features.json` and `save_run`'s files (`summary.json`
last), and releases its claim. A failed command releases its remaining claims and removes the
run directories it left empty. `features.json` records each input feature file (`file`, or
`source` and `target`): resolved `path`, `sha256`, `dataset`, `excluded_subjects`, `subjects`
(after the exclusions), `n_trials`, `ch_names`, `bands` and the file's `meta` (with its
`feature_config`); `config.yaml`'s `features` section is the command's configuration, not the
extraction's. `--probe` sets `protocol.subject_probe`; a run's result line then ends with the
probe means. `report` prints the results table, the paired tests and, if a run has probe
scores, `probe_table`.
`os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` is set before TensorFlow is imported; TF is imported lazily inside the commands.

## 11. Testing

`pytest` (configured in `pyproject.toml`; `-m "not slow"` skips end-to-end training).
Every module has unit tests that use synthetic data only. Tests that train use
`tiny_config`-sized models and at most a few epochs. `tests/conftest.py` sets
`TF_FORCE_GPU_ALLOW_GROWTH=true`, `TF_DETERMINISTIC_OPS=1` and `NVIDIA_TF32_OVERRIDE=0` (unless already
set) before TensorFlow starts, so on a GPU the suite and its worker processes run deterministic float32
kernels; the bit-for-bit and tight-tolerance tests need both. Required checks include:
GRL gradient sign and scale; encoder parameter count 574,528; LegacyNadam against a
NumPy re-implementation of Keras-2 Nadam; Welch and band power on known sinusoids;
the topomap reproducing values at electrode positions with the anterior side up;
MAHNOB Status cropping with a fallback; manifest arrays vs MNE files; no test or target
labels reaching the trainer; identical splits across methods; jobs=2 equal to jobs=1; workers
letting GPU memory grow; an encoder that stays channels-last under a `channels_first` Keras default;
channel names colliding ignoring case rejected; manifest missing-value tokens in any letter case,
the same for a CSV file and a DataFrame; CLI end to end on synthetic data. Also: exact per-head lambda wiring (one SGD(lr=1) step)
and dropout on in the train step / off at inference; group labels across databases are not
a function of the domain labels; within-database SINN weights the test subject like one
training subject; prediction rows stay aligned on interleaved feature files with
non-natural ids; no gradient-registry growth across folds; same database or shared trials
rejected across databases; Experiment 1 on null data (subject-normalised rows at chance, not
below; subject-grouped database identification near 50 % and high with a database shift);
configuration types, ranges, choices and YAML errors; report rows that never average runs of
different lambdas or source databases, with warnings for repeated seeds and different test
subjects; run-directory claims, the summary re-check and `features.json`; `--folds` checked
for every label before training; `extract` not overwriting without `--overwrite`.
