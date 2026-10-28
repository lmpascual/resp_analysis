# resp_analysis

Neuropixels 2.0 recordings of the ventral respiratory column (preBötC / rVRG) in urethane-anaesthetised mice, from
raw SpikeGLX files to breathing measures, single-unit phase tuning and PETHs, and population dynamics (Poisson HMM,
PCA). The worked example is **e097** (Vglut2:ai32, 2026-09-04): two 15 s anoxia challenges, each followed by gasping
and recovery, plus 8 optogenetic laser trains earlier in the session.

```
resp_analysis/
  README.md          this guide: setup, data, and the commands in order
  configs/           local_paths.example.yaml -> copy to local_paths.yaml (where the data are on your computer)
  environment/       npxsort.yml + npxsort-requirements.txt (sorting), cibrrig.yml (analysis)
  sorting/           step 2: spike sorting (npxsort: Kilosort 4 + Lupin, bombcell labels), review browser, docs
  analysis/
    session/         whole-session breathing, heart-rate and laser summary
    units/           unit checks: recovery after anoxia, matching units across sorts, epoch outliers, optotagging
    population/      breaths and phase, single-unit phase curves / PETHs / rasters, HMM and PCA, transitions, anoxia #2
    slides/          slide decks built from those figures
  archive/           the earlier pipeline and analyses, no longer used (see archive/README.md)
```

What was decided and found along the way, and why (thresholds, curation rules, which units count): the analysis
record [sorting/docs/e097_analysis_record.md](sorting/docs/e097_analysis_record.md). What every sorting stage and unit
label means: [sorting/docs/unit_labels_explained.md](sorting/docs/unit_labels_explained.md).

## 1. What you need

- **Computer:** Windows 10/11 (tested; the scripts are plain Python and should also run on Linux). Re-sorting needs
  an NVIDIA GPU for Kilosort 4. Disk: 153 GB for the raw AP file, ~30 GB for the CatGT extracts,
  ~11 GB of scratch per sorted window (+ 14 GB for optotagging).
- **Software:** [Miniforge](https://github.com/conda-forge/miniforge) (conda), git,
  [CatGT](https://billkarsh.github.io/SpikeGLX/) (only to make the sorting input), Node.js (only for the unit slide
  deck).
- **Data, shared separately (not in git):**
  1. **The e097 animal folder** `e097_g2/`: `e097_g2_t0.nidq.bin/.meta` (diaphragm EMG, airflow, ECG, laser; 1.2 GB)
     and `e097_g2_imec0/e097_g2_t0.imec0.ap.bin/.meta` (153 GB).
  2. **`e097_sorted_results.zip`** (177 MB): the Kilosort 4 / Lupin sorts of the two anoxia windows with the manual
     review verdicts, made by `sorting/pack_results.py`. The analyses use the units reviewed `single`, by unit id. A
     new sort gives different ids (Lupin's detection is unseeded; Kilosort 4 on a GPU isn't guaranteed to repeat), so
     use this zip to reproduce the analyses on the same units. It also holds the experiment log (`e097.docx`) and the
     laser-train tables.

## 2. Setup (once)

```bat
git clone https://github.com/lmpascual/resp_analysis.git
cd resp_analysis
copy configs\local_paths.example.yaml configs\local_paths.yaml
notepad configs\local_paths.yaml
```

Set the four folders in `local_paths.yaml`: `raw_data` (the folder that contains `e097_g2/`), `catgt`, `scratch` and
`outputs`. Every config and script finds its files from these (`${outputs}/e097/...` in the configs).

**Sorting environment** (`npxsort`: SpikeInterface 0.105.2, Kilosort 4.1.7). Torch with CUDA goes first; plain
`pip install torch` gives a CPU-only build on Windows:

```bat
conda env create -f environment\npxsort.yml
conda activate npxsort
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r environment\npxsort-requirements.txt
python sorting\check_env.py
```

`check_env.py` must end with `Environment OK.` (`python sorting\check_env.py --no-gpu` if you won't re-sort). Pick the
torch build for your CUDA version on pytorch.org; `nvidia-smi` shows the newest your driver supports.

**Analysis environment** (`cibrrig`: Nick Bush's [cibrrig](https://github.com/nbush257/cibrrig), the code of Bush &
Ramirez 2024 and Bush et al. 2025, installed from GitHub at the commit used here, plus hmmlearn):

```bat
conda env create -f environment\cibrrig.yml
```

**Node** (only for `analysis/slides/make_unit_slides.js`): `npm install pptxgenjs@3.12.0` in any folder, then
`set NODE_PATH=<that folder>\node_modules`.

## 3. Data in place

1. Put the animal folder under `raw_data`, so that `<raw_data>\e097_g2\e097_g2_t0.nidq.bin` exists.
2. Unzip `e097_sorted_results.zip` into `outputs` (it creates `<outputs>\e097\intact_window_v2\...`,
   `anoxia2_window_v2\...` and `e097\e097\e097.docx`).
3. Make the CatGT extracts (300–9000 Hz 12th-order Butterworth, local CAR 75–300 µm, probe 0). Make each `-dest`
   folder first. They are needed to re-sort, for the review browser and Phy, and for `match_sorts`, `epoch_outliers`
   and `optotag`; the breathing, PETH and population analyses don't need them.

   ```bat
   CatGT -dir=<raw_data> -run=e097 -g=2 -t=0 -prb_fld -ap -prb=0 -startsecs=1740 -maxsecs=420 -apfilter=butter,12,300,9000 -loccar_um=75,300 -dest=<catgt>/catgt_e097_intact_420s
   CatGT -dir=<raw_data> -run=e097 -g=2 -t=0 -prb_fld -ap -prb=0 -startsecs=2250 -maxsecs=360 -apfilter=butter,12,300,9000 -loccar_um=75,300 -dest=<catgt>/catgt_e097_anoxia2_360s
   CatGT -dir=<raw_data> -run=e097 -g=2 -t=0 -prb_fld -ap -prb=0 -startsecs=840 -maxsecs=600 -apfilter=butter,12,300,9000 -loccar_um=75,300 -dest=<catgt>/catgt_e097_laser_600s
   ```

**Time base.** Sorted spike times are seconds from the start of their CatGT extract. Add 1740 s (anoxia #1 window) or
2250 s (anoxia #2 window) for raw-run time, which is what the experiment log uses (anoxia #1 at 1800–1815 s, anoxia #2
at 2310–2325 s).

## 4. Run the chain

Run from the repository root. `<outputs>` stands for your `outputs` folder. Each script's docstring (`python <script>
--help`) has the details and every option.

| Step | Env | Command | Makes | Time |
|---|---|---|---|---|
| 1. Session summary | cibrrig | `python analysis\session\e097_session_summary.py` | `<outputs>\e097\e097_whole_session_data.json`: breathing and heart rate over the whole session, milestones, laser trains | 30 s |
| 2. Preprocess (for the review browser, Phy, steps 4b–4c) | npxsort | `cd sorting` then `python run_sorting.py configs\e097_intact_window.yaml --stop-after preprocess` (and `configs\e097_anoxia2_window.yaml`) | the preprocessed binary in `<scratch>` (it also rewrites `config_used.yaml` with your folders; the zip keeps the original) | 1 min each |
| 3. Review the units (optional) | npxsort | `python sorting\review_units.py <outputs>\e097\intact_window_v2 --time-offset-s 1740` | browser; verdicts in `review\kilosort4_unit_review.tsv` | — |
| 4a. Recovery after anoxia | npxsort | see below | `<outputs>\e097\anoxia_recovery\...` | — |
| 4b. Match units between the two anoxia sorts | npxsort | see below | `anoxia_recovery\matched_units\matched_pairs.tsv` | — |
| 4c. Epoch outlier screen (QC) | npxsort | see below | `<outputs>\e097\epoch_outliers\...` | — |
| 4d. Optotagging | npxsort | `python analysis\units\optotag.py analysis\units\configs\e097_optotag_train8_uncorrected.yaml` then `... e097_optotag.yaml` | `<outputs>\e097\optotag_laser_window\` (read `SUMMARY.md`) | long (drift-corrects 600 s) |
| 5. Breathing, phase tuning, PETHs, HMM, PCA | cibrrig | see below (three commands) | `<outputs>\e097\population_dynamics_v2\` | 4 min each |
| 6. Transitions eupnea → gasping → recovery | cibrrig | `python analysis\population\transition_windows.py` then `python analysis\population\transition_slides.py` | `population_dynamics_v2\transitions\` | 1 min |
| 7. Anoxia #2 in the anoxia #1 state space | cibrrig | `python analysis\population\anoxia2_dynamics.py`, then `transition_windows.py --run <outputs>\e097\population_dynamics_anoxia2` and `transition_slides.py --run <outputs>\e097\population_dynamics_anoxia2 --no-manifest` | `<outputs>\e097\population_dynamics_anoxia2\` | — |
| 8. Anoxia #1 vs #2 side by side | cibrrig | `python analysis\population\trial_compare.py`; `python analysis\population\states_pca_3d.py` (HMM states in 3D PC1–PC3, every epoch of both trials togglable); `python analysis\population\pca_softnorm.py` (a soft-normalised PCA beside the Bush & Ramirez one; analysis record D13, F18); `python analysis\population\transition_movie.py [--run <run>]` (time-lapse GIF of the epochs fading one into the next in 3D PC1–PC3, one fixed angle); `python analysis\population\trajectory_movie.py [--run <run>] [--pcs 1,3,2]` (MP4 in the style of the Bush & Ramirez 2024 supplementary video: diaphragm, raster and every 5 ms bin in 3D from eupnea through gasping to recovery; needs ffmpeg for H.264) | `figures\compare\`, `epoch_metrics.tsv` in both runs; `population_dynamics_v2\figures\hmm_states_pca_3d.html`; `hmm_states_pca_softnorm_K6_by_epoch*.png` in both runs, `population_dynamics_v2\figures\pca_pc2_single_unit_steps.png`; `transitions\movie\` in each run | — |
| 9. Slide decks | node / cibrrig | see below | `.pptx` | — |

Step 4 (units):

```bat
conda activate npxsort
python analysis\units\anoxia_recovery.py <outputs>\e097\intact_window_v2 --switch-s 60 --gasp-start-s 85 --gasp-end-s 130 --time-offset-s 1740 --marks "active exp.:338" --breathing-json <outputs>\e097\e097_whole_session_data.json --out <outputs>\e097\anoxia_recovery\intact_window_kilosort4
python analysis\units\anoxia_recovery.py <outputs>\e097\anoxia2_window_v2 --switch-s 60 --gasp-start-s 87 --gasp-end-s 104 --time-offset-s 2250 --marks "slow breathing:175,breathing back:295" --breathing-json <outputs>\e097\e097_whole_session_data.json --out <outputs>\e097\anoxia_recovery\anoxia2_window_kilosort4
python analysis\units\match_sorts.py <outputs>\e097\intact_window_v2 <outputs>\e097\anoxia2_window_v2 --offset-a-s 1740 --offset-b-s 2250 --switch-a-s 60 --switch-b-s 60 --recovery-a <outputs>\e097\anoxia_recovery\intact_window_kilosort4 --recovery-b <outputs>\e097\anoxia_recovery\anoxia2_window_kilosort4 --out <outputs>\e097\anoxia_recovery\matched_units
python analysis\units\epoch_outliers.py <outputs>\e097\intact_window_v2 --time-offset-s 1740 --out <outputs>\e097\epoch_outliers\intact_window_v2_kilosort4
```

Step 5 (behaviour, single units, population). The recovery after anoxia is split into epochs at change points in
breathing and population state, so the first run fits the models, `epoch_change_points.py` finds the boundaries, and
the second run computes everything per epoch (the HMM fits don't depend on the epochs and come out the same):

```bat
conda activate cibrrig
python analysis\population\population_dynamics.py --out <outputs>\e097\population_dynamics_v2
python analysis\population\epoch_change_points.py <outputs>\e097\population_dynamics_v2 --ref-k 6 --n-segments 5 --out <outputs>\e097\population_dynamics_v2\epochs_anoxia1.json
python analysis\population\population_dynamics.py --conditions-file <outputs>\e097\population_dynamics_v2\epochs_anoxia1.json --out <outputs>\e097\population_dynamics_v2
```

It reads the 28 units reviewed `single` from `intact_window_v2`, and the diaphragm EMG and airflow from the raw NIDQ
file. Breaths, inspiratory onsets and offsets and the phase φ come from cibrrig (`filt_int_ds_dia`,
`compute_dia_phase`); gasps follow the Bush & Ramirez 2024 rule. Outputs:
- `breaths.tsv`: every breath with its timing, amplitude, epoch and gasp flag (the behavioural measures);
- `unit_phase_tuning.tsv` and `unit_peth_features.tsv`: preferred phase, modulation and class, and PETH features, per
  unit and epoch;
- `hmm_*.tsv`, `hmm_fits.npz`: Poisson HMMs for K = 3–8 states;
- `figures/`: PCA trajectories, HMM states, PETHs around inspiration onset and offset, phase curves, and
  breath-by-breath rasters per unit (`figures\units\`, `figures\units_slides\`, `figures\breath_raster_viewer.html`).

`--figures-only` redraws the figures from `cache\` without refitting.

Step 9 (decks):

```bat
node analysis\slides\make_unit_slides.js <outputs>\e097\population_dynamics_v2\figures\units_slides\slides.json <outputs>\e097\population_dynamics_v2\e097_single_units_slides.pptx
python analysis\slides\amend_vrc_deck.py <in.pptx> <out.pptx>
```

`amend_vrc_deck.py` (cibrrig env) updates an existing deck, "VRC network changes after anoxia-induced gasping", which
isn't in the repository.

### Re-sorting from the raw data instead

```bat
conda activate npxsort
cd sorting
python run_sorting.py configs\e097_intact_window.yaml
python run_sorting.py configs\e097_anoxia2_window.yaml
```

About 2 h per window on a GPU (the intact window took 116 min). Then review the units in the browser (step 3); the
review files of the shared sort won't apply, because the unit ids change. Settings, curation rules and why they were
chosen: [sorting/README.md](sorting/README.md) and the analysis record.

## 5. The Bush & Ramirez 2024 gasping recordings

Nick Bush's recordings for Bush & Ramirez 2024 (Nat Neurosci) include 39 gates with hypoxia-induced gasping (ChAT,
Dbx1, Vgat and Vglut2 lines; Neuropixels 1.0). They are on the lab drive at `Z:/ramirezlab/nbush/projects/dynaresp`
(not public: "available upon reasonable request"). Per gate, `data/processed/<mouse>/catgt_<mouse>_<gate>/` holds
the NIDQ file and `<mouse>_<gate>_imec0/` with Nick's CatGT output (`-aphipass=300 -gbldmx -gfix`, the whole ~44 min
gate) and his Kilosort 2 sort. Set `dynaresp` in `configs/local_paths.yaml` to that `processed` folder.

m2021-31_g4 (Dbx1) is set up. `sorting/configs/m2021-31_g4_hypoxia_window.yaml` sorts 2050–2648 s of the gate: room
air, hypercapnia, 10% O2 and gasping (timings from the gas-valve lines on the NIDQ; see the config). Its `nidq`
section maps the NIDQ channels (diaphragm EMG on XA1, plethysmography on XA0).

```bat
conda activate npxsort
cd sorting
python run_sorting.py configs\m2021-31_g4_hypoxia_window.yaml
cd ..
conda activate cibrrig
python analysis\population\population_dynamics.py --sort <outputs>\bush2024\m2021-31_g4\hypoxia_window --labels good --conditions-file analysis\population\configs\m2021-31_g4_conditions.json --out <outputs>\bush2024\m2021-31_g4\population_dynamics
```

`--labels good` uses bombcell-good units until the units are reviewed in the browser. Without it, the analysis uses
the units reviewed `single`, as for e097. The gate ends while the mouse is still gasping, so there is no recovery
period. For another gate: copy the config, change `input.folder`, the window, the epochs and `nidq.bin`, and write a
conditions file from that gate's gas-valve times.

m2020-19 (Vglut2) is the only mouse in this data that never had hypercapnia: room air, then N2, as in e097. Two of
its gates are set up, sorted with Kilosort 4 only: g3 (1140–1540 s: two N2 challenges with a hypoxia between; one of
the paper's 39) and g4 (1140–1415 s: one N2 and 151 s of recovery). In this mouse the tissue moves 40–130 µm during
gasping, more toward the top of the probe, and an uncorrected sort splits every neuron across the move. The gates are
therefore sorted from recordings corrected with a motion map measured from the units an uncorrected sort split
(`sorting/anchor_motion.py`; configs `m2020-19_g*_anoxia_window_anchor.yaml` with the build and apply commands in
their headers; maps in `sorting/configs/motion/`). Bridged rigid DREDGE (`motion.bridge_s`, configs
`m2020-19_g*_anoxia_window.yaml`) kept fewer neurons as one unit across the move and is kept only for comparison. Run
the population analysis like m2021-31_g4, with `--sort <outputs>\bush2024\m2020-19_g3\anoxia_window_anchor` (or g4) and
the matching `analysis\population\configs\m2020-19_g*_conditions.json`. This mouse's gas valves are on NIDQ digital
bits 3 (N2) and 4 (hypoxia), which the gate list doesn't name.

## 6. Known gaps

- Lupin isn't reproducible run to run, and Kilosort 4 on a GPU hasn't been tested for exact repeats (analysis record
  F6, O8). Use the shared sort to reproduce numbers.
- `state_examples_K6_by_epoch.png` and `hmm_states_pca_K6_by_epoch*.png` (PC1–PC3, `_PC1-PC2`, `_PC2-PC3`) in both
  runs' `figures\` come from functions in `transition_windows.py` (`state_examples_shared`, `states_pca_shared`) that
  no script calls yet. Call them with both run folders, e.g. `python -c "import sys; sys.path.insert(0,
  r'analysis\population'); import transition_windows as tw; tw.states_pca_shared([tw.pdm.OUT /
  'population_dynamics_v2', tw.pdm.OUT / 'population_dynamics_anoxia2'])"`.
- Only e097's two anoxia windows and its laser window are analysed with this pipeline. The diving-reflex and pre-laser
  analyses, and the other animals, used the archived pipeline and haven't been redone.
