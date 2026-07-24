#!/bin/bash
# Final pre-submission reproduction queue.
# Reruns every multi-seed revision experiment in the paper, one by one, with
# the released code at HEAD (constant-lr revision protocol; --scheduler cosine
# only in the explicitly labelled A/B run). Fresh outputs land in
# results_rerun/ -- the historical results/ directory is left untouched.
set -u
cd "$(dirname "$0")/.."
R=results_rerun
mkdir -p "$R"
PY=${PY:-python}
S5="--seeds 1234 42 7 2024 31337"
S10="--seeds 1234 42 7 2024 31337 101 202 303 404 505"
S3="--seeds 1234 42 7"

run() { # run <name> <script> <args...>
  local name=$1; shift
  echo "=== $(date '+%m-%d %H:%M') $name: $*" | tee -a "$R/queue.log"
  if $PY "$@" --out "$R/$name.csv" > "$R/$name.log" 2>&1; then
    echo "--- $name DONE $(date '+%H:%M')" | tee -a "$R/queue.log"
  else
    echo "--- $name FAILED rc=$? (see $R/$name.log)" | tee -a "$R/queue.log"
  fi
}

# --- E1 revision: parity (10 ep) -------------------------------------------
run e1_alone_mnist experiments/run.py --dataset mnist $S5 --epochs 10
run e1_alone_sc    experiments/run.py --dataset speechcommands $S5 --epochs 10
run e1_alone_imdb  experiments/run.py --dataset imdb $S5 --epochs 10
run e1_joint       experiments/run.py --dataset mnist,speechcommands,imdb $S5 --epochs 10

# --- E1 low-resource transfer (AAN arms) -----------------------------------
run lr10_alone experiments/run.py --dataset imdb@0.1 $S5 --epochs 10
run lr10_joint experiments/run.py --dataset mnist,speechcommands,imdb@0.1 $S5 --epochs 10
run lr01_alone experiments/run.py --dataset imdb@0.01 $S10 --epochs 10
run lr01_joint experiments/run.py --dataset mnist,speechcommands,imdb@0.01 $S10 --epochs 10

# --- structure-free controls (flat MLP / attention / gated trunks) ----------
run ctl_mlp_alone_100 experiments/baseline_unified.py --dataset imdb $S5 --epochs 10 --trunk mlp
run ctl_mlp_joint_100 experiments/baseline_unified.py --dataset mnist,speechcommands,imdb $S5 --epochs 10 --trunk mlp
run ctl_mlp_alone_10  experiments/baseline_unified.py --dataset imdb@0.1 $S5 --epochs 10 --trunk mlp
run ctl_mlp_joint_10  experiments/baseline_unified.py --dataset mnist,speechcommands,imdb@0.1 $S5 --epochs 10 --trunk mlp
run ctl_mlp_alone_01  experiments/baseline_unified.py --dataset imdb@0.01 $S10 --epochs 10 --trunk mlp
run ctl_mlp_joint_01  experiments/baseline_unified.py --dataset mnist,speechcommands,imdb@0.01 $S10 --epochs 10 --trunk mlp
run ctl_mlp_alone_01_lr3e4 experiments/baseline_unified.py --dataset imdb@0.01 $S10 --epochs 10 --trunk mlp --lr 3e-4
run ctl_tf_alone_01   experiments/baseline_unified.py --dataset imdb@0.01 $S10 --epochs 10 --trunk transformer
run ctl_tf_joint_01   experiments/baseline_unified.py --dataset mnist,speechcommands,imdb@0.01 $S10 --epochs 10 --trunk transformer
run ctl_gru_alone_01  experiments/baseline_unified.py --dataset imdb@0.01 $S10 --epochs 10 --trunk gru
run ctl_gru_joint_01  experiments/baseline_unified.py --dataset mnist,speechcommands,imdb@0.01 $S10 --epochs 10 --trunk gru

# --- E2: root-embedding probes (30-ep frozen model, 3 seeds) ----------------
run e2_probe experiments/probe_e2.py --epochs 30 $S3

# --- E3: deep-chain regime (depth 81, 15 ep, batch 512) ---------------------
for v in ran gau tau ptau gtau tau1; do
  run e3_$v experiments/run.py --dataset speechcommands_mfcc --version $v $S5 --epochs 15 --batch-size 512
done

# --- E6: structural ablations (algorithms, 50 ep) ---------------------------
for a in none no-level-jump depth-pad edge-cut; do
  run e6_$a experiments/run.py --dataset algorithms --version gau $S5 --epochs 50 --ablate $a
done

# --- shallow references -----------------------------------------------------
run sc30_gaau experiments/run.py --dataset speechcommands $S5 --epochs 30
run sc30_tau  experiments/run.py --dataset speechcommands --version tau $S3 --epochs 30
run sc30_tau1 experiments/run.py --dataset speechcommands --version tau1 $S3 --epochs 30
run mnist5    experiments/run.py --dataset mnist $S5 --epochs 5
run e1_joint_tau  experiments/run.py --dataset mnist,speechcommands,imdb --version tau $S5 --epochs 10
run e1_joint_gtau experiments/run.py --dataset mnist,speechcommands,imdb --version gtau $S5 --epochs 10
run e1_joint_tau1 experiments/run.py --dataset mnist,speechcommands,imdb --version tau1 $S5 --epochs 10

# --- E5: all-in-one (50 ep, 3 seeds, single-domain batches) -----------------
run e5_all experiments/run.py --dataset mnist,speechcommands,speechcommands_mfcc,imdb,sst,algorithms,upfd,iris $S3 --epochs 50 --bucket-domains

# --- insurance A/B: original cosine schedule on the joint setting -----------
run ab_cosine_e1_joint experiments/run.py --dataset mnist,speechcommands,imdb --seeds 1234 42 --epochs 10 --scheduler cosine

echo "=== QUEUE COMPLETE $(date) ===" | tee -a "$R/queue.log"
