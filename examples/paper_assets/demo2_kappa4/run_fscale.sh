#!/bin/bash
# D2: F-amplitude scaling study.  The residual (sim - theory) should be
# c3 s^3 + c5 s^5 if it is the order-4 / order-6 truncation of the
# kappa^3 channel; c3 is then directly comparable to the exact
# F^3.kappa^3 the package now computes.
#
# s = 1 reuses the existing 2M runs in sims/ (F_scale defaults to 1).
# s = 0.5 needs ~20M for 3 sigma on a residual predicted at c3/8 ~ 1.2e-5.
# s = 1.5 needs only 2M (predicted residual ~3.2e-4) and mainly pins c5.
#
# Usage: ./run_fscale.sh [n_parallel]
set -u
NP=${1:-22}
cd "$(dirname "$0")"
run () {  # $1 = F_scale, $2 = seed, $3 = n_real
  out="sims_fscale/sim_F$1_dt0.02_s$2.npz"
  [ -f "$out" ] && return 0
  python sim_dt_study.py --dt 0.02 --n_real "$3" --seed "$2" --F_scale "$1" \
      --out "$out" > "sims_fscale/sim_F$1_dt0.02_s$2.log" 2>&1
}
export -f run
{
  for s in $(seq 300 499); do echo "0.5 $s 100000"; done
  for s in $(seq 700 719); do echo "1.5 $s 100000"; done
} | xargs -P "$NP" -n 3 bash -c 'run "$0" "$1" "$2"'
echo "ALL FSCALE RUNS DONE"
