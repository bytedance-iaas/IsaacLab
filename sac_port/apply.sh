#!/bin/bash
# Apply the SAC and FlashSAC patches, in order, to the installed rsl-rl-lib (5.4.1) in the active Python environment.
# A patch whose algorithm module already exists is skipped, so this also upgrades an environment with only SAC.
set -e
D=$(dirname "$(readlink -f "$0")")
SP=$(python -c "import rsl_rl, os; print(os.path.dirname(os.path.dirname(rsl_rl.__file__)))")
cd "$SP"
for alg in sac flashsac; do
    if [ -e "rsl_rl/algorithms/$alg.py" ]; then
        echo "rsl_rl_$alg.patch already applied"
        continue
    fi
    patch -p1 --dry-run < "$D/rsl_rl_$alg.patch" > /dev/null
    patch -p1 < "$D/rsl_rl_$alg.patch"
done
