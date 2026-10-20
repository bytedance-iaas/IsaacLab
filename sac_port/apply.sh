#!/bin/bash
# Apply the SAC patch to the installed rsl-rl-lib (5.4.1) in the active Python environment.
set -e
SP=$(python -c "import rsl_rl, os; print(os.path.dirname(os.path.dirname(rsl_rl.__file__)))")
cd "$SP"
patch -p1 --dry-run < "$(dirname "$(readlink -f "$0")")/rsl_rl_sac.patch" > /dev/null
patch -p1 < "$(dirname "$(readlink -f "$0")")/rsl_rl_sac.patch"
