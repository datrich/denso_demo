#!/usr/bin/env bash
# Dung workspace DemoDiffusion tren server Linux (khong can sudo):
#   - uv (cai vao $WS/.local/bin)
#   - Isaac Lab Arena (+ submodule Isaac Lab) -> $WS/IsaacLab-Arena/.venv  (Isaac Sim 6.1 pip, Python 3.12)
#   - openpi (commit Arena da kiem chung) -> $WS/openpi/.venv  (JAX, Python 3.11)
#   - checkpoint pi0_droid_jointpos (~12 GB, gs://openpi-assets-simeval) -> $WS/.cache/openpi
# Chay:  bash demodiffusion/linux/setup_ws.sh          (lam lai an toan: buoc nao xong roi thi bo qua)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ARENA_COMMIT="${ARENA_COMMIT:-4be2c00}"                       # IsaacLab-Arena main 2026-10-13
OPENPI_COMMIT="${OPENPI_COMMIT:-c23745b5ad24e98f66967ea795a07b2588ed6c79}"  # = isaaclab_arena_openpi/docker/OPENPI_COMMIT
mkdir -p "$WS" "$WS/logs" "$UV_CACHE_DIR" "$OPENPI_DATA_HOME"
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step "uv"
if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WS/.local/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv --version

# submodule cua Arena dung git@github.com: -> doi sang https (server khong co khoa SSH GitHub)
GITHTTPS=(-c url."https://github.com/".insteadOf=git@github.com:)

step "Isaac Lab Arena @ $ARENA_COMMIT"
if [ ! -d "$WS/IsaacLab-Arena/.git" ]; then
    git "${GITHTTPS[@]}" clone -q https://github.com/isaac-sim/IsaacLab-Arena.git "$WS/IsaacLab-Arena"
fi
cd "$WS/IsaacLab-Arena"
git "${GITHTTPS[@]}" fetch -q origin && git checkout -q "$ARENA_COMMIT"
git "${GITHTTPS[@]}" submodule update --init --depth 1 submodules/IsaacLab   # bo qua Isaac-GR00T (khong can)
git log --oneline -1
uv sync 2>&1 | tail -5
.venv/bin/python -c "import isaacsim, isaaclab_arena; print('arena OK, isaacsim', isaacsim.__file__)"

step "openpi @ $OPENPI_COMMIT"
if [ ! -d "$WS/openpi/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none https://github.com/Physical-Intelligence/openpi.git "$WS/openpi"
fi
cd "$WS/openpi"
git checkout -q "$OPENPI_COMMIT"
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive -q
GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -5
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e packages/openpi-client -q
.venv/bin/python -c "from openpi.training import config; c = config.get_config('pi0_droid_jointpos_polaris'); print('openpi OK, config', c.name)"

step "checkpoint pi0_droid_jointpos (~12 GB)"
.venv/bin/python -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets-simeval/pi0_droid_jointpos')
print('checkpoint:', p)"

# openpi-client cho phia Arena (client websocket)
step "openpi-client trong venv Arena"
cd "$WS/IsaacLab-Arena" && uv pip install --python .venv/bin/python -e "$WS/openpi/packages/openpi-client" -q
.venv/bin/python -c "import openpi_client; print('openpi_client OK')"
step "XONG"
