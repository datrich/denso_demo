# Dung chung cho cac script Linux (source, khong chay truc tiep).
# Bien co the dat truoc: ISAAC_ROOT, GPU, ROS_ENV, VENV, PIP_INDEX_URL

REPO_ROS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../denso_demo/ur5e_ros
REPO="$(dirname "$REPO_ROS")"                                    # .../denso_demo
ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaacsim}"
VENV="${VENV:-$REPO/.venv-linux}"
GPU="${GPU:-1}"

# ROS 2 Jazzy: dung env.sh cua workspace neu co (ROS_DOMAIN_ID, chi localhost), khong thi ROS he thong
ROS_ENV="${ROS_ENV:-$HOME/ros2_ws/env.sh}"
if [ -f "$ROS_ENV" ]; then
    # shellcheck disable=SC1090
    source "$ROS_ENV"
else
    source /opt/ros/jazzy/setup.bash
fi

if [ -d "$VENV" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
