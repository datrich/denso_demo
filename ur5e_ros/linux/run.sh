#!/usr/bin/env bash
# Chay 1 tool trong ur5e_ros/tools voi moi truong ROS + venv:
#   bash ur5e_ros/linux/run.sh check_sim --move
#   bash ur5e_ros/linux/run.sh pick_place_demo
set -eo pipefail  # khong dung -u: setup.bash cua ROS tham chieu bien chua dat
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
tool="$1"; shift
cd "$REPO_ROS"
exec python "tools/${tool%.py}.py" "$@"
