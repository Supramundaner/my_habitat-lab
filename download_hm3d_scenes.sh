#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${MATTERPORT_TOKEN_ID:-}" || -z "${MATTERPORT_TOKEN_SECRET:-}" ]]; then
    echo "Error: Please set your Matterport credentials first:"
    echo "export MATTERPORT_TOKEN_ID=<your_token_id>"
    echo "export MATTERPORT_TOKEN_SECRET=<your_token_secret>"
    echo ""
    echo "You can get these from your Matterport account at:"
    echo "https://my.matterport.com/settings/account/devtools"
    exit 1
fi

if [[ -z "${HABITAT_DATA_ROOT:-}" ]]; then
    echo "Error: HABITAT_DATA_ROOT must be an absolute destination directory." >&2
    exit 1
fi
if [[ "${HABITAT_DATA_ROOT}" != /* ]]; then
    echo "Error: HABITAT_DATA_ROOT must be absolute: ${HABITAT_DATA_ROOT}" >&2
    exit 1
fi

readonly DATA_DIR="${HABITAT_DATA_ROOT}"
mkdir -p -- "${DATA_DIR}"

echo "Downloading HM3D validation scenes..."
echo "Data will be saved to: $DATA_DIR"

# Download HM3D validation scenes
echo "Downloading HM3D val scenes..."
python3 -m habitat_sim.utils.datasets_download \
    --username "${MATTERPORT_TOKEN_ID}" \
    --password "${MATTERPORT_TOKEN_SECRET}" \
    --uids hm3d_val_v0.2 \
    --data-path "${DATA_DIR}"

echo "HM3D validation scenes downloaded successfully."

if [[ "${1:-}" == "--include-train" ]]; then
    echo "Downloading HM3D train scenes..."
    python3 -m habitat_sim.utils.datasets_download \
        --username "${MATTERPORT_TOKEN_ID}" \
        --password "${MATTERPORT_TOKEN_SECRET}" \
        --uids hm3d_train_v0.2 \
        --data-path "${DATA_DIR}"
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--include-train]" >&2
    exit 2
fi

echo "Download complete!"
