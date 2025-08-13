#!/bin/bash

# HM3D Scene Download Script for OVON
# Please set your Matterport credentials before running this script

# Check if credentials are set
if [ -z "$MATTERPORT_TOKEN_ID" ] || [ -z "$MATTERPORT_TOKEN_SECRET" ]; then
    echo "Error: Please set your Matterport credentials first:"
    echo "export MATTERPORT_TOKEN_ID=<your_token_id>"
    echo "export MATTERPORT_TOKEN_SECRET=<your_token_secret>"
    echo ""
    echo "You can get these from your Matterport account at:"
    echo "https://my.matterport.com/settings/account/devtools"
    exit 1
fi

# Set data directory
DATA_DIR="/home/yaoaa/habitat-lab/data"

echo "Downloading HM3D validation scenes..."
echo "Data will be saved to: $DATA_DIR"

# Download HM3D validation scenes
echo "Downloading HM3D val scenes..."
python -m habitat_sim.utils.datasets_download \
    --username $MATTERPORT_TOKEN_ID \
    --password $MATTERPORT_TOKEN_SECRET \
    --uids hm3d_val_v0.2 \
    --data-path $DATA_DIR

if [ $? -eq 0 ]; then
    echo "✅ HM3D validation scenes downloaded successfully!"
    echo "You can now run OVON experiments with the downloaded scenes."
else
    echo "❌ Failed to download HM3D scenes. Please check your credentials and internet connection."
fi

# Optional: Also download train scenes if needed
read -p "Do you also want to download HM3D training scenes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading HM3D train scenes..."
    python -m habitat_sim.utils.datasets_download \
        --username $MATTERPORT_TOKEN_ID \
        --password $MATTERPORT_TOKEN_SECRET \
        --uids hm3d_train_v0.2 \
        --data-path $DATA_DIR
        
    if [ $? -eq 0 ]; then
        echo "✅ HM3D training scenes downloaded successfully!"
    else
        echo "❌ Failed to download HM3D training scenes."
    fi
fi

echo "Download complete!"
