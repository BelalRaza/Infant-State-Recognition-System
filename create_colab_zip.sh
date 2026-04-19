#!/bin/bash
# Creates a zip file with everything needed for Google Colab.
# Run from the project root: bash create_colab_zip.sh

set -e

ZIP_NAME="infant_cry_colab.zip"
WRAPPER="Infant-State-Recognition-System"

echo "Creating Colab-ready zip file..."
echo "================================"

cd "$(dirname "$0")"
rm -f "$ZIP_NAME"

# Create a temp staging directory with the proper wrapper folder
STAGING=$(mktemp -d)
mkdir -p "$STAGING/$WRAPPER"

# Copy source code
cp -r src "$STAGING/$WRAPPER/src"
cp requirements.txt "$STAGING/$WRAPPER/"
cp Infant_Cry_Classifier_Colab.ipynb "$STAGING/$WRAPPER/"

# Copy ONLY original audio files (no _aug_ files)
echo "Copying original audio files (excluding _aug_)..."
for cls in hunger belly_pain burping discomfort tiredness; do
    mkdir -p "$STAGING/$WRAPPER/data/raw/$cls"
    for f in data/raw/$cls/*.wav; do
        case "$f" in
            *_aug_*) ;; # skip augmented
            *) cp "$f" "$STAGING/$WRAPPER/data/raw/$cls/" ;;
        esac
    done
    count=$(ls "$STAGING/$WRAPPER/data/raw/$cls/" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $cls: $count original files"
done

# Create empty output directories
mkdir -p "$STAGING/$WRAPPER/features"
mkdir -p "$STAGING/$WRAPPER/models"
mkdir -p "$STAGING/$WRAPPER/results/metrics"
mkdir -p "$STAGING/$WRAPPER/results/plots"

# Build zip
cd "$STAGING"
zip -r "$OLDPWD/$ZIP_NAME" "$WRAPPER" -x "*.pyc" "*__pycache__*"
cd "$OLDPWD"

# Cleanup
rm -rf "$STAGING"

echo ""
echo "================================"
echo "Created: $ZIP_NAME"
echo "Size: $(du -h "$ZIP_NAME" | cut -f1)"
echo ""
echo "Upload this file to Google Colab and run the notebook."
