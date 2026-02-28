#!/usr/bin/env bash
# Extract judgment PDFs and metadata from tar archives.
#
# Usage:
#   ./extract-judgments.sh 2024 2025
#   ./extract-judgments.sh 1950 2026

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <start-year> <end-year>"
    echo "Example: $0 2024 2025"
    exit 1
fi

START_YEAR=$1
END_YEAR=$2

DATA_BASE="judgments-data/data/tar"
META_BASE="judgments-data/metadata/tar"

echo "Extracting tar files for years $START_YEAR to $END_YEAR..."

for ((year=START_YEAR; year<=END_YEAR; year++)); do
    echo ""
    echo "Processing year $year..."

    # --- Data tars (english only) ---
    ENGLISH_DIR="$DATA_BASE/year=$year/english"
    ENGLISH_TAR="$ENGLISH_DIR/english.tar"

    if [[ -f "$ENGLISH_TAR" ]]; then
        EXTRACT_DIR="$ENGLISH_DIR/english"
        echo "  Extracting: $ENGLISH_TAR"
        echo "  To: $EXTRACT_DIR"

        # Clean existing extraction
        if [[ -d "$EXTRACT_DIR" ]]; then
            echo "  Removing existing directory..."
            rm -rf "$EXTRACT_DIR"
        fi

        mkdir -p "$EXTRACT_DIR"

        if tar -xf "$ENGLISH_TAR" -C "$EXTRACT_DIR" 2>/dev/null; then
            echo "  ✓ Extracted successfully"
        else
            echo "  ✗ Error extracting data tar" >&2
        fi
    else
        echo "  English tar not found: $ENGLISH_TAR"
    fi

    # --- Metadata tars ---
    META_DIR="$META_BASE/year=$year"

    if [[ -d "$META_DIR" ]]; then
        for tarfile in "$META_DIR"/*.tar; do
            [[ -f "$tarfile" ]] || continue

            tarname=$(basename "$tarfile" .tar)
            extract_dir="$META_DIR/$tarname"

            echo "  Extracting: $tarfile"
            echo "  To: $extract_dir"

            if [[ -d "$extract_dir" ]]; then
                echo "  Removing existing directory..."
                rm -rf "$extract_dir"
            fi

            mkdir -p "$extract_dir"

            if tar -xf "$tarfile" -C "$extract_dir" 2>/dev/null; then
                echo "  ✓ Extracted successfully"
            else
                echo "  ✗ Error extracting: $tarfile" >&2
            fi
        done
    else
        echo "  Metadata path not found: $META_DIR"
    fi
done

echo ""
echo "Extraction complete!"
