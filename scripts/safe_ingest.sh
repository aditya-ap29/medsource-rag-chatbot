#!/bin/bash

# Safe data ingestion script that prevents OpenMP/threading conflicts
# on macOS ARM architecture

echo "🔧 Setting up environment for macOS ARM compatibility..."

# Prevent OpenMP conflicts between different libraries
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Disable PyTorch parallelism to avoid conflicts
export OMP_DISPLAY_ENV=FALSE

# Set memory allocator for better stability
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "✅ Environment configured"
echo "   - Single-threaded mode enabled"
echo "   - This will be slower but stable"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run the ingestion script
echo "📥 Starting data ingestion..."
python scripts/ingest_data.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Data ingestion completed successfully!"
else
    echo ""
    echo "❌ Data ingestion failed with exit code: $exit_code"
fi

exit $exit_code
