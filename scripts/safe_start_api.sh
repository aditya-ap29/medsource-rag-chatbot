#!/bin/bash

# Safe API startup script that prevents OpenMP/threading conflicts
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
echo "   - Single-threaded mode for ML libraries"
echo "   - Multi-worker FastAPI still enabled"
echo ""

# Activate virtual environment
source venv/bin/activate

# Start the API server
echo "🚀 Starting MedSource RAG API..."
echo "   API will be available at: http://localhost:8000"
echo "   Docs available at: http://localhost:8000/docs"
echo ""

python -m src.api

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ API server stopped gracefully"
else
    echo ""
    echo "❌ API server exited with code: $exit_code"
fi

exit $exit_code
