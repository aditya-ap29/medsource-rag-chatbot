#!/bin/bash
# MedSource RAG Chatbot - One-Command Setup Script
# Usage: bash start.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo ""
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_success "Python $python_version detected"
}

# Create virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Could not activate virtual environment"
        exit 1
    fi
}

# Install dependencies
install_deps() {
    print_info "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Setup environment file
setup_env() {
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_success "Created .env file"
        print_warning "Please edit .env and add your API keys!"
        
        # Ask user for LLM provider
        echo ""
        echo "Which LLM provider would you like to use?"
        echo "1) OpenAI GPT-4 (requires API key, $$$)"
        echo "2) Mistral 7B (open-source, free, runs locally)"
        echo "3) Qwen 1.5 7B (open-source, free, runs locally)"
        read -p "Enter choice (1-3): " llm_choice
        
        case $llm_choice in
            1)
                echo "LLM_PROVIDER=openai" >> .env
                print_info "Set to OpenAI. Don't forget to add your API key to .env!"
                ;;
            2)
                echo "LLM_PROVIDER=mistral" >> .env
                print_success "Set to Mistral (open-source)"
                ;;
            3)
                echo "LLM_PROVIDER=qwen" >> .env
                print_success "Set to Qwen (open-source)"
                ;;
            *)
                echo "LLM_PROVIDER=openai" >> .env
                print_warning "Invalid choice, defaulting to OpenAI"
                ;;
        esac
    else
        print_info ".env file already exists"
    fi
}

# Create directories
create_dirs() {
    mkdir -p data/raw_documents
    mkdir -p data/vector_store
    mkdir -p data/processed
    mkdir -p logs
    print_success "Created data directories"
}

# Ingest sample data
ingest_data() {
    print_info "Creating sample medical data and building vector store..."
    python scripts/ingest_data.py --source sample
    print_success "Vector store created successfully"
}

# Start API server
start_server() {
    print_header "🚀 Starting MedSource API Server"
    print_info "Server will start at http://localhost:8000"
    print_info "Press Ctrl+C to stop the server"
    echo ""
    print_info "Once started, you can:"
    echo "  - Visit http://localhost:8000/docs for API documentation"
    echo "  - Run 'python scripts/test_api.py' in another terminal to test"
    echo ""
    
    python -m src.api
}

# Main setup flow
main() {
    print_header "🏥 MedSource RAG Chatbot - Setup"
    
    # Check Python
    print_info "Checking Python installation..."
    check_python
    
    # Setup virtual environment
    print_info "Setting up virtual environment..."
    setup_venv
    activate_venv
    
    # Install dependencies
    install_deps
    
    # Setup environment
    print_info "Configuring environment..."
    setup_env
    
    # Create directories
    print_info "Creating directories..."
    create_dirs
    
    # Ingest data
    print_header "📚 Data Ingestion"
    ingest_data
    
    # All done
    print_header "✅ Setup Complete!"
    echo ""
    print_success "MedSource is ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Edit .env if needed:  nano .env"
    echo "  2. Start server:          bash start.sh server"
    echo "  3. Test API:              python scripts/test_api.py"
    echo "  4. View docs:             http://localhost:8000/docs"
    echo ""
    
    # Ask if user wants to start server
    read -p "Start API server now? (y/n): " start_now
    if [ "$start_now" = "y" ] || [ "$start_now" = "Y" ]; then
        start_server
    else
        echo ""
        print_info "To start server later, run: bash start.sh server"
    fi
}

# Handle command line arguments
if [ "$1" = "server" ]; then
    # Just start server
    activate_venv
    start_server
elif [ "$1" = "test" ]; then
    # Run tests
    activate_venv
    python scripts/test_api.py
elif [ "$1" = "ingest" ]; then
    # Re-run data ingestion
    activate_venv
    python scripts/ingest_data.py --source "${2:-sample}"
else
    # Full setup
    main
fi
