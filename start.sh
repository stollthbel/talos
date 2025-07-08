#!/bin/bash

# Talos Capital - Phase 4/6 Start Script
echo "🏛️ Starting Talos Capital - Phase 4/6 Enhanced Trading Journal"
echo "=============================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "� Activating virtual environment..."
    source .venv/bin/activate
fi

# Start the Flask backend in the background
echo "🚀 Starting Flask backend on port 5000..."
/workspaces/talos/.venv/bin/python talos-backend.py &
BACKEND_PID=$!

# Give the backend time to start
sleep 3

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "✅ Backend started successfully (PID: $BACKEND_PID)"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start a simple HTTP server for the frontend
echo "🌐 Starting frontend server on port 3000..."
echo "📊 Access your Talos Capital dashboard at: http://localhost:3000"
echo "🔧 API backend available at: http://localhost:5000"
echo "📝 Trading Journal with rich text editing and drawing tools available!"
echo ""
echo "Features:"
echo "• 📖 Rich text trading journal with templates"
echo "• 🎨 Drawing and annotation tools"
echo "• 🧠 Mood tracking and market analysis"
echo "• 🏷️ Tags and private entries"
echo "• 📊 Advanced analytics and charts"
echo "• 💳 Stripe subscription management"
echo "• 🧪 Zero-knowledge proof integration"
echo ""
echo "Press Ctrl+C to stop all servers"

# Start Python HTTP server for frontend
cd /workspaces/talos
python3 -m http.server 3000

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down Talos Capital..."
    kill $BACKEND_PID 2>/dev/null
    echo "✅ All servers stopped"
    exit 0
}

# Trap Ctrl+C to cleanup
trap cleanup SIGINT

wait
