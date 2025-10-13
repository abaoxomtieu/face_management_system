#!/bin/bash

echo "🎭 Starting Face Recognition Streamlit App (Updated for class_id)"
echo "================================================================="

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source /Users/baohoton/Desktop/Code/Job/WISE/BE/.venv/bin/activate

# Check if Streamlit is installed
echo "📦 Checking Streamlit..."
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Streamlit not installed. Installing..."
    pip install -r requirements_streamlit.txt
fi

# Check if API is running
echo "🔍 Checking API status..."
curl -s http://localhost:9001/api/face-recognition/health > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ API is not running. Please start it first:"
    echo "   python app.py"
    exit 1
fi

echo "✅ API is running"

# Test API integration
echo "🧪 Testing API integration..."
if python3 test_streamlit_ui.py > /dev/null 2>&1; then
    echo "✅ API integration test passed"
else
    echo "⚠️  API integration test failed - check API server"
fi

# Start Streamlit
echo "📱 Starting Streamlit app..."
echo "📋 Features available:"
echo "   - Class Management (class_id support)"
echo "   - Student Management (class_id support)"
echo "   - Attendance Testing (class_id support)"
echo "   - Student Deletion (complete removal)"
echo "   - System Monitoring"
echo ""

streamlit run streamlit_app.py --server.port 8501
