#!/usr/bin/env python3
"""
Quick start script for Advanced DeepDream Implementation
"""

import sys
import os
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'streamlit', 'pillow', 
        'matplotlib', 'numpy', 'opencv-python', 'scikit-image'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def run_streamlit_app():
    """Run the Streamlit web interface"""
    print("🚀 Starting Advanced DeepDream Web Interface...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def run_command_line():
    """Run the command-line version"""
    print("🎨 Running DeepDream command-line version...")
    
    try:
        from advanced_deepdream import main
        main()
    except Exception as e:
        print(f"❌ Error running DeepDream: {e}")

def main():
    """Main function"""
    print("🎨 Advanced DeepDream Implementation")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Ask user what to run
    print("\nWhat would you like to run?")
    print("1. 🌐 Web Interface (Streamlit)")
    print("2. 💻 Command Line Version")
    print("3. 🧪 Run Tests")
    print("4. ❌ Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_streamlit_app()
    elif choice == "2":
        run_command_line()
    elif choice == "3":
        print("🧪 Running tests...")
        subprocess.run([sys.executable, "-m", "pytest", "test_deepdream.py", "-v"])
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
