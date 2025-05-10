# scripts/run.py
import argparse
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import config
from src.config import API_HOST, API_PORT, UI_PORT

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, description, check=True):
    """Run a command with proper error handling"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{description}")
    logger.info(f"{'='*60}")
    
    try:
        if check:
            subprocess.run(command, shell=True, check=True)
        else:
            process = subprocess.Popen(command, shell=True)
            return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def download_dataset():
    """Download and process the dataset"""
    logger.info("Starting dataset download and processing...")
    run_command(
        f"{sys.executable} scripts/download_dataset.py",
        "Downloading and processing fashion dataset"
    )

def extract_features(subset_size=None, upload_to_cloudinary=False):
    """Extract features from dataset"""
    logger.info("Starting feature extraction...")
    
    # Build command
    cmd = f"{sys.executable} scripts/extract_features.py"
    
    # Add subset size if specified
    if subset_size:
        cmd += f" --subset_size {subset_size}"
    
    # Add cloudinary flag if requested
    if upload_to_cloudinary:
        cmd += " --upload_to_cloudinary"
    
    run_command(cmd, f"Extracting features{f' (subset: {subset_size})' if subset_size else ' (full dataset)'}")

def run_api(host=API_HOST, port=API_PORT, reload=True):
    """Run the FastAPI server"""
    logger.info(f"Starting API server at http://{host}:{port}")
    
    cmd = f"uvicorn src.api.app:app --host {host} --port {port}"
    if reload:
        cmd += " --reload"
    
    process = run_command(cmd, "Starting FastAPI server", check=False)
    
    # Wait for server to start
    logger.info("Waiting for API server to start...")
    time.sleep(3)
    
    return process

def run_ui(port=UI_PORT):
    """Run the Streamlit UI"""
    logger.info(f"Starting Streamlit UI at http://localhost:{port}")
    
    cmd = f"streamlit run src/ui/streamlit_app.py --server.port {port}"
    process = run_command(cmd, "Starting Streamlit UI", check=False)
    
    return process

def test_environment():
    """Run environment tests"""
    logger.info("Testing environment setup...")
    run_command(
        f"{sys.executable} tests/test_environment.py",
        "Running environment tests"
    )

def main():
    """Main entry point for the runner script"""
    parser = argparse.ArgumentParser(
        description="Fashion Recommendation System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Action arguments
    parser.add_argument("--test", action="store_true", 
                        help="Test environment setup")
    parser.add_argument("--download", action="store_true", 
                        help="Download and process dataset")
    parser.add_argument("--extract", action="store_true", 
                        help="Extract features from dataset")
    parser.add_argument("--api", action="store_true", 
                        help="Run API server")
    parser.add_argument("--ui", action="store_true", 
                        help="Run Streamlit UI")
    parser.add_argument("--all", action="store_true", 
                        help="Run complete pipeline (download, extract, api, ui)")
    
    # Configuration arguments
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Number of images to process (default: all)")
    parser.add_argument("--upload-to-cloudinary", action="store_true",
                        help="Upload images to Cloudinary")
    parser.add_argument("--no-reload", dest="reload", action="store_false",
                        help="Disable auto-reload for API")
    parser.add_argument("--api-host", type=str, default=API_HOST,
                        help="API host")
    parser.add_argument("--api-port", type=int, default=API_PORT,
                        help="API port")
    parser.add_argument("--ui-port", type=int, default=UI_PORT,
                        help="UI port")
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not any([args.test, args.download, args.extract, args.api, args.ui, args.all]):
        parser.print_help()
        return
    
    # If --all is specified, run everything
    if args.all:
        args.download = True
        args.extract = True
        args.api = True
        args.ui = True
    
    try:
        # Test environment
        if args.test:
            test_environment()
        
        # Download dataset
        if args.download:
            download_dataset()
        
        # Extract features
        if args.extract:
            extract_features(
                subset_size=args.subset_size,
                upload_to_cloudinary=args.upload_to_cloudinary
            )
        
        # Start servers
        processes = []
        
        if args.api:
            api_process = run_api(
                host=args.api_host,
                port=args.api_port,
                reload=args.reload
            )
            processes.append(api_process)
        
        if args.ui:
            ui_process = run_ui(port=args.ui_port)
            processes.append(ui_process)
        
        # Wait for processes if any were started
        if processes:
            logger.info("\nServers are running. Press Ctrl+C to stop.")
            try:
                for process in processes:
                    process.wait()
            except KeyboardInterrupt:
                logger.info("\nShutting down servers...")
                for process in processes:
                    process.terminate()
                    process.wait()
                logger.info("Servers stopped.")
        
        logger.info("\nAll tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()