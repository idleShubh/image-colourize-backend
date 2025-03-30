import uvicorn
import os

if __name__ == "__main__":
    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configure uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload
        reload_dirs=[current_dir],  # Watch the current directory for changes
        workers=1,  # Use 1 worker for development
        log_level="info",
        timeout_keep_alive=30,  # Keep connections alive for 30 seconds
        limit_concurrency=1000,  # Limit concurrent connections
        limit_max_requests=10000,  # Limit maximum requests per worker
        loop="uvloop",  # Use uvloop for better performance
        http="h11",  # Use h11 for better performance
    ) 