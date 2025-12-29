#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3 Video Segmentation API Runner

This script starts the FastAPI server for the SAM3 video segmentation API.
"""

import argparse
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 Video Segmentation API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAM3 Video Segmentation API")
    print("=" * 60)
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"ReDoc: http://{args.host}:{args.port}/redoc")
    print("=" * 60)
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=1,  # SAM3 requires single worker due to GPU memory management
    )


if __name__ == "__main__":
    main()

