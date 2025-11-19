#!/usr/bin/env python3
"""
CRYPPO - Main Entry Point
Simple wrapper around cryppo.py for convenience

Usage:
    python main.py check
    python main.py train --symbol BTCUSDT --episodes 1 --sample-data
    python main.py backtest --model checkpoints/final_agent.pth --sample-data
    python main.py models
    python main.py data list
    python main.py data download --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01
"""

if __name__ == "__main__":
    # Import and run the main function from cryppo.py
    from cryppo import main
    main()
