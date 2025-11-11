'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/lib/api';
import Link from 'next/link';

export default function Home() {
  const [systemInfo, setSystemInfo] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [healthRes, infoRes] = await Promise.all([
        apiClient.healthCheck(),
        apiClient.getSystemInfo(),
      ]);
      setHealth(healthRes.data);
      setSystemInfo(infoRes.data);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          RL Crypto Trading System
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          Deep Reinforcement Learning for 1-Hour Cryptocurrency Trading
        </p>
        <div className="mt-4">
          {health && (
            <span className="badge badge-success">
              {health.status === 'healthy' ? '● Online' : '● Offline'}
            </span>
          )}
        </div>
      </div>

      {/* System Info Cards */}
      {systemInfo && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="card">
            <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
              GPU Status
            </h3>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {systemInfo.cuda_available ? '✓ Available' : '✗ Not Available'}
            </p>
            {systemInfo.cuda_device && (
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                {systemInfo.cuda_device}
              </p>
            )}
          </div>

          <div className="card">
            <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
              Timeframe
            </h3>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {systemInfo.timeframe?.toUpperCase() || '1H'}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              1-Hour Candlesticks Only
            </p>
          </div>

          <div className="card">
            <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
              Supported Symbols
            </h3>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {systemInfo.supported_symbols?.length || 0}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              {systemInfo.supported_symbols?.slice(0, 2).join(', ')}...
            </p>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Link href="/data" className="group">
          <div className="card hover:shadow-xl transition-shadow cursor-pointer">
            <div className="text-4xl mb-4">📊</div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              Data Management
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Download and manage 1H market data
            </p>
          </div>
        </Link>

        <Link href="/train" className="group">
          <div className="card hover:shadow-xl transition-shadow cursor-pointer">
            <div className="text-4xl mb-4">🚂</div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              Train Model
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Train D3QN agent on historical data
            </p>
          </div>
        </Link>

        <Link href="/backtest" className="group">
          <div className="card hover:shadow-xl transition-shadow cursor-pointer">
            <div className="text-4xl mb-4">📈</div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              Backtest
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Evaluate models on test data
            </p>
          </div>
        </Link>

        <Link href="/system" className="group">
          <div className="card hover:shadow-xl transition-shadow cursor-pointer">
            <div className="text-4xl mb-4">⚙️</div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              System
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Configuration and settings
            </p>
          </div>
        </Link>
      </div>

      {/* Features Section */}
      <div className="mt-12 card">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-bold text-lg text-gray-900 dark:text-white mb-2">
              ✓ Deep Dueling DQN (D3QN)
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              State-of-the-art reinforcement learning algorithm combining DQN, Double DQN, and Dueling architecture.
            </p>
          </div>
          <div>
            <h3 className="font-bold text-lg text-gray-900 dark:text-white mb-2">
              ✓ Technical Indicators
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              20+ indicators including RSI, MACD, Bollinger Bands, ATR, Stochastic, and more.
            </p>
          </div>
          <div>
            <h3 className="font-bold text-lg text-gray-900 dark:text-white mb-2">
              ✓ Realistic Simulation
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Accurate fee calculations, leverage support, and risk management with stop-loss/take-profit.
            </p>
          </div>
          <div>
            <h3 className="font-bold text-lg text-gray-900 dark:text-white mb-2">
              ✓ Comprehensive Backtesting
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Sharpe ratio, Sortino ratio, max drawdown, win rate, and profit factor metrics.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
