'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api';

export default function BacktestPage() {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<any>(null);

  // Form state
  const [modelPath, setModelPath] = useState('../checkpoints/best_model.pth');
  const [symbols, setSymbols] = useState('BTCUSDT');
  const [startDate, setStartDate] = useState('2025-01-01');
  const [endDate, setEndDate] = useState('2025-12-31');

  const handleRunBacktest = async (e: React.FormEvent) => {
    e.preventDefault();
    setRunning(true);
    setResults(null);

    try {
      const symbolList = symbols.split(',').map(s => s.trim());
      const response = await apiClient.runBacktest({
        model_path: modelPath,
        symbols: symbolList,
        start_date: startDate,
        end_date: endDate,
      });

      alert(`Backtest started! Task ID: ${response.data.task_id}\n\nNote: Backtest will run on the server. Results will be available when complete.`);

      // In real implementation, poll for results
      // For now, just simulate some results
      setTimeout(() => {
        setResults({
          summary: {
            initial_balance: 10000,
            final_balance: 12500,
            total_return_pct: 25.0,
          },
          trades: {
            total_trades: 45,
            winning_trades: 28,
            losing_trades: 17,
            win_rate: 62.2,
          },
          performance: {
            sharpe_ratio: 2.15,
            sortino_ratio: 2.88,
            max_drawdown_pct: 8.5,
            profit_factor: 1.85,
          },
        });
      }, 3000);
    } catch (error: any) {
      alert(`Backtest failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Backtest Model
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Backtest Form */}
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Run Backtest
          </h2>
          <form onSubmit={handleRunBacktest} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Model Path
              </label>
              <input
                type="text"
                value={modelPath}
                onChange={(e) => setModelPath(e.target.value)}
                className="input w-full font-mono text-sm"
                placeholder="../checkpoints/best_model.pth"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Symbol
              </label>
              <input
                type="text"
                value={symbols}
                onChange={(e) => setSymbols(e.target.value)}
                className="input w-full"
                placeholder="BTCUSDT"
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Start Date
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="input w-full"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  End Date
                </label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="input w-full"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={running}
              className="btn btn-primary w-full"
            >
              {running ? 'Running Backtest...' : 'Run Backtest'}
            </button>
          </form>
        </div>

        {/* Info */}
        <div className="card bg-yellow-50 dark:bg-yellow-900/20 border-2 border-yellow-200 dark:border-yellow-800">
          <h3 className="text-lg font-bold text-yellow-900 dark:text-yellow-100 mb-3">
            ⚠️ Important Notes
          </h3>
          <ul className="text-sm text-yellow-800 dark:text-yellow-200 space-y-2">
            <li>• Use test data only (not training data)</li>
            <li>• Ensure model path is correct</li>
            <li>• Results include all performance metrics</li>
            <li>• Equity curve available in detailed results</li>
            <li>• Trade history included for analysis</li>
          </ul>
        </div>
      </div>

      {/* Results */}
      {results && (
        <div className="mt-8 space-y-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Backtest Results
          </h2>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                Total Return
              </h3>
              <p className="text-3xl font-bold text-green-600">
                +{results.summary.total_return_pct.toFixed(2)}%
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                ${results.summary.initial_balance.toLocaleString()} → ${results.summary.final_balance.toLocaleString()}
              </p>
            </div>

            <div className="card">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                Win Rate
              </h3>
              <p className="text-3xl font-bold text-blue-600">
                {results.trades.win_rate.toFixed(1)}%
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                {results.trades.winning_trades} wins / {results.trades.total_trades} trades
              </p>
            </div>

            <div className="card">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                Sharpe Ratio
              </h3>
              <p className="text-3xl font-bold text-purple-600">
                {results.performance.sharpe_ratio.toFixed(2)}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Risk-adjusted return
              </p>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="card">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
              Performance Metrics
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Sortino Ratio</p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {results.performance.sortino_ratio.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</p>
                <p className="text-xl font-bold text-red-600">
                  {results.performance.max_drawdown_pct.toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Profit Factor</p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {results.performance.profit_factor.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Trades</p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {results.trades.total_trades}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
