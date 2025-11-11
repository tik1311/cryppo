'use client';

import { useState, useEffect, useRef } from 'react';
import { apiClient } from '@/lib/api';
import { WebSocketClient } from '@/lib/websocket';
import TradingChart from '@/components/TradingChart';
import TradesList from '@/components/TradesList';
import { Time } from 'lightweight-charts';

export default function BacktestPage() {
  const [models, setModels] = useState<any[]>([]);
  const [dataFiles, setDataFiles] = useState<any[]>([]);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [gpuInfo, setGpuInfo] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  const wsClient = useRef<WebSocketClient | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Form state
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDataFile, setSelectedDataFile] = useState('');
  const [initialBalance, setInitialBalance] = useState('10000');
  const [maxLeverage, setMaxLeverage] = useState('3.0');
  const [feeRate, setFeeRate] = useState('0.04');
  const [useGPU, setUseGPU] = useState(true);

  useEffect(() => {
    fetchModels();
    fetchDataFiles();
    fetchSystemInfo();

    // Setup WebSocket
    const ws = new WebSocketClient();
    wsClient.current = ws;

    ws.on('backtest_progress', (data: any) => {
      if (data.task_id === currentTaskId) {
        setProgress(data.progress || 0);
        setProgressMessage(data.message || '');
      }
    });

    ws.on('backtest_complete', (data: any) => {
      if (data.task_id === currentTaskId) {
        setProgress(100);
        setProgressMessage('Backtest completed!');

        // Set results from data
        if (data.results) {
          setResults(data.results);
        }

        setTimeout(() => {
          setRunning(false);
          setProgress(0);
          setProgressMessage('');
          setCurrentTaskId(null);
        }, 2000);
      }
    });

    ws.on('backtest_error', (data: any) => {
      if (data.task_id === currentTaskId) {
        setProgressMessage(data.message || 'Backtest failed');
        setTimeout(() => {
          setRunning(false);
          setProgress(0);
          setProgressMessage('');
          setCurrentTaskId(null);
        }, 3000);
      }
    });

    ws.connect();

    return () => {
      ws.disconnect();
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [currentTaskId]);

  const fetchSystemInfo = async () => {
    try {
      const response = await apiClient.getSystemInfo();
      const info = response.data;
      setGpuAvailable(info.cuda_available);
      setGpuInfo(info.cuda_device || 'CPU Only');
    } catch (error) {
      console.error('Failed to fetch system info:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await apiClient.getAvailableModels();
      const modelsList = response.data.models || [];
      setModels(modelsList);
      if (modelsList.length > 0) {
        setSelectedModel(modelsList[0].path);
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const fetchDataFiles = async () => {
    try {
      const response = await apiClient.getAvailableData();
      const filesList = response.data.data || [];
      setDataFiles(filesList);
      if (filesList.length > 0) {
        setSelectedDataFile(filesList[0].filename);
      }
    } catch (error) {
      console.error('Failed to fetch data files:', error);
    }
  };

  const pollTaskStatus = async (taskId: string) => {
    try {
      const response = await apiClient.getTaskStatus(taskId);
      const status = response.data;

      if (status.progress !== undefined) {
        setProgress(status.progress);
      }
      if (status.message) {
        setProgressMessage(status.message);
      }

      if (status.status === 'completed') {
        setProgress(100);

        // Fetch results
        try {
          const resultsResponse = await apiClient.getBacktestResults(taskId);
          setResults(resultsResponse.data);
        } catch (err) {
          console.error('Failed to fetch results:', err);
        }

        setTimeout(() => {
          setRunning(false);
          setProgress(0);
          setProgressMessage('');
          setCurrentTaskId(null);
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
          }
        }, 2000);
      } else if (status.status === 'failed') {
        setTimeout(() => {
          setRunning(false);
          setProgress(0);
          setProgressMessage('');
          setCurrentTaskId(null);
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
          }
        }, 3000);
      }
    } catch (error) {
      console.error('Failed to poll task status:', error);
    }
  };

  const handleRunBacktest = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedModel) {
      alert('Please select a model first');
      return;
    }

    if (!selectedDataFile) {
      alert('Please select a data file first');
      return;
    }

    setRunning(true);
    setProgress(0);
    setProgressMessage('Initializing backtest...');
    setResults(null);

    try {
      const response = await apiClient.runBacktest({
        model_path: selectedModel,
        symbols: [selectedDataFile.split('_')[0]],
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        initial_balance: parseFloat(initialBalance),
        max_leverage: parseFloat(maxLeverage),
        fee_rate: parseFloat(feeRate),
        use_gpu: useGPU,
      });

      const taskId = response.data.task_id;
      setCurrentTaskId(taskId);

      // Start polling task status
      pollIntervalRef.current = setInterval(() => {
        pollTaskStatus(taskId);
      }, 1000);

    } catch (error: any) {
      setProgressMessage(`Backtest failed: ${error.response?.data?.detail || error.message}`);
      setTimeout(() => {
        setRunning(false);
        setProgress(0);
        setProgressMessage('');
      }, 3000);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Backtest Model
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Test your trained models on historical data
        </p>
      </div>

      {/* GPU Status */}
      <div className={`mb-6 p-4 rounded-lg border-2 ${gpuAvailable ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}>
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${gpuAvailable ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`}></div>
          <div>
            <div className="font-semibold text-gray-900">
              {gpuAvailable ? '🚀 GPU Available' : '⚠️ CPU Mode'}
            </div>
            <div className="text-sm text-gray-600">{gpuInfo}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Backtest Form */}
        <div className="lg:col-span-2 space-y-6">
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
              Backtest Configuration
            </h2>
            <form onSubmit={handleRunBacktest} className="space-y-6">
              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Trained Model *
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="input w-full"
                  required
                  disabled={running}
                >
                  {models.length === 0 ? (
                    <option value="">No models available</option>
                  ) : (
                    models.map((model, idx) => (
                      <option key={idx} value={model.path}>
                        {model.filename} ({model.size_mb} MB)
                      </option>
                    ))
                  )}
                </select>
                {models.length === 0 && (
                  <p className="mt-2 text-sm text-red-600">
                    Please train a model first from the Train page
                  </p>
                )}
              </div>

              {/* Data File Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Test Data *
                </label>
                <select
                  value={selectedDataFile}
                  onChange={(e) => setSelectedDataFile(e.target.value)}
                  className="input w-full"
                  required
                  disabled={running}
                >
                  {dataFiles.length === 0 ? (
                    <option value="">No data files available</option>
                  ) : (
                    dataFiles.map((file, idx) => (
                      <option key={idx} value={file.filename}>
                        {file.filename} ({file.size_mb} MB)
                      </option>
                    ))
                  )}
                </select>
                {dataFiles.length === 0 && (
                  <p className="mt-2 text-sm text-red-600">
                    Please download data first from the Data page
                  </p>
                )}
              </div>

              {/* Trading Parameters */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Initial Balance ($)
                  </label>
                  <input
                    type="number"
                    value={initialBalance}
                    onChange={(e) => setInitialBalance(e.target.value)}
                    className="input w-full"
                    min="100"
                    step="100"
                    required
                    disabled={running}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Max Leverage
                  </label>
                  <input
                    type="number"
                    value={maxLeverage}
                    onChange={(e) => setMaxLeverage(e.target.value)}
                    className="input w-full"
                    min="1"
                    max="10"
                    step="0.5"
                    required
                    disabled={running}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Trading Fee (%)
                </label>
                <input
                  type="number"
                  value={feeRate}
                  onChange={(e) => setFeeRate(e.target.value)}
                  className="input w-full"
                  min="0"
                  max="1"
                  step="0.01"
                  required
                  disabled={running}
                />
                <p className="mt-1 text-xs text-gray-500">
                  Typical: 0.02% (maker) - 0.04% (taker)
                </p>
              </div>

              {/* GPU Toggle */}
              {gpuAvailable && (
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={useGPU}
                    onChange={(e) => setUseGPU(e.target.checked)}
                    className="w-4 h-4"
                    disabled={running}
                  />
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Use GPU for faster backtesting
                  </label>
                </div>
              )}

              {/* Progress Bar */}
              {running && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-700 dark:text-gray-300">{progressMessage}</span>
                    <span className="text-primary-600 font-semibold">{progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-green-500 to-blue-600 h-full rounded-full transition-all duration-300 ease-out"
                      style={{ width: `${progress}%` }}
                    >
                      <div className="h-full w-full bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
                    </div>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={running || models.length === 0 || dataFiles.length === 0}
                className="btn btn-primary w-full text-lg py-3"
              >
                {running ? 'Running Backtest...' : 'Run Backtest'}
              </button>
            </form>
          </div>
        </div>

        {/* Info Panel */}
        <div className="space-y-6">
          <div className="card bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-200 dark:border-blue-800">
            <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-3">
              📊 Backtest Metrics
            </h3>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-2">
              <li>• Total Return %</li>
              <li>• Sharpe Ratio</li>
              <li>• Sortino Ratio</li>
              <li>• Max Drawdown</li>
              <li>• Win Rate</li>
              <li>• Profit Factor</li>
              <li>• Total Trades</li>
            </ul>
          </div>

          <div className="card bg-yellow-50 dark:bg-yellow-900/20 border-2 border-yellow-200 dark:border-yellow-800">
            <h3 className="text-lg font-bold text-yellow-900 dark:text-yellow-100 mb-3">
              💡 Backtest Tips
            </h3>
            <ul className="text-sm text-yellow-800 dark:text-yellow-200 space-y-2">
              <li>• Use out-of-sample data</li>
              <li>• Check multiple timeframes</li>
              <li>• Account for realistic fees</li>
              <li>• Avoid overfitting</li>
              <li>• GPU speeds up analysis</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Results Section */}
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
              <p className={`text-3xl font-bold ${(results.summary?.total_return_pct || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {(results.summary?.total_return_pct || 0) >= 0 ? '+' : ''}
                {(results.summary?.total_return_pct || 0).toFixed(2)}%
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                ${(results.summary?.initial_balance || 0).toLocaleString()} → ${(results.summary?.final_balance || 0).toLocaleString()}
              </p>
            </div>

            <div className="card">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                Win Rate
              </h3>
              <p className="text-3xl font-bold text-blue-600">
                {(results.trades?.win_rate || 0).toFixed(1)}%
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                {results.trades?.winning_trades || 0} wins / {results.trades?.total_trades || 0} trades
              </p>
            </div>

            <div className="card">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                Sharpe Ratio
              </h3>
              <p className="text-3xl font-bold text-purple-600">
                {(results.performance?.sharpe_ratio || 0).toFixed(2)}
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
                  {(results.performance?.sortino_ratio || 0).toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</p>
                <p className="text-xl font-bold text-red-600">
                  {(results.performance?.max_drawdown_pct || 0).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Profit Factor</p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {(results.performance?.profit_factor || 0).toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Trades</p>
                <p className="text-xl font-bold text-gray-900 dark:text-white">
                  {results.trades?.total_trades || 0}
                </p>
              </div>
            </div>
          </div>

          {/* Price Chart */}
          {results.chart_data && results.chart_data.length > 0 && (
            <TradingChart
              data={results.chart_data}
              symbol={selectedDataFile ? selectedDataFile.split('_')[0].replace('USDT', '/USDT') : 'BTC/USDT'}
              height={500}
            />
          )}

          {/* Trade History */}
          {results.trade_history && results.trade_history.length > 0 && (
            <TradesList
              trades={results.trade_history}
              showPnL={true}
              maxItems={30}
            />
          )}
        </div>
      )}
    </div>
  );
}
