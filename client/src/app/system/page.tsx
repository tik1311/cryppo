'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/lib/api';

export default function SystemPage() {
  const [config, setConfig] = useState<any>(null);
  const [systemInfo, setSystemInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [configRes, infoRes] = await Promise.all([
        apiClient.getConfig(),
        apiClient.getSystemInfo(),
      ]);
      setConfig(configRes.data);
      setSystemInfo(infoRes.data);
    } catch (error) {
      console.error('Failed to fetch system data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600 dark:text-gray-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        System Configuration
      </h1>

      {/* System Info */}
      <div className="card mb-8">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          System Information
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">CUDA Available</p>
            <p className="text-lg font-bold text-gray-900 dark:text-white">
              {systemInfo?.cuda_available ? '✓ Yes' : '✗ No'}
            </p>
            {systemInfo?.cuda_device && (
              <p className="text-xs text-gray-500 mt-1">{systemInfo.cuda_device}</p>
            )}
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Number of GPUs</p>
            <p className="text-lg font-bold text-gray-900 dark:text-white">
              {systemInfo?.num_gpus || 0}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Python Version</p>
            <p className="text-lg font-bold text-gray-900 dark:text-white">
              {systemInfo?.python_version?.split(' ')[0] || 'N/A'}
            </p>
          </div>
        </div>
      </div>

      {/* Trading Config */}
      {config?.trading && (
        <div className="card mb-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Trading Parameters
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Initial Balance</p>
              <p className="font-medium text-gray-900 dark:text-white">
                ${config.trading.initial_balance.toLocaleString()}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Max Leverage</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {config.trading.max_leverage}x
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Fee (Taker)</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {(config.trading.fee_taker * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Fee (Maker)</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {(config.trading.fee_maker * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {(config.trading.max_drawdown * 100).toFixed(0)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Stop Loss</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {(config.trading.stop_loss_pct * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Model Config */}
      {config?.model && (
        <div className="card mb-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Model Configuration
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Model Type</p>
              <p className="font-medium text-gray-900 dark:text-white uppercase">
                {config.model.type}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Hidden Layers</p>
              <p className="font-medium text-gray-900 dark:text-white font-mono">
                {config.model.network.hidden_layers.join(', ')}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Learning Rate</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {config.model.training.learning_rate}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Gamma</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {config.model.training.gamma}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Batch Size</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {config.model.training.batch_size}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Reward Type</p>
              <p className="font-medium text-gray-900 dark:text-white capitalize">
                {config.model.reward_type}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Data Config */}
      {config?.data && (
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Data Configuration
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Timeframe</p>
              <p className="font-medium text-gray-900 dark:text-white uppercase">
                {config.data.timeframe}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Window Size</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {config.data.window_size} hours
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Symbols</p>
              <p className="font-medium text-gray-900 dark:text-white">
                {config.data.symbols.join(', ')}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
