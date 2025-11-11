'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';

export default function TrainPage() {
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);

  // Form state
  const [symbols, setSymbols] = useState('BTCUSDT');
  const [modelType, setModelType] = useState('d3qn');
  const [timesteps, setTimesteps] = useState('2000000');
  const [learningRate, setLearningRate] = useState('0.00025');
  const [gamma, setGamma] = useState('0.995');
  const [rewardType, setRewardType] = useState('sharpe');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await apiClient.getAvailableModels();
      setModels(response.data.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    setTraining(true);

    try {
      const symbolList = symbols.split(',').map(s => s.trim());
      const response = await apiClient.startTraining({
        symbols: symbolList,
        model_type: modelType,
        timesteps: parseInt(timesteps),
        learning_rate: parseFloat(learningRate),
        gamma: parseFloat(gamma),
        reward_type: rewardType,
      });

      alert(`Training started! Task ID: ${response.data.task_id}\n\nNote: This is a background task. Training will continue on the server.`);

      // Refresh models after a delay
      setTimeout(fetchModels, 5000);
    } catch (error: any) {
      alert(`Training failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Train D3QN Model
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Training Form */}
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Start New Training
          </h2>
          <form onSubmit={handleStartTraining} className="space-y-4">
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

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Model Type
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="input w-full"
              >
                <option value="d3qn">Deep Dueling DQN (D3QN)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Timesteps
              </label>
              <input
                type="number"
                value={timesteps}
                onChange={(e) => setTimesteps(e.target.value)}
                className="input w-full"
                min="100000"
                step="100000"
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  value={learningRate}
                  onChange={(e) => setLearningRate(e.target.value)}
                  className="input w-full"
                  step="0.00001"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Gamma
                </label>
                <input
                  type="number"
                  value={gamma}
                  onChange={(e) => setGamma(e.target.value)}
                  className="input w-full"
                  step="0.001"
                  min="0"
                  max="1"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Reward Function
              </label>
              <select
                value={rewardType}
                onChange={(e) => setRewardType(e.target.value)}
                className="input w-full"
              >
                <option value="profit">Profit</option>
                <option value="return">Return</option>
                <option value="sharpe">Sharpe Ratio</option>
                <option value="sortino">Sortino Ratio</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={training}
              className="btn btn-primary w-full"
            >
              {training ? 'Starting Training...' : 'Start Training'}
            </button>
          </form>
        </div>

        {/* Training Info */}
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
              Training Configuration
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Architecture:</span>
                <span className="font-medium text-gray-900 dark:text-white">[256, 256, 128]</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Batch Size:</span>
                <span className="font-medium text-gray-900 dark:text-white">64</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Replay Buffer:</span>
                <span className="font-medium text-gray-900 dark:text-white">100,000</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Target Update:</span>
                <span className="font-medium text-gray-900 dark:text-white">1,000 steps</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Epsilon Decay:</span>
                <span className="font-medium text-gray-900 dark:text-white">0.9995</span>
              </div>
            </div>
          </div>

          <div className="card bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-200 dark:border-blue-800">
            <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-2">
              ℹ️ Training Notes
            </h3>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-2">
              <li>• Training runs in background on server</li>
              <li>• Progress updates via WebSocket (Phase 2)</li>
              <li>• Models saved to checkpoints/ directory</li>
              <li>• Use CUDA GPU if available</li>
              <li>• Typical training: 4-8 hours</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Available Models */}
      <div className="card mt-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Trained Models
          </h2>
          <button
            onClick={fetchModels}
            disabled={loading}
            className="btn btn-secondary"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {models.length === 0 ? (
          <p className="text-gray-600 dark:text-gray-400 text-center py-8">
            No trained models available. Start training to create a model.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Filename
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Size
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Created
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Path
                  </th>
                </tr>
              </thead>
              <tbody>
                {models.map((model, index) => (
                  <tr
                    key={index}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    <td className="py-3 px-4 text-sm text-gray-900 dark:text-white font-mono">
                      {model.filename}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                      {model.size_mb} MB
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                      {new Date(model.created).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-500 dark:text-gray-500 font-mono text-xs">
                      {model.path}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
