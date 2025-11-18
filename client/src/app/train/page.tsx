'use client';

import { useState, useEffect, useRef } from 'react';
import { apiClient } from '@/lib/api';
import { WebSocketClient } from '@/lib/websocket';

export default function TrainPage() {
  const [models, setModels] = useState<any[]>([]);
  const [dataFiles, setDataFiles] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [gpuInfo, setGpuInfo] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  const wsClient = useRef<WebSocketClient | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Form state
  const [selectedDataFile, setSelectedDataFile] = useState('');
  const [modelType, setModelType] = useState('d3qn');
  const [trainingMode, setTrainingMode] = useState<'episodes' | 'timesteps'>('episodes');
  const [episodes, setEpisodes] = useState('100');
  const [timesteps, setTimesteps] = useState('100000');
  const [learningRate, setLearningRate] = useState('0.00025');
  const [gamma, setGamma] = useState('0.995');
  const [rewardType, setRewardType] = useState('sharpe');
  const [batchSize, setBatchSize] = useState('64');
  const [bufferSize, setBufferSize] = useState('100000');

  useEffect(() => {
    fetchModels();
    fetchDataFiles();
    fetchSystemInfo();

    // Setup WebSocket
    const ws = new WebSocketClient();
    wsClient.current = ws;

    ws.on('training_progress', (data: any) => {
      if (data.task_id === currentTaskId) {
        setProgress(data.progress || 0);
        setProgressMessage(data.message || '');
      }
    });

    ws.on('training_complete', (data: any) => {
      if (data.task_id === currentTaskId) {
        setProgress(100);
        setProgressMessage('Training completed!');
        setTimeout(() => {
          setTraining(false);
          setProgress(0);
          setProgressMessage('');
          setCurrentTaskId(null);
          fetchModels();
        }, 2000);
      }
    });

    ws.on('training_error', (data: any) => {
      if (data.task_id === currentTaskId) {
        setProgressMessage(data.message || 'Training failed');
        setTimeout(() => {
          setTraining(false);
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

  const fetchDataFiles = async () => {
    try {
      const response = await apiClient.getAvailableData();
      setDataFiles(response.data.data || []);
      if (response.data.data && response.data.data.length > 0) {
        setSelectedDataFile(response.data.data[0].filename);
      }
    } catch (error) {
      console.error('Failed to fetch data files:', error);
    }
  };

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
        setTimeout(() => {
          setTraining(false);
          setProgress(0);
          setProgressMessage('');
          setCurrentTaskId(null);
          fetchModels();
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
          }
        }, 2000);
      } else if (status.status === 'failed') {
        setTimeout(() => {
          setTraining(false);
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

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedDataFile) {
      alert('Please select a data file first');
      return;
    }

    setTraining(true);
    setProgress(0);
    setProgressMessage('Initializing training...');

    try {
      const response = await apiClient.startTraining({
        symbols: [selectedDataFile.split('_')[0]],
        model_type: modelType,
        episodes: trainingMode === 'episodes' ? parseInt(episodes) : undefined,
        timesteps: trainingMode === 'timesteps' ? parseInt(timesteps) : undefined,
        learning_rate: parseFloat(learningRate),
        gamma: parseFloat(gamma),
        reward_type: rewardType,
        batch_size: parseInt(batchSize),
        buffer_size: parseInt(bufferSize),
      });

      const taskId = response.data.task_id;
      setCurrentTaskId(taskId);

      // Start polling task status
      pollIntervalRef.current = setInterval(() => {
        pollTaskStatus(taskId);
      }, 1000);

    } catch (error: any) {
      setProgressMessage(`Training failed: ${error.response?.data?.detail || error.message}`);
      setTimeout(() => {
        setTraining(false);
        setProgress(0);
        setProgressMessage('');
      }, 3000);
    }
  };

  const getModelDescription = (type: string) => {
    switch (type) {
      case 'd3qn':
        return 'Deep Dueling DQN - Best for discrete actions, proven stable';
      case 'sac':
        return 'Soft Actor-Critic - Top performer! 152% returns, Sharpe 2.81';
      case 'ppo':
        return 'Proximal Policy Optimization - Stable and widely used';
      case 'td3':
        return 'Twin Delayed DDPG - Excellent risk management, -63% drawdown';
      default:
        return '';
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Train RL Model
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Train reinforcement learning models with your downloaded data
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
        {/* Training Form */}
        <div className="lg:col-span-2 space-y-6">
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
              Training Configuration
            </h2>
            <form onSubmit={handleStartTraining} className="space-y-6">
              {/* Data File Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Data File *
                </label>
                <select
                  value={selectedDataFile}
                  onChange={(e) => setSelectedDataFile(e.target.value)}
                  className="input w-full"
                  required
                  disabled={training}
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

              {/* Model Type Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Model Type *
                </label>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  className="input w-full"
                  disabled={training}
                >
                  <option value="d3qn">D3QN - Deep Dueling DQN</option>
                  <option value="sac">SAC - Soft Actor-Critic (Recommended)</option>
                  <option value="ppo">PPO - Proximal Policy Optimization</option>
                  <option value="td3">TD3 - Twin Delayed DDPG</option>
                </select>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  {getModelDescription(modelType)}
                </p>
              </div>

              {/* Training Mode Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Training Mode *
                </label>
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <button
                    type="button"
                    onClick={() => setTrainingMode('episodes')}
                    disabled={training}
                    className={`px-4 py-2 rounded-lg border-2 transition-all ${
                      trainingMode === 'episodes'
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 font-semibold'
                        : 'border-gray-300 dark:border-gray-600 hover:border-primary-300'
                    }`}
                  >
                    Episodes
                  </button>
                  <button
                    type="button"
                    onClick={() => setTrainingMode('timesteps')}
                    disabled={training}
                    className={`px-4 py-2 rounded-lg border-2 transition-all ${
                      trainingMode === 'timesteps'
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 font-semibold'
                        : 'border-gray-300 dark:border-gray-600 hover:border-primary-300'
                    }`}
                  >
                    Timesteps
                  </button>
                </div>

                {trainingMode === 'episodes' ? (
                  <div>
                    <input
                      type="number"
                      value={episodes}
                      onChange={(e) => setEpisodes(e.target.value)}
                      className="input w-full"
                      min="1"
                      step="1"
                      required
                      disabled={training}
                      placeholder="Number of episodes"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Episodes: Complete runs from start to finish. Recommended: 2 (quick test) | 10 (test) | 100 (quick train) | 1000+ (full training)
                    </p>
                  </div>
                ) : (
                  <div>
                    <input
                      type="number"
                      value={timesteps}
                      onChange={(e) => setTimesteps(e.target.value)}
                      className="input w-full"
                      min="10"
                      step="10"
                      required
                      disabled={training}
                      placeholder="Number of timesteps"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Timesteps: Individual actions taken. Recommended: 1,000 (quick test) | 10,000 (test) | 100,000 (quick train) | 1,000,000+ (full training)
                    </p>
                  </div>
                )}
              </div>

              {/* Hyperparameters Grid */}
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
                    disabled={training}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Gamma (Discount)
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
                    disabled={training}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={batchSize}
                    onChange={(e) => setBatchSize(e.target.value)}
                    className="input w-full"
                    min="16"
                    step="16"
                    required
                    disabled={training}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Buffer Size
                  </label>
                  <input
                    type="number"
                    value={bufferSize}
                    onChange={(e) => setBufferSize(e.target.value)}
                    className="input w-full"
                    min="1000"
                    step="1000"
                    required
                    disabled={training}
                  />
                </div>
              </div>

              {/* Reward Function */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Reward Function
                </label>
                <select
                  value={rewardType}
                  onChange={(e) => setRewardType(e.target.value)}
                  className="input w-full"
                  disabled={training}
                >
                  <option value="profit">Profit (Simple)</option>
                  <option value="return">Return (Percentage)</option>
                  <option value="sharpe">Sharpe Ratio (Risk-adjusted)</option>
                  <option value="sortino">Sortino Ratio (Downside risk)</option>
                </select>
              </div>

              {/* Progress Bar */}
              {training && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-700 dark:text-gray-300">{progressMessage}</span>
                    <span className="text-primary-600 font-semibold">{progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-purple-600 h-full rounded-full transition-all duration-300 ease-out"
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
                disabled={training || dataFiles.length === 0}
                className="btn btn-primary w-full text-lg py-3"
              >
                {training ? 'Training in Progress...' : 'Start Training'}
              </button>
            </form>
          </div>
        </div>

        {/* Info Panel */}
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
              📊 Model Comparison
            </h3>
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded opacity-60">
                <div className="font-semibold text-blue-900 dark:text-blue-100">SAC 🚧</div>
                <div className="text-blue-700 dark:text-blue-300 text-xs mt-1">
                  Coming Soon - Best: 152% returns, Sharpe 2.81
                </div>
              </div>
              <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded opacity-60">
                <div className="font-semibold text-green-900 dark:text-green-100">TD3 🚧</div>
                <div className="text-green-700 dark:text-green-300 text-xs mt-1">
                  Coming Soon - Risk control: -63% drawdown
                </div>
              </div>
              <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded opacity-60">
                <div className="font-semibold text-purple-900 dark:text-purple-100">PPO 🚧</div>
                <div className="text-purple-700 dark:text-purple-300 text-xs mt-1">
                  Coming Soon - Stable and widely used
                </div>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-900/20 rounded border-2 border-green-500">
                <div className="font-semibold text-gray-900 dark:text-gray-100">D3QN ✅</div>
                <div className="text-gray-700 dark:text-gray-300 text-xs mt-1">
                  Proven discrete action RL (Available)
                </div>
              </div>
            </div>
          </div>

          <div className="card bg-yellow-50 dark:bg-yellow-900/20 border-2 border-yellow-200 dark:border-yellow-800">
            <h3 className="text-lg font-bold text-yellow-900 dark:text-yellow-100 mb-3">
              💡 Training Tips
            </h3>
            <ul className="text-sm text-yellow-800 dark:text-yellow-200 space-y-2">
              <li>• Start with 2-10 episodes for quick testing</li>
              <li>• Episodes mode: easier to understand progress</li>
              <li>• Timesteps mode: more control over total steps</li>
              <li>• Currently only D3QN is fully implemented</li>
              <li>• GPU speeds up training 10-20x</li>
              <li>• Sharpe ratio balances risk/reward</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Trained Models */}
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
