/**
 * API Client for RL Crypto Trading Backend
 */

import axios, { AxiosInstance } from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Health & System
  async healthCheck() {
    return this.client.get('/api/health');
  }

  async getSystemInfo() {
    return this.client.get('/api/system/info');
  }

  async getConfig() {
    return this.client.get('/api/config');
  }

  // Data Management
  async downloadData(data: {
    symbols: string[];
    start_date: string;
    end_date: string;
    timeframe?: string;
    exchange?: string;
  }) {
    return this.client.post('/api/data/download', data);
  }

  async getAvailableData() {
    return this.client.get('/api/data/available');
  }

  // Training
  async startTraining(data: {
    symbols: string[];
    model_type: string;
    timesteps: number;
    learning_rate?: number;
    gamma?: number;
    reward_type?: string;
  }) {
    return this.client.post('/api/train/start', data);
  }

  async pauseTraining(taskId: string) {
    return this.client.post(`/api/train/pause/${taskId}`);
  }

  async resumeTraining(taskId: string) {
    return this.client.post(`/api/train/resume/${taskId}`);
  }

  async cancelTraining(taskId: string) {
    return this.client.post(`/api/train/cancel/${taskId}`);
  }

  async getAvailableModels() {
    return this.client.get('/api/models/available');
  }

  // Backtesting
  async runBacktest(data: {
    model_path: string;
    symbols: string[];
    start_date: string;
    end_date: string;
  }) {
    return this.client.post('/api/backtest/run', data);
  }

  async getBacktestResults(taskId: string) {
    return this.client.get(`/api/backtest/results/${taskId}`);
  }

  // Paper Trading
  async startPaperTrading(data: {
    model_path: string;
    symbol: string;
    initial_balance?: number;
    max_leverage?: number;
    fee_rate?: number;
    update_interval?: number;
  }) {
    return this.client.post('/api/paper-trading/start', data);
  }

  async stopPaperTrading() {
    return this.client.post('/api/paper-trading/stop');
  }

  async getPaperTradingStatus() {
    return this.client.get('/api/paper-trading/status');
  }

  // Tasks
  async getTaskStatus(taskId: string) {
    return this.client.get(`/api/tasks/${taskId}`);
  }

  async listTasks() {
    return this.client.get('/api/tasks');
  }
}

export const apiClient = new APIClient();
export default apiClient;
