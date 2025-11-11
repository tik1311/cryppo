'use client'

import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '@/lib/api'
import { WebSocketClient } from '@/lib/websocket'
import TradingChart from '@/components/TradingChart'
import MetricsDisplay from '@/components/MetricsDisplay'
import TradesList from '@/components/TradesList'
import { Time } from 'lightweight-charts'

interface PaperTradingConfig {
  model_path: string
  symbol: string
  initial_balance: number
  max_leverage: number
  fee_rate: number
  update_interval: number
}

interface PaperTrade {
  timestamp: number
  action: 'BUY' | 'SELL' | 'HOLD'
  price: number
  value: number
  pnl: number
  balance: number
}

interface CandleData {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export default function PaperTradingPage() {
  const [config, setConfig] = useState<PaperTradingConfig>({
    model_path: '',
    symbol: 'BTCUSDT',
    initial_balance: 10000,
    max_leverage: 3.0,
    fee_rate: 0.0004,
    update_interval: 5
  })

  const [models, setModels] = useState<any[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [metrics, setMetrics] = useState({
    balance: 10000,
    initial_balance: 10000,
    total_profit: 0,
    total_return: 0,
    win_rate: 0,
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    max_drawdown: 0,
    sharpe_ratio: 0,
    current_position: 'NEUTRAL' as 'LONG' | 'SHORT' | 'NEUTRAL',
    position_size: 0,
    unrealized_pnl: 0
  })

  const [trades, setTrades] = useState<PaperTrade[]>([])
  const [chartData, setChartData] = useState<CandleData[]>([])
  const [wsClient, setWsClient] = useState<WebSocketClient | null>(null)

  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await apiClient.getAvailableModels()
        const models = response.data.models || []
        setModels(models)
        if (models.length > 0) {
          setConfig(prev => ({ ...prev, model_path: models[0].path }))
        }
      } catch (err: any) {
        console.error('Error fetching models:', err)
      }
    }
    fetchModels()
  }, [])

  // Setup WebSocket
  useEffect(() => {
    const ws = new WebSocketClient()
    setWsClient(ws)

    ws.on('paper_trading_update', (data: any) => {
      // Update metrics
      if (data.metrics) {
        setMetrics(prev => ({ ...prev, ...data.metrics }))
      }

      // Add new trade
      if (data.trade) {
        setTrades(prev => [...prev, data.trade])
      }

      // Update chart
      if (data.candle) {
        setChartData(prev => {
          const newData = [...prev]
          const existingIndex = newData.findIndex(c => c.time === data.candle.time)
          if (existingIndex >= 0) {
            newData[existingIndex] = data.candle
          } else {
            newData.push(data.candle)
          }
          return newData.slice(-500) // Keep last 500 candles
        })
      }
    })

    ws.on('paper_trading_stopped', (data: any) => {
      setIsRunning(false)
      if (data.message) {
        alert(data.message)
      }
    })

    ws.connect()

    return () => {
      ws.disconnect()
    }
  }, [])

  const handleStart = async () => {
    if (!config.model_path) {
      setError('Please select a model')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await apiClient.startPaperTrading({
        model_path: config.model_path,
        symbol: config.symbol,
        initial_balance: config.initial_balance,
        max_leverage: config.max_leverage,
        fee_rate: config.fee_rate,
        update_interval: config.update_interval
      })

      setIsRunning(true)
      setTrades([])
      setChartData([])
      setMetrics({
        balance: config.initial_balance,
        initial_balance: config.initial_balance,
        total_profit: 0,
        total_return: 0,
        win_rate: 0,
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        max_drawdown: 0,
        sharpe_ratio: 0,
        current_position: 'NEUTRAL',
        position_size: 0,
        unrealized_pnl: 0
      })
    } catch (err: any) {
      setError(err.message || 'Failed to start paper trading')
      console.error('Error starting paper trading:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleStop = async () => {
    setIsLoading(true)
    try {
      await apiClient.stopPaperTrading()
      setIsRunning(false)
    } catch (err: any) {
      setError(err.message || 'Failed to stop paper trading')
      console.error('Error stopping paper trading:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Paper Trading</h1>
        <p className="mt-2 text-gray-600">
          Test your trained models in real-time simulation with live market data
        </p>
      </div>

      {/* Configuration Panel */}
      {!isRunning && (
        <div className="card">
          <h2 className="mb-4 text-xl font-semibold text-gray-900">Configuration</h2>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Model</label>
              <select
                className="input mt-1"
                value={config.model_path}
                onChange={(e) => setConfig({ ...config, model_path: e.target.value })}
              >
                <option value="">Select a model...</option>
                {models.map((model, idx) => (
                  <option key={idx} value={model.path}>
                    {model.filename} ({model.size_mb} MB)
                  </option>
                ))}
              </select>
            </div>

            {/* Symbol */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Symbol</label>
              <select
                className="input mt-1"
                value={config.symbol}
                onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
              >
                <option value="BTCUSDT">BTC/USDT</option>
                <option value="ETHUSDT">ETH/USDT</option>
                <option value="SOLUSDT">SOL/USDT</option>
                <option value="XRPUSDT">XRP/USDT</option>
                <option value="ADAUSDT">ADA/USDT</option>
              </select>
            </div>

            {/* Initial Balance */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Initial Balance ($)</label>
              <input
                type="number"
                className="input mt-1"
                value={config.initial_balance}
                onChange={(e) => setConfig({ ...config, initial_balance: parseFloat(e.target.value) })}
                min="100"
                step="100"
              />
            </div>

            {/* Max Leverage */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Max Leverage</label>
              <input
                type="number"
                className="input mt-1"
                value={config.max_leverage}
                onChange={(e) => setConfig({ ...config, max_leverage: parseFloat(e.target.value) })}
                min="1"
                max="10"
                step="0.5"
              />
            </div>

            {/* Fee Rate */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Fee Rate (%)</label>
              <input
                type="number"
                className="input mt-1"
                value={(config.fee_rate * 100).toFixed(2)}
                onChange={(e) => setConfig({ ...config, fee_rate: parseFloat(e.target.value) / 100 })}
                min="0"
                max="1"
                step="0.01"
              />
            </div>

            {/* Update Interval */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Update Interval (seconds)</label>
              <input
                type="number"
                className="input mt-1"
                value={config.update_interval}
                onChange={(e) => setConfig({ ...config, update_interval: parseInt(e.target.value) })}
                min="1"
                max="60"
              />
            </div>
          </div>

          {error && (
            <div className="mt-4 rounded-lg bg-danger/10 p-4 text-danger">
              {error}
            </div>
          )}

          <div className="mt-6 flex gap-3">
            <button
              onClick={handleStart}
              disabled={isLoading || !config.model_path}
              className="btn btn-primary"
            >
              {isLoading ? 'Starting...' : 'Start Paper Trading'}
            </button>
          </div>
        </div>
      )}

      {/* Live Trading Dashboard */}
      {isRunning && (
        <>
          {/* Metrics */}
          <MetricsDisplay metrics={metrics} isLive={true} />

          {/* Chart */}
          {chartData.length > 0 && (
            <TradingChart
              data={chartData}
              symbol={config.symbol.replace('USDT', '/USDT')}
              height={500}
            />
          )}

          {/* Trades */}
          <TradesList trades={trades} showPnL={true} maxItems={50} />

          {/* Control Buttons */}
          <div className="card">
            <div className="flex gap-3">
              <button
                onClick={handleStop}
                disabled={isLoading}
                className="btn bg-danger text-white hover:bg-danger/90"
              >
                {isLoading ? 'Stopping...' : 'Stop Paper Trading'}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
