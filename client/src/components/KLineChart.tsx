'use client'

import { useEffect, useRef, useState } from 'react'

interface CandleData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface KLineChartProps {
  data: any[]
  symbol: string
  height?: number
  timeframe?: string
  onTimeframeChange?: (tf: string) => void
  priceLines?: {
    entry?: number
    takeProfit?: number
    stopLoss?: number
    breakEven?: number
    liquidation?: number
  }
}

export default function KLineChart({
  data: rawData,
  symbol,
  height = 650,
  timeframe = '1m',
  onTimeframeChange,
  priceLines
}: KLineChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<any>(null)
  const [currentTool, setCurrentTool] = useState<string>('')
  const [klineCharts, setKlineCharts] = useState<any>(null)
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['MA'])
  const [showSettings, setShowSettings] = useState(false)

  const timeframes = [
    { value: '1m', label: '1m' },
    { value: '5m', label: '5m' },
    { value: '15m', label: '15m' },
    { value: '30m', label: '30m' },
    { value: '1h', label: '1H' },
    { value: '4h', label: '4H' },
    { value: '1d', label: '1D' }
  ]

  const indicators = [
    { value: 'MA', label: 'MA - Moving Average' },
    { value: 'EMA', label: 'EMA - Exponential MA' },
    { value: 'BOLL', label: 'BOLL - Bollinger Bands' },
    { value: 'MACD', label: 'MACD' },
    { value: 'RSI', label: 'RSI' },
    { value: 'KDJ', label: 'KDJ' }
  ]

  const drawingTools = [
    { value: 'segment', label: '📐 Trendline', icon: '📐' },
    { value: 'rayLine', label: '📏 Ray Line', icon: '📏' },
    { value: 'fibonacciLine', label: '📊 Fibonacci', icon: '📊' },
    { value: 'parallelStraightLine', label: '⫽ Channel', icon: '⫽' },
    { value: 'priceChannelLine', label: '📉 Price Channel', icon: '📉' }
  ]

  // Dynamic import klinecharts
  useEffect(() => {
    import('klinecharts').then((module) => {
      setKlineCharts(module)
    })
  }, [])

  // Convert data format
  const convertData = (data: any[]) => {
    return data.map(d => ({
      timestamp: (typeof d.time === 'number' ? d.time : parseInt(d.time)) * 1000,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume || 0
    }))
  }

  // Initialize chart
  useEffect(() => {
    if (!chartRef.current || !klineCharts || rawData.length === 0) return

    const chartData = convertData(rawData)

    chartInstanceRef.current = klineCharts.init(chartRef.current, {
      styles: {
        candle: {
          type: 'candle_solid',
          bar: {
            upColor: '#26a69a',
            downColor: '#ef5350',
            upBorderColor: '#26a69a',
            downBorderColor: '#ef5350',
            upWickColor: '#26a69a',
            downWickColor: '#ef5350'
          }
        },
        grid: {
          horizontal: {
            show: true,
            size: 1,
            color: '#292929',
            style: 'dashed'
          },
          vertical: {
            show: true,
            size: 1,
            color: '#292929',
            style: 'dashed'
          }
        },
        xAxis: {
          axisLine: {
            show: true,
            color: '#888888'
          }
        },
        yAxis: {
          axisLine: {
            show: true,
            color: '#888888'
          }
        },
        crosshair: {
          show: true,
          horizontal: {
            show: true,
            line: {
              color: '#888888',
              size: 1,
              style: 'dashed'
            }
          },
          vertical: {
            show: true,
            line: {
              color: '#888888',
              size: 1,
              style: 'dashed'
            }
          }
        }
      }
    })

    chartInstanceRef.current.applyNewData(chartData)

    // Add default indicators
    if (selectedIndicators.includes('MA')) {
      chartInstanceRef.current.createIndicator('MA', false, { id: 'candle_pane' })
    }

    return () => {
      if (chartInstanceRef.current) {
        klineCharts.dispose(chartRef.current!)
        chartInstanceRef.current = null
      }
    }
  }, [klineCharts, rawData])

  // Update latest candle
  useEffect(() => {
    if (chartInstanceRef.current && rawData.length > 0) {
      const latestCandle = convertData([rawData[rawData.length - 1]])[0]
      chartInstanceRef.current.updateData(latestCandle)
    }
  }, [rawData])

  // Draw price lines
  useEffect(() => {
    if (!chartInstanceRef.current || !priceLines || !klineCharts) return

    chartInstanceRef.current.removeOverlay()

    if (priceLines.entry) {
      chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.entry }],
        styles: {
          line: { color: '#3b82f6', size: 2 },
          text: { color: '#3b82f6' }
        },
        extendData: 'Entry'
      })
    }

    if (priceLines.takeProfit) {
      chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.takeProfit }],
        styles: {
          line: { color: '#22c55e', size: 2 },
          text: { color: '#22c55e' }
        },
        extendData: 'TP'
      })
    }

    if (priceLines.stopLoss) {
      chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.stopLoss }],
        styles: {
          line: { color: '#ef4444', size: 2 },
          text: { color: '#ef4444' }
        },
        extendData: 'SL'
      })
    }

    if (priceLines.breakEven) {
      chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.breakEven }],
        styles: {
          line: { color: '#eab308', size: 1, style: 'dashed' },
          text: { color: '#eab308' }
        },
        extendData: 'BE'
      })
    }

    if (priceLines.liquidation) {
      chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.liquidation }],
        styles: {
          line: { color: '#a855f7', size: 2, style: 'dashed' },
          text: { color: '#a855f7' }
        },
        extendData: 'Liq'
      })
    }
  }, [priceLines, klineCharts])

  const handleDrawingTool = (toolName: string) => {
    if (!chartInstanceRef.current) return

    if (currentTool === toolName) {
      setCurrentTool('')
      return
    }

    setCurrentTool(toolName)
    chartInstanceRef.current.createOverlay({
      name: toolName,
      lock: false
    })
  }

  const clearDrawings = () => {
    if (!chartInstanceRef.current) return
    chartInstanceRef.current.removeOverlay({ name: 'segment' })
    chartInstanceRef.current.removeOverlay({ name: 'rayLine' })
    chartInstanceRef.current.removeOverlay({ name: 'fibonacciLine' })
    chartInstanceRef.current.removeOverlay({ name: 'parallelStraightLine' })
    chartInstanceRef.current.removeOverlay({ name: 'priceChannelLine' })
    setCurrentTool('')
  }

  const toggleIndicator = (indicator: string) => {
    if (!chartInstanceRef.current) return

    if (selectedIndicators.includes(indicator)) {
      setSelectedIndicators(selectedIndicators.filter(i => i !== indicator))
      chartInstanceRef.current.removeIndicator({ name: indicator })
    } else {
      setSelectedIndicators([...selectedIndicators, indicator])
      if (indicator === 'MACD' || indicator === 'RSI' || indicator === 'KDJ') {
        chartInstanceRef.current.createIndicator(indicator, true)
      } else {
        chartInstanceRef.current.createIndicator(indicator, false, { id: 'candle_pane' })
      }
    }
  }

  if (!klineCharts) {
    return (
      <div className="card p-8 text-center text-gray-500" style={{ height: `${height}px` }}>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <div>Loading professional chart...</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="card p-0 overflow-hidden">
      {/* Top Toolbar */}
      <div className="p-3 bg-gray-900 border-b border-gray-700 flex items-center gap-4 flex-wrap">
        {/* Symbol */}
        <div className="text-lg font-bold text-white">{symbol}</div>

        {/* Timeframe Selector */}
        <div className="flex items-center gap-1">
          {timeframes.map((tf) => (
            <button
              key={tf.value}
              onClick={() => onTimeframeChange?.(tf.value)}
              className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                timeframe === tf.value
                  ? 'bg-primary text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>

        {/* Indicators Button */}
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="px-3 py-1 text-sm font-medium rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors"
        >
          📊 Indicators ({selectedIndicators.length})
        </button>
      </div>

      {/* Indicators Panel */}
      {showSettings && (
        <div className="p-4 bg-gray-800 border-b border-gray-700">
          <div className="text-sm font-semibold text-white mb-2">Technical Indicators</div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
            {indicators.map((ind) => (
              <button
                key={ind.value}
                onClick={() => toggleIndicator(ind.value)}
                className={`px-3 py-2 text-xs font-medium rounded transition-colors ${
                  selectedIndicators.includes(ind.value)
                    ? 'bg-primary text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {ind.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Drawing Tools */}
      <div className="p-3 bg-gray-850 border-b border-gray-700">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="text-xs font-semibold text-gray-400 mr-2">Drawing Tools:</div>
          {drawingTools.map((tool) => (
            <button
              key={tool.value}
              onClick={() => handleDrawingTool(tool.value)}
              className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                currentTool === tool.value
                  ? 'bg-primary text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              title={tool.label}
            >
              {tool.icon}
            </button>
          ))}
          <button
            onClick={clearDrawings}
            className="px-3 py-1.5 text-xs font-medium rounded bg-red-900 text-red-300 hover:bg-red-800 transition-colors ml-2"
          >
            🗑️ Clear
          </button>
          {currentTool && (
            <div className="text-xs text-gray-400 ml-auto">
              Click and drag on chart to draw
            </div>
          )}
        </div>
      </div>

      {/* Chart */}
      <div ref={chartRef} style={{ height: `${height}px`, width: '100%', backgroundColor: '#1a1a1a' }} />

      {/* Price Lines Legend */}
      {priceLines && (
        <div className="p-3 bg-gray-900 border-t border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
            {priceLines.entry && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                <div>
                  <div className="text-gray-400">Entry</div>
                  <div className="font-semibold text-white">${priceLines.entry.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.takeProfit && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500"></div>
                <div>
                  <div className="text-gray-400">TP</div>
                  <div className="font-semibold text-green-400">${priceLines.takeProfit.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.stopLoss && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-500"></div>
                <div>
                  <div className="text-gray-400">SL</div>
                  <div className="font-semibold text-red-400">${priceLines.stopLoss.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.breakEven && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                <div>
                  <div className="text-gray-400">BE</div>
                  <div className="font-semibold text-yellow-400">${priceLines.breakEven.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.liquidation && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                <div>
                  <div className="text-gray-400">Liq</div>
                  <div className="font-semibold text-purple-400">${priceLines.liquidation.toFixed(2)}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
