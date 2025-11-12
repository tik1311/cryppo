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
  const lastDataLengthRef = useRef<number>(0)
  const priceLineOverlaysRef = useRef<string[]>([])

  const [klineCharts, setKlineCharts] = useState<any>(null)
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['MA'])
  const [showSettings, setShowSettings] = useState(false)
  const [currentTool, setCurrentTool] = useState<string>('')

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
    { value: 'MA', label: 'MA', name: 'Moving Average' },
    { value: 'EMA', label: 'EMA', name: 'Exponential MA' },
    { value: 'BOLL', label: 'BOLL', name: 'Bollinger' },
    { value: 'MACD', label: 'MACD', name: 'MACD' },
    { value: 'RSI', label: 'RSI', name: 'RSI' },
    { value: 'KDJ', label: 'KDJ', name: 'KDJ' }
  ]

  const drawingTools = [
    { value: 'segment', label: 'Trendline', icon: '📐' },
    { value: 'rayLine', label: 'Ray Line', icon: '📏' },
    { value: 'fibonacciLine', label: 'Fibonacci', icon: '📊' },
    { value: 'parallelStraightLine', label: 'Channel', icon: '⫽' },
    { value: 'priceChannelLine', label: 'Price Channel', icon: '📉' }
  ]

  // Load klinecharts library
  useEffect(() => {
    import('klinecharts').then((module) => {
      setKlineCharts(module)
    })
  }, [])

  // Convert data format for klinecharts
  const convertData = (data: any[]) => {
    return data.map(d => ({
      timestamp: (typeof d.time === 'number' ? d.time : parseInt(d.time)) * 1000,
      open: parseFloat(d.open),
      high: parseFloat(d.high),
      low: parseFloat(d.low),
      close: parseFloat(d.close),
      volume: parseFloat(d.volume || 0)
    }))
  }

  // Initialize chart ONCE
  useEffect(() => {
    if (!chartRef.current || !klineCharts) return

    // Initialize chart with dark theme
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

    // Cleanup on unmount
    return () => {
      if (chartInstanceRef.current) {
        klineCharts.dispose(chartRef.current!)
        chartInstanceRef.current = null
      }
    }
  }, [klineCharts])

  // Load initial data or handle timeframe change
  useEffect(() => {
    if (!chartInstanceRef.current || rawData.length === 0) return

    const chartData = convertData(rawData)

    // Use applyNewData for initial load or timeframe change
    chartInstanceRef.current.applyNewData(chartData, false)
    lastDataLengthRef.current = rawData.length

    // Scroll to real-time (rightmost)
    setTimeout(() => {
      chartInstanceRef.current?.scrollToRealTime()
    }, 100)
  }, [klineCharts, timeframe]) // Only when library loads or timeframe changes

  // Add default indicators after chart initialization
  useEffect(() => {
    if (!chartInstanceRef.current || !klineCharts) return

    // Add MA indicator on candle pane by default
    chartInstanceRef.current.createIndicator('MA', true, { id: 'candle_pane' })
  }, [chartInstanceRef.current, klineCharts])

  // Handle real-time updates
  useEffect(() => {
    if (!chartInstanceRef.current || rawData.length === 0) return

    const chartData = convertData(rawData)

    // If data length increased, we have new candles
    if (rawData.length > lastDataLengthRef.current) {
      // Apply all new data
      chartInstanceRef.current.applyNewData(chartData, false)
      lastDataLengthRef.current = rawData.length

      // Scroll to real-time
      setTimeout(() => {
        chartInstanceRef.current?.scrollToRealTime()
      }, 50)
    } else if (rawData.length === lastDataLengthRef.current) {
      // Same length - update last candle
      const latestCandle = chartData[chartData.length - 1]
      chartInstanceRef.current.updateData(latestCandle)
    }
  }, [rawData])

  // Handle price lines (separate from drawing tools)
  useEffect(() => {
    if (!chartInstanceRef.current || !klineCharts) return

    // Remove old price line overlays
    priceLineOverlaysRef.current.forEach(id => {
      chartInstanceRef.current.removeOverlay({ id })
    })
    priceLineOverlaysRef.current = []

    if (!priceLines) return

    // Add new price lines
    if (priceLines.entry) {
      const id = chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.entry }],
        styles: {
          line: { color: '#3b82f6', size: 2 },
          text: { color: '#3b82f6' }
        },
        extendData: 'Entry'
      })
      if (id) priceLineOverlaysRef.current.push(id)
    }

    if (priceLines.takeProfit) {
      const id = chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.takeProfit }],
        styles: {
          line: { color: '#22c55e', size: 2 },
          text: { color: '#22c55e' }
        },
        extendData: 'TP'
      })
      if (id) priceLineOverlaysRef.current.push(id)
    }

    if (priceLines.stopLoss) {
      const id = chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.stopLoss }],
        styles: {
          line: { color: '#ef4444', size: 2 },
          text: { color: '#ef4444' }
        },
        extendData: 'SL'
      })
      if (id) priceLineOverlaysRef.current.push(id)
    }

    if (priceLines.breakEven) {
      const id = chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.breakEven }],
        styles: {
          line: { color: '#eab308', size: 1, style: 'dashed' },
          text: { color: '#eab308' }
        },
        extendData: 'BE'
      })
      if (id) priceLineOverlaysRef.current.push(id)
    }

    if (priceLines.liquidation) {
      const id = chartInstanceRef.current.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.liquidation }],
        styles: {
          line: { color: '#a855f7', size: 2, style: 'dashed' },
          text: { color: '#a855f7' }
        },
        extendData: 'Liq'
      })
      if (id) priceLineOverlaysRef.current.push(id)
    }
  }, [priceLines, klineCharts])

  const handleDrawingTool = (toolName: string) => {
    if (!chartInstanceRef.current) return

    if (currentTool === toolName) {
      setCurrentTool('')
      return
    }

    setCurrentTool(toolName)

    // Create overlay for drawing
    chartInstanceRef.current.createOverlay({
      name: toolName,
      lock: false,
      mode: 'normal'
    })
  }

  const clearDrawings = () => {
    if (!chartInstanceRef.current) return

    // Clear only drawing tool overlays, not price lines
    drawingTools.forEach(tool => {
      chartInstanceRef.current.removeOverlay({ name: tool.value })
    })
    setCurrentTool('')
  }

  const toggleIndicator = (indicator: string) => {
    if (!chartInstanceRef.current) return

    if (selectedIndicators.includes(indicator)) {
      setSelectedIndicators(selectedIndicators.filter(i => i !== indicator))
      chartInstanceRef.current.removeIndicator('candle_pane', indicator)
    } else {
      setSelectedIndicators([...selectedIndicators, indicator])

      // Oscillator indicators go to separate pane
      if (indicator === 'MACD' || indicator === 'RSI' || indicator === 'KDJ') {
        chartInstanceRef.current.createIndicator(indicator, false)
      } else {
        // Overlay indicators on candle pane
        chartInstanceRef.current.createIndicator(indicator, true, { id: 'candle_pane' })
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

  if (rawData.length === 0) {
    return (
      <div className="card p-8 text-center text-gray-500" style={{ height: `${height}px` }}>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="text-gray-400 mb-2">No data available</div>
            <div className="text-sm">Start paper trading to see chart</div>
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
                title={ind.name}
              >
                {ind.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Drawing Tools */}
      <div className="p-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="text-xs font-semibold text-gray-400 mr-2">Drawing Tools:</div>
          {drawingTools.map((tool) => (
            <button
              key={tool.value}
              onClick={() => handleDrawingTool(tool.value)}
              className={`px-3 py-1.5 text-sm font-medium rounded transition-colors ${
                currentTool === tool.value
                  ? 'bg-primary text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              title={tool.label}
            >
              {tool.icon} {tool.label}
            </button>
          ))}
          <button
            onClick={clearDrawings}
            className="px-3 py-1.5 text-sm font-medium rounded bg-red-900 text-red-300 hover:bg-red-800 transition-colors ml-2"
          >
            🗑️ Clear Drawings
          </button>
          {currentTool && (
            <div className="text-xs text-yellow-400 ml-auto">
              ✏️ Click and drag on chart to draw {drawingTools.find(t => t.value === currentTool)?.label}
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
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <div>
                  <div className="text-gray-400">Entry</div>
                  <div className="font-semibold text-white">${priceLines.entry.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.takeProfit && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <div>
                  <div className="text-gray-400">Take Profit</div>
                  <div className="font-semibold text-green-400">${priceLines.takeProfit.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.stopLoss && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div>
                  <div className="text-gray-400">Stop Loss</div>
                  <div className="font-semibold text-red-400">${priceLines.stopLoss.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.breakEven && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div>
                  <div className="text-gray-400">Break Even</div>
                  <div className="font-semibold text-yellow-400">${priceLines.breakEven.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.liquidation && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                <div>
                  <div className="text-gray-400">Liquidation</div>
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
