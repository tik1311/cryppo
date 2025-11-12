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
  const [showIndicators, setShowIndicators] = useState(false)
  const [showDrawingTools, setShowDrawingTools] = useState(false)
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
    { value: 'MA', label: 'MA', name: 'Moving Average', category: 'Trend' },
    { value: 'EMA', label: 'EMA', name: 'Exponential MA', category: 'Trend' },
    { value: 'BOLL', label: 'BOLL', name: 'Bollinger Bands', category: 'Volatility' },
    { value: 'SAR', label: 'SAR', name: 'Parabolic SAR', category: 'Trend' },
    { value: 'MACD', label: 'MACD', name: 'MACD', category: 'Momentum' },
    { value: 'RSI', label: 'RSI', name: 'RSI', category: 'Momentum' },
    { value: 'KDJ', label: 'KDJ', name: 'KDJ', category: 'Momentum' },
    { value: 'VOL', label: 'VOL', name: 'Volume', category: 'Volume' }
  ]

  const drawingTools = [
    { value: 'segment', label: 'Trend Line', icon: '📐' },
    { value: 'rayLine', label: 'Ray', icon: '📏' },
    { value: 'priceLine', label: 'Horizontal Line', icon: '➖' },
    { value: 'priceChannelLine', label: 'Price Channel', icon: '📊' },
    { value: 'parallelStraightLine', label: 'Parallel Channel', icon: '⫽' },
    { value: 'fibonacciLine', label: 'Fibonacci Retracement', icon: '📉' },
    { value: 'rect', label: 'Rectangle', icon: '▭' },
    { value: 'circle', label: 'Circle', icon: '○' },
    { value: 'text', label: 'Text', icon: 'T' }
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

    // Initialize chart with professional dark theme
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
          },
          tooltip: {
            showRule: 'always',
            showType: 'rect',
            labels: ['T: ', 'O: ', 'C: ', 'H: ', 'L: ', 'V: '],
            text: {
              size: 12,
              family: 'Helvetica Neue',
              weight: 'normal',
              color: '#D9D9D9'
            }
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
            color: '#888888',
            size: 1
          },
          tickLine: {
            show: true,
            length: 3,
            size: 1,
            color: '#888888'
          },
          tickText: {
            show: true,
            color: '#D9D9D9',
            family: 'Helvetica Neue',
            weight: 'normal',
            size: 12
          }
        },
        yAxis: {
          axisLine: {
            show: true,
            color: '#888888',
            size: 1
          },
          tickLine: {
            show: true,
            length: 3,
            size: 1,
            color: '#888888'
          },
          tickText: {
            show: true,
            color: '#D9D9D9',
            family: 'Helvetica Neue',
            weight: 'normal',
            size: 12
          }
        },
        crosshair: {
          show: true,
          horizontal: {
            show: true,
            line: {
              show: true,
              style: 'dashed',
              dashValue: [4, 2],
              size: 1,
              color: '#888888'
            },
            text: {
              show: true,
              color: '#D9D9D9',
              size: 12,
              family: 'Helvetica Neue',
              weight: 'normal',
              paddingLeft: 2,
              paddingRight: 2,
              paddingTop: 2,
              paddingBottom: 2,
              borderSize: 1,
              borderColor: '#505050',
              backgroundColor: '#505050'
            }
          },
          vertical: {
            show: true,
            line: {
              show: true,
              style: 'dashed',
              dashValue: [4, 2],
              size: 1,
              color: '#888888'
            },
            text: {
              show: true,
              color: '#D9D9D9',
              size: 12,
              family: 'Helvetica Neue',
              weight: 'normal',
              paddingLeft: 2,
              paddingRight: 2,
              paddingTop: 2,
              paddingBottom: 2,
              borderSize: 1,
              borderColor: '#505050',
              backgroundColor: '#505050'
            }
          }
        },
        indicator: {
          tooltip: {
            showRule: 'always',
            showType: 'rect',
            showName: true,
            showParams: true,
            text: {
              size: 12,
              family: 'Helvetica Neue',
              weight: 'normal',
              color: '#D9D9D9'
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

    // Skip if this is initial load (handled by timeframe effect)
    if (lastDataLengthRef.current === 0) return

    const chartData = convertData(rawData)
    const latestCandle = chartData[chartData.length - 1]

    // Use updateData() for all real-time updates
    // It automatically detects:
    // - Same timestamp → update current candle
    // - New timestamp → append new candle
    chartInstanceRef.current.updateData(latestCandle)

    // Update our ref
    lastDataLengthRef.current = rawData.length

    // Auto-scroll to latest candle if new candle added
    if (rawData.length > lastDataLengthRef.current - 1) {
      setTimeout(() => {
        chartInstanceRef.current?.scrollToRealTime()
      }, 50)
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
      if (indicator === 'MACD' || indicator === 'RSI' || indicator === 'KDJ' || indicator === 'VOL') {
        chartInstanceRef.current.createIndicator(indicator, false)
      } else {
        // Overlay indicators on candle pane
        chartInstanceRef.current.createIndicator(indicator, true, { id: 'candle_pane' })
      }
    }
  }

  if (!klineCharts) {
    return (
      <div className="rounded-lg bg-[#0f0f0f] border border-gray-800 p-8 text-center text-gray-400" style={{ height: `${height}px` }}>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <div>Loading chart...</div>
          </div>
        </div>
      </div>
    )
  }

  if (rawData.length === 0) {
    return (
      <div className="rounded-lg bg-[#0f0f0f] border border-gray-800 p-8 text-center text-gray-400" style={{ height: `${height}px` }}>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="text-gray-500 mb-2">📊 No chart data available</div>
            <div className="text-sm text-gray-600">Start paper trading to begin</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-lg bg-[#0f0f0f] border border-gray-800 overflow-hidden">
      {/* Professional Top Toolbar */}
      <div className="px-4 py-2.5 bg-[#0a0a0a] border-b border-gray-800 flex items-center gap-6">
        {/* Symbol */}
        <div className="flex items-center gap-2">
          <div className="text-base font-bold text-white">{symbol}</div>
          <div className="px-1.5 py-0.5 text-[10px] font-medium bg-gray-800 text-gray-400 rounded">PERPETUAL</div>
        </div>

        {/* Timeframes */}
        <div className="flex items-center gap-1">
          {timeframes.map((tf) => (
            <button
              key={tf.value}
              onClick={() => onTimeframeChange?.(tf.value)}
              className={`px-2.5 py-1 text-xs font-medium rounded transition-all ${
                timeframe === tf.value
                  ? 'bg-[#1a73e8] text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>

        {/* Separator */}
        <div className="h-6 w-px bg-gray-800"></div>

        {/* Indicators */}
        <button
          onClick={() => setShowIndicators(!showIndicators)}
          className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded transition-all ${
            showIndicators
              ? 'bg-gray-800 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-800'
          }`}
        >
          <span>📊</span>
          <span>Indicators</span>
          {selectedIndicators.length > 0 && (
            <span className="px-1.5 py-0.5 text-[10px] bg-[#1a73e8] text-white rounded">{selectedIndicators.length}</span>
          )}
        </button>

        {/* Drawing Tools */}
        <button
          onClick={() => setShowDrawingTools(!showDrawingTools)}
          className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded transition-all ${
            showDrawingTools
              ? 'bg-gray-800 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-800'
          }`}
        >
          <span>✏️</span>
          <span>Draw</span>
        </button>

        {currentTool && (
          <>
            <div className="h-6 w-px bg-gray-800"></div>
            <div className="flex items-center gap-2">
              <div className="text-xs text-gray-400">
                Active: <span className="text-white">{drawingTools.find(t => t.value === currentTool)?.label}</span>
              </div>
              <button
                onClick={() => setCurrentTool('')}
                className="px-2 py-0.5 text-[10px] text-gray-400 hover:text-white bg-gray-800 rounded"
              >
                Cancel
              </button>
            </div>
          </>
        )}
      </div>

      {/* Indicators Panel */}
      {showIndicators && (
        <div className="px-4 py-3 bg-[#0a0a0a] border-b border-gray-800">
          <div className="grid grid-cols-4 gap-2">
            {indicators.map((ind) => {
              const isActive = selectedIndicators.includes(ind.value)
              return (
                <button
                  key={ind.value}
                  onClick={() => toggleIndicator(ind.value)}
                  className={`px-3 py-2 text-left rounded transition-all ${
                    isActive
                      ? 'bg-[#1a73e8] text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <div className="text-xs font-medium">{ind.label}</div>
                  <div className="text-[10px] opacity-80">{ind.name}</div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Drawing Tools Panel */}
      {showDrawingTools && (
        <div className="px-4 py-3 bg-[#0a0a0a] border-b border-gray-800">
          <div className="flex items-center gap-2 flex-wrap">
            {drawingTools.map((tool) => (
              <button
                key={tool.value}
                onClick={() => handleDrawingTool(tool.value)}
                className={`flex items-center gap-2 px-3 py-2 rounded transition-all ${
                  currentTool === tool.value
                    ? 'bg-[#1a73e8] text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
                }`}
                title={tool.label}
              >
                <span className="text-base">{tool.icon}</span>
                <span className="text-xs">{tool.label}</span>
              </button>
            ))}
            <div className="h-6 w-px bg-gray-700 mx-1"></div>
            <button
              onClick={clearDrawings}
              className="px-3 py-2 text-xs text-gray-400 hover:text-white bg-gray-800 hover:bg-gray-700 rounded transition-all"
            >
              🗑️ Clear All
            </button>
          </div>
        </div>
      )}

      {/* Chart */}
      <div ref={chartRef} style={{ height: `${height}px`, width: '100%', backgroundColor: '#0f0f0f' }} />

      {/* Price Lines Legend */}
      {priceLines && Object.values(priceLines).some(v => v !== undefined) && (
        <div className="px-4 py-2.5 bg-[#0a0a0a] border-t border-gray-800">
          <div className="flex items-center gap-4 text-xs">
            {priceLines.entry && (
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-blue-500"></div>
                <span className="text-gray-400">Entry:</span>
                <span className="text-white font-medium">${priceLines.entry.toFixed(2)}</span>
              </div>
            )}
            {priceLines.takeProfit && (
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-green-500"></div>
                <span className="text-gray-400">TP:</span>
                <span className="text-green-400 font-medium">${priceLines.takeProfit.toFixed(2)}</span>
              </div>
            )}
            {priceLines.stopLoss && (
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                <span className="text-gray-400">SL:</span>
                <span className="text-red-400 font-medium">${priceLines.stopLoss.toFixed(2)}</span>
              </div>
            )}
            {priceLines.breakEven && (
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                <span className="text-gray-400">BE:</span>
                <span className="text-yellow-400 font-medium">${priceLines.breakEven.toFixed(2)}</span>
              </div>
            )}
            {priceLines.liquidation && (
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-purple-500"></div>
                <span className="text-gray-400">Liq:</span>
                <span className="text-purple-400 font-medium">${priceLines.liquidation.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
