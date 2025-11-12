'use client'

import { useEffect, useRef, useState } from 'react'
import { init, dispose } from '@klinecharts/pro'
import type { Chart } from '@klinecharts/pro'

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
  priceLines
}: KLineChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<Chart | null>(null)
  const [currentTool, setCurrentTool] = useState<string>('')

  // Convert data format for KLineChart
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

  useEffect(() => {
    if (!chartRef.current || rawData.length === 0) return

    // Initialize chart
    const chartData = convertData(rawData)

    chartInstanceRef.current = init(chartRef.current, {
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
            color: '#393939'
          },
          vertical: {
            color: '#393939'
          }
        }
      },
      customApi: {
        formatDate: (timestamp: number) => {
          const date = new Date(timestamp)
          return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          })
        }
      }
    })

    // Load data
    chartInstanceRef.current.applyNewData(chartData)

    return () => {
      if (chartInstanceRef.current) {
        dispose(chartRef.current!)
        chartInstanceRef.current = null
      }
    }
  }, [rawData])

  // Update data when new candles arrive
  useEffect(() => {
    if (chartInstanceRef.current && rawData.length > 0) {
      const latestCandle = convertData([rawData[rawData.length - 1]])[0]
      chartInstanceRef.current.updateData(latestCandle)
    }
  }, [rawData])

  // Draw price lines as overlays
  useEffect(() => {
    if (!chartInstanceRef.current || !priceLines) return

    const pane = chartInstanceRef.current.getChart()
    if (!pane) return

    // Clear previous overlays
    pane.removeOverlay()

    // Entry Price Line
    if (priceLines.entry) {
      pane.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.entry }],
        styles: {
          line: {
            color: '#3b82f6',
            size: 2
          },
          text: {
            color: '#3b82f6'
          }
        },
        extendData: { text: 'Entry' }
      })
    }

    // Take Profit Line
    if (priceLines.takeProfit) {
      pane.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.takeProfit }],
        styles: {
          line: {
            color: '#22c55e',
            size: 2
          },
          text: {
            color: '#22c55e'
          }
        },
        extendData: { text: 'TP' }
      })
    }

    // Stop Loss Line
    if (priceLines.stopLoss) {
      pane.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.stopLoss }],
        styles: {
          line: {
            color: '#ef4444',
            size: 2
          },
          text: {
            color: '#ef4444'
          }
        },
        extendData: { text: 'SL' }
      })
    }

    // Break Even Line
    if (priceLines.breakEven) {
      pane.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.breakEven }],
        styles: {
          line: {
            color: '#eab308',
            size: 1,
            style: 'dashed'
          },
          text: {
            color: '#eab308'
          }
        },
        extendData: { text: 'BE' }
      })
    }

    // Liquidation Line
    if (priceLines.liquidation) {
      pane.createOverlay({
        name: 'priceLine',
        points: [{ value: priceLines.liquidation }],
        styles: {
          line: {
            color: '#a855f7',
            size: 2,
            style: 'dashed'
          },
          text: {
            color: '#a855f7'
          }
        },
        extendData: { text: 'Liq' }
      })
    }
  }, [priceLines])

  const handleDrawingTool = (toolName: string) => {
    if (!chartInstanceRef.current) return

    const pane = chartInstanceRef.current.getChart()
    if (!pane) return

    if (currentTool === toolName) {
      // Disable tool
      setCurrentTool('')
      return
    }

    // Enable tool
    setCurrentTool(toolName)
    pane.createOverlay({
      name: toolName,
      lock: false
    })
  }

  const clearDrawings = () => {
    if (!chartInstanceRef.current) return
    const pane = chartInstanceRef.current.getChart()
    if (!pane) return

    // Remove all overlays except price lines
    pane.removeOverlay({ name: 'straightLine' })
    pane.removeOverlay({ name: 'segment' })
    pane.removeOverlay({ name: 'fibonacciLine' })
    pane.removeOverlay({ name: 'parallelStraightLine' })

    setCurrentTool('')
  }

  return (
    <div className="card p-0 overflow-hidden">
      {/* Drawing Tools Toolbar */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="text-sm font-semibold text-gray-700 dark:text-gray-300 mr-4">
            Drawing Tools:
          </div>
          <button
            onClick={() => handleDrawingTool('segment')}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              currentTool === 'segment'
                ? 'bg-primary text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            📐 Trendline
          </button>
          <button
            onClick={() => handleDrawingTool('fibonacciLine')}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              currentTool === 'fibonacciLine'
                ? 'bg-primary text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            📊 Fibonacci
          </button>
          <button
            onClick={() => handleDrawingTool('parallelStraightLine')}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              currentTool === 'parallelStraightLine'
                ? 'bg-primary text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            ⫽ Channel
          </button>
          <button
            onClick={clearDrawings}
            className="px-3 py-1.5 rounded text-sm font-medium bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-800 transition-colors ml-2"
          >
            🗑️ Clear
          </button>
          <div className="text-xs text-gray-500 dark:text-gray-400 ml-auto">
            {currentTool ? 'Click and drag on chart to draw' : 'Select a tool to start drawing'}
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div ref={chartRef} style={{ height: `${height}px`, width: '100%' }} />

      {/* Price Lines Legend */}
      {priceLines && (
        <div className="p-4 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
            {priceLines.entry && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <div>
                  <div className="text-gray-500 dark:text-gray-400">Entry</div>
                  <div className="font-semibold">${priceLines.entry.toFixed(2)}</div>
                </div>
              </div>
            )}
            {priceLines.takeProfit && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <div>
                  <div className="text-gray-500 dark:text-gray-400">Take Profit</div>
                  <div className="font-semibold text-green-600 dark:text-green-400">
                    ${priceLines.takeProfit.toFixed(2)}
                  </div>
                </div>
              </div>
            )}
            {priceLines.stopLoss && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div>
                  <div className="text-gray-500 dark:text-gray-400">Stop Loss</div>
                  <div className="font-semibold text-red-600 dark:text-red-400">
                    ${priceLines.stopLoss.toFixed(2)}
                  </div>
                </div>
              </div>
            )}
            {priceLines.breakEven && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div>
                  <div className="text-gray-500 dark:text-gray-400">Break Even</div>
                  <div className="font-semibold text-yellow-600 dark:text-yellow-400">
                    ${priceLines.breakEven.toFixed(2)}
                  </div>
                </div>
              </div>
            )}
            {priceLines.liquidation && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                <div>
                  <div className="text-gray-500 dark:text-gray-400">Liquidation</div>
                  <div className="font-semibold text-purple-600 dark:text-purple-400">
                    ${priceLines.liquidation.toFixed(2)}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
