'use client'

import { useEffect, useRef, useState } from 'react'
import { init, dispose } from 'klinecharts'
import type { KLineData } from 'klinecharts'

interface CandlestickChartProps {
  data: KLineData[]
  height?: number
  showVolume?: boolean
  showIndicators?: boolean
  symbol?: string
}

export default function CandlestickChart({
  data,
  height = 500,
  showVolume = true,
  showIndicators = true,
  symbol = 'BTC/USDT'
}: CandlestickChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<any>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!chartRef.current || data.length === 0) {
      setIsLoading(false)
      return
    }

    // Initialize chart
    const chart = init(chartRef.current)
    chartInstanceRef.current = chart

    // Configure chart
    chart.setStyles({
      grid: {
        show: true,
        horizontal: {
          show: true,
          color: '#E5E7EB',
          style: 'dashed'
        },
        vertical: {
          show: true,
          color: '#E5E7EB',
          style: 'dashed'
        }
      },
      candle: {
        type: 'candle_solid',
        bar: {
          upColor: '#10b981',
          downColor: '#ef4444',
          upBorderColor: '#10b981',
          downBorderColor: '#ef4444',
          upWickColor: '#10b981',
          downWickColor: '#ef4444'
        },
        tooltip: {
          showRule: 'always',
          showType: 'standard',
          labels: ['T: ', 'O: ', 'C: ', 'H: ', 'L: ', 'V: '],
          values: null,
          defaultValue: 'n/a',
          rect: {
            position: 'fixed',
            paddingLeft: 8,
            paddingRight: 8,
            paddingTop: 8,
            paddingBottom: 8,
            offsetLeft: 12,
            offsetTop: 12,
            offsetRight: 12,
            borderRadius: 4,
            borderSize: 1,
            borderColor: '#D1D5DB',
            color: '#FAFAFA'
          },
          text: {
            size: 12,
            family: 'Inter, sans-serif',
            weight: 'normal',
            color: '#1F2937',
            marginLeft: 8,
            marginTop: 6,
            marginRight: 8,
            marginBottom: 6
          }
        }
      },
      indicator: {
        tooltip: {
          showRule: 'always',
          showType: 'standard',
          showName: true,
          showParams: true,
          defaultValue: 'n/a',
          text: {
            size: 12,
            family: 'Inter, sans-serif',
            weight: 'normal',
            color: '#1F2937',
            marginTop: 6,
            marginRight: 8,
            marginBottom: 6,
            marginLeft: 8
          }
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
            color: '#6B7280'
          },
          text: {
            show: true,
            color: '#FFFFFF',
            size: 12,
            family: 'Inter, sans-serif',
            weight: 'normal',
            borderSize: 1,
            borderColor: '#6B7280',
            borderRadius: 2,
            paddingLeft: 4,
            paddingRight: 4,
            paddingTop: 4,
            paddingBottom: 4,
            backgroundColor: '#6B7280'
          }
        },
        vertical: {
          show: true,
          line: {
            show: true,
            style: 'dashed',
            dashValue: [4, 2],
            size: 1,
            color: '#6B7280'
          },
          text: {
            show: true,
            color: '#FFFFFF',
            size: 12,
            family: 'Inter, sans-serif',
            weight: 'normal',
            borderSize: 1,
            borderColor: '#6B7280',
            borderRadius: 2,
            paddingLeft: 4,
            paddingRight: 4,
            paddingTop: 4,
            paddingBottom: 4,
            backgroundColor: '#6B7280'
          }
        }
      }
    })

    // Load data
    chart.applyNewData(data)

    // Add volume indicator if enabled
    if (showVolume) {
      chart.createIndicator('VOL', false, { id: 'candle_pane' })
    }

    // Add technical indicators if enabled
    if (showIndicators) {
      chart.createIndicator('MA', true, { id: 'candle_pane' })
      chart.createIndicator('RSI', false)
      chart.createIndicator('MACD', false)
    }

    setIsLoading(false)

    // Cleanup
    return () => {
      if (chartInstanceRef.current) {
        dispose(chartRef.current!)
        chartInstanceRef.current = null
      }
    }
  }, [data, showVolume, showIndicators])

  // Update chart when new data arrives
  useEffect(() => {
    if (chartInstanceRef.current && data.length > 0) {
      const lastCandle = data[data.length - 1]
      chartInstanceRef.current.updateData(lastCandle)
    }
  }, [data])

  return (
    <div className="w-full">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          {symbol} - 1H Timeframe
        </h3>
        {isLoading && (
          <span className="text-sm text-gray-500">Loading chart...</span>
        )}
      </div>
      <div
        ref={chartRef}
        className="w-full rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800"
        style={{ height: `${height}px` }}
      />
      {data.length === 0 && !isLoading && (
        <div className="flex h-64 items-center justify-center text-gray-500">
          No data available
        </div>
      )}
    </div>
  )
}
