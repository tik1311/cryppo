'use client'

import { useEffect, useRef } from 'react'
import { createChart, ColorType, IChartApi, ISeriesApi, Time } from 'lightweight-charts'

interface CandleData {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface PriceLine {
  price: number
  color: string
  lineWidth: number
  lineStyle: number  // 0=Solid, 1=Dotted, 2=Dashed, 3=LargeDashed
  title: string
}

interface TradingChartProps {
  data: CandleData[]
  height?: number
  symbol?: string
  priceLines?: {
    entry?: number
    takeProfit?: number
    stopLoss?: number
    breakEven?: number
    liquidation?: number
  }
}

export default function TradingChart({
  data,
  height = 400,
  symbol = 'BTC/USDT',
  priceLines
}: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#1F2937',
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      grid: {
        vertLines: { color: '#E5E7EB' },
        horzLines: { color: '#E5E7EB' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#D1D5DB',
      },
      timeScale: {
        borderColor: '#D1D5DB',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    chartRef.current = chart

    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#10b981',
      wickDownColor: '#ef4444',
      wickUpColor: '#10b981',
    })

    candleSeriesRef.current = candleSeries

    // Set data
    candleSeries.setData(data)

    // Fit content
    chart.timeScale().fitContent()

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [data, height])

  // Update chart with new data
  useEffect(() => {
    if (candleSeriesRef.current && data.length > 0) {
      candleSeriesRef.current.setData(data)
      chartRef.current?.timeScale().fitContent()
    }
  }, [data])

  // Add price lines when position changes
  useEffect(() => {
    if (!candleSeriesRef.current || !priceLines) return

    // Remove existing price lines
    candleSeriesRef.current.createPriceLine

    // Add Entry Price line
    if (priceLines.entry) {
      candleSeriesRef.current.createPriceLine({
        price: priceLines.entry,
        color: '#3B82F6', // Blue
        lineWidth: 2,
        lineStyle: 0, // Solid
        axisLabelVisible: true,
        title: 'Entry',
      })
    }

    // Add Take Profit line
    if (priceLines.takeProfit) {
      candleSeriesRef.current.createPriceLine({
        price: priceLines.takeProfit,
        color: '#10B981', // Green
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: 'TP',
      })
    }

    // Add Stop Loss line
    if (priceLines.stopLoss) {
      candleSeriesRef.current.createPriceLine({
        price: priceLines.stopLoss,
        color: '#EF4444', // Red
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: 'SL',
      })
    }

    // Add Break Even line
    if (priceLines.breakEven) {
      candleSeriesRef.current.createPriceLine({
        price: priceLines.breakEven,
        color: '#F59E0B', // Amber
        lineWidth: 1,
        lineStyle: 1, // Dotted
        axisLabelVisible: true,
        title: 'BE',
      })
    }

    // Add Liquidation line
    if (priceLines.liquidation) {
      candleSeriesRef.current.createPriceLine({
        price: priceLines.liquidation,
        color: '#DC2626', // Dark Red
        lineWidth: 3,
        lineStyle: 0, // Solid
        axisLabelVisible: true,
        title: 'Liquidation',
      })
    }
  }, [priceLines])

  return (
    <div className="w-full">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          {symbol} - 1H Timeframe
        </h3>
        <div className="flex items-center gap-4">
          {priceLines && (
            <div className="flex items-center gap-3 text-xs">
              {priceLines.entry && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-blue-500"></div>
                  <span>Entry</span>
                </div>
              )}
              {priceLines.takeProfit && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-green-500 border-dashed border-t-2"></div>
                  <span>TP</span>
                </div>
              )}
              {priceLines.stopLoss && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-red-500 border-dashed border-t-2"></div>
                  <span>SL</span>
                </div>
              )}
              {priceLines.liquidation && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-red-600"></div>
                  <span>Liq</span>
                </div>
              )}
            </div>
          )}
          <div className="text-sm text-gray-500">
            {data.length > 0 ? `${data.length} candles` : 'No data'}
          </div>
        </div>
      </div>
      <div
        ref={chartContainerRef}
        className="rounded-lg border border-gray-200 bg-white shadow-sm"
      />
    </div>
  )
}
