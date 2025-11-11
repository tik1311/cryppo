'use client'

import { useEffect, useRef, useState } from 'react'
import {
  ChartCanvas,
  Chart,
  CandlestickSeries,
  XAxis,
  YAxis,
  CrossHairCursor,
  MouseCoordinateX,
  MouseCoordinateY,
  TrendLine,
  FibonacciRetracement,
  drawingObjectSelector,
  EdgeIndicator,
  OHLCTooltip,
  ZoomButtons,
  withDeviceRatio,
  withSize
} from 'react-financial-charts'
import { format } from 'd3-format'
import { timeFormat } from 'd3-time-format'
import { scaleTime } from 'd3-scale'

interface CandleData {
  date: Date
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface FinancialChartProps {
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

function FinancialChartInner({
  data: rawData,
  symbol,
  height = 650,
  priceLines,
  width,
  ratio
}: FinancialChartProps & { width: number; ratio: number }) {
  const [trends, setTrends] = useState<any[]>([])
  const [fibonaccis, setFibonaccis] = useState<any[]>([])
  const [enableTrendLine, setEnableTrendLine] = useState(false)
  const [enableFibonacci, setEnableFibonacci] = useState(false)

  // Convert data to required format
  const data: CandleData[] = rawData.map(d => ({
    date: new Date((typeof d.time === 'number' ? d.time : parseInt(d.time)) * 1000),
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close,
    volume: d.volume
  })).sort((a, b) => a.date.getTime() - b.date.getTime())

  if (!data || data.length === 0) {
    return (
      <div className="card p-8 text-center text-gray-500">
        Loading chart data...
      </div>
    )
  }

  const margin = { left: 50, right: 70, top: 20, bottom: 30 }
  const gridHeight = height - margin.top - margin.bottom
  const gridWidth = width - margin.left - margin.right

  const xAccessor = (d: CandleData) => d.date
  const xExtents = [
    xAccessor(data[Math.max(0, data.length - 150)]),
    xAccessor(data[data.length - 1])
  ]

  const xScale = scaleTime()
  const yExtents = (d: CandleData) => [d.high, d.low]

  const handleTrendLineComplete = (newTrend: any) => {
    setTrends([...trends, newTrend])
    setEnableTrendLine(false)
  }

  const handleFibonacciComplete = (newFib: any) => {
    setFibonaccis([...fibonaccis, newFib])
    setEnableFibonacci(false)
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
            onClick={() => setEnableTrendLine(!enableTrendLine)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              enableTrendLine
                ? 'bg-primary text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            📐 Trendline
          </button>
          <button
            onClick={() => setEnableFibonacci(!enableFibonacci)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              enableFibonacci
                ? 'bg-primary text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            📊 Fibonacci
          </button>
          <button
            onClick={() => {
              setTrends([])
              setFibonaccis([])
            }}
            className="px-3 py-1.5 rounded text-sm font-medium bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-800 transition-colors ml-2"
          >
            🗑️ Clear All
          </button>
          <div className="text-xs text-gray-500 dark:text-gray-400 ml-auto">
            Click tool, then drag on chart to draw
          </div>
        </div>
      </div>

      {/* Chart */}
      <div style={{ position: 'relative' }}>
        <ChartCanvas
          height={height}
          width={width}
          ratio={ratio}
          margin={margin}
          data={data}
          xScale={xScale}
          xAccessor={xAccessor}
          xExtents={xExtents}
          seriesName={symbol}
        >
          <Chart id={1} yExtents={yExtents}>
            <XAxis />
            <YAxis />

            {/* Candlestick */}
            <CandlestickSeries
              fill={(d: CandleData) => (d.close > d.open ? '#26a69a' : '#ef5350')}
              wickStroke={(d: CandleData) => (d.close > d.open ? '#26a69a' : '#ef5350')}
            />

            {/* Price Lines */}
            {priceLines?.entry && (
              <TrendLine
                type="LINE"
                snap={false}
                enabled={false}
                stroke="#3b82f6"
                strokeWidth={2}
                strokeOpacity={0.8}
                strokeDasharray="Solid"
              />
            )}

            {/* Tooltip */}
            <OHLCTooltip origin={[8, 16]} />

            {/* Trends */}
            {trends.map((trend, i) => (
              <TrendLine
                key={i}
                {...trend}
                enabled={false}
                onComplete={handleTrendLineComplete}
              />
            ))}

            {enableTrendLine && (
              <TrendLine
                type="LINE"
                snap={false}
                enabled={true}
                onComplete={handleTrendLineComplete}
                stroke="#fbbf24"
                strokeWidth={2}
              />
            )}

            {/* Fibonacci */}
            {fibonaccis.map((fib, i) => (
              <FibonacciRetracement
                key={i}
                {...fib}
                enabled={false}
              />
            ))}

            {enableFibonacci && (
              <FibonacciRetracement
                enabled={true}
                onComplete={handleFibonacciComplete}
              />
            )}

            <EdgeIndicator
              itemType="last"
              rectWidth={margin.right}
              fill={(d: CandleData) => (d.close > d.open ? '#26a69a' : '#ef5350')}
              lineStroke={(d: CandleData) => (d.close > d.open ? '#26a69a' : '#ef5350')}
              displayFormat={format('.2f')}
              yAccessor={(d: CandleData) => d.close}
            />

            <MouseCoordinateX displayFormat={timeFormat('%Y-%m-%d %H:%M')} />
            <MouseCoordinateY rectWidth={margin.right} displayFormat={format('.2f')} />
          </Chart>

          <CrossHairCursor />
          <ZoomButtons />
        </ChartCanvas>
      </div>

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

const FinancialChart = withSize({ style: { minHeight: 400 } })(
  withDeviceRatio()(FinancialChartInner)
)

export default FinancialChart
