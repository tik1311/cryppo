'use client'

import { useEffect, useRef, memo } from 'react'

interface TradingViewChartProps {
  symbol: string
  height?: number
  theme?: 'light' | 'dark'
  interval?: string
  priceLines?: {
    entry?: number
    takeProfit?: number
    stopLoss?: number
    breakEven?: number
    liquidation?: number
  }
}

function TradingViewChart({
  symbol,
  height = 600,
  theme = 'dark',
  interval = '1',
  priceLines
}: TradingViewChartProps) {
  const container = useRef<HTMLDivElement>(null)
  const scriptRef = useRef<HTMLScriptElement | null>(null)

  useEffect(() => {
    if (!container.current) return

    // Clear previous widget
    container.current.innerHTML = ''

    // Create widget container
    const widgetContainer = document.createElement('div')
    widgetContainer.className = 'tradingview-widget-container'
    widgetContainer.style.height = `${height}px`
    widgetContainer.style.width = '100%'

    const widgetDiv = document.createElement('div')
    widgetDiv.className = 'tradingview-widget-container__widget'
    widgetDiv.style.height = 'calc(100% - 32px)'
    widgetDiv.style.width = '100%'

    widgetContainer.appendChild(widgetDiv)
    container.current.appendChild(widgetContainer)

    // TradingView symbol format conversion
    // From "BTCUSDT" to "BINANCE:BTCUSDT"
    const tvSymbol = `BINANCE:${symbol.replace('/', '')}`

    // Configure widget
    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'
    script.type = 'text/javascript'
    script.async = true

    const config = {
      autosize: true,
      symbol: tvSymbol,
      interval: interval,
      timezone: 'Etc/UTC',
      theme: theme,
      style: '1', // Candlestick
      locale: 'en',
      enable_publishing: false,
      allow_symbol_change: false,
      save_image: true,
      calendar: false,
      support_host: 'https://www.tradingview.com',
      // Advanced features - Enable drawing tools
      drawings_access: { type: 'black', tools: [{ name: 'Regression Trend' }] },
      enabled_features: ['study_templates', 'use_localstorage_for_settings'],
      disabled_features: ['header_symbol_search', 'symbol_search_hot_key'],
      // Drawing tools enabled
      studies_overrides: {},
      overrides: {
        'mainSeriesProperties.candleStyle.upColor': '#26a69a',
        'mainSeriesProperties.candleStyle.downColor': '#ef5350',
        'mainSeriesProperties.candleStyle.borderUpColor': '#26a69a',
        'mainSeriesProperties.candleStyle.borderDownColor': '#ef5350',
        'mainSeriesProperties.candleStyle.wickUpColor': '#26a69a',
        'mainSeriesProperties.candleStyle.wickDownColor': '#ef5350'
      },
      show_popup_button: true,
      popup_width: '1000',
      popup_height: '650',
      container_id: 'tradingview_widget'
    }

    script.innerHTML = JSON.stringify(config)
    widgetContainer.appendChild(script)
    scriptRef.current = script

    return () => {
      if (container.current) {
        container.current.innerHTML = ''
      }
    }
  }, [symbol, height, theme, interval])

  return (
    <div className="card p-0 overflow-hidden">
      <div
        ref={container}
        style={{ height: `${height}px`, width: '100%' }}
      />
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
      <div className="px-4 py-2 bg-gray-100 dark:bg-gray-900 text-xs text-center text-gray-500 dark:text-gray-400">
        Powered by{' '}
        <a
          href="https://www.tradingview.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary hover:underline"
        >
          TradingView
        </a>
      </div>
    </div>
  )
}

export default memo(TradingViewChart)
