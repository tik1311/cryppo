'use client'

interface Trade {
  id?: string
  timestamp: string | number
  action: 'OPEN_LONG' | 'OPEN_SHORT' | 'CLOSE_LONG' | 'CLOSE_SHORT' | 'HOLD' | 'BUY' | 'SELL'
  position?: 'LONG' | 'SHORT'
  entry_price?: number
  exit_price?: number | null
  current_price?: number
  price?: number  // Legacy support
  quantity?: number
  leverage?: number
  liquidation_price?: number
  take_profit?: number
  stop_loss?: number
  break_even?: number
  value?: number
  pnl?: number
  pnl_pct?: number
  balance?: number
  status?: 'OPEN' | 'CLOSED'
  reason?: string
}

interface TradesListProps {
  trades: Trade[]
  maxItems?: number
  showPnL?: boolean
}

export default function TradesList({ trades, maxItems = 20, showPnL = true }: TradesListProps) {
  const displayTrades = trades.slice(-maxItems).reverse()

  const formatTimestamp = (timestamp: string | number) => {
    const date = new Date(typeof timestamp === 'string' ? timestamp : timestamp * 1000)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })
  }

  const formatCurrency = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A'
    return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const getActionBadge = (action: string, position?: string) => {
    const colors = {
      OPEN_LONG: 'bg-green-100 text-green-700 border-green-300',
      OPEN_SHORT: 'bg-red-100 text-red-700 border-red-300',
      CLOSE_LONG: 'bg-blue-100 text-blue-700 border-blue-300',
      CLOSE_SHORT: 'bg-purple-100 text-purple-700 border-purple-300',
      BUY: 'bg-success/10 text-success border-success',  // Legacy
      SELL: 'bg-danger/10 text-danger border-danger',    // Legacy
      HOLD: 'bg-gray-100 text-gray-600 border-gray-300'
    }

    const displayText = action.replace('_', ' ')
    const color = colors[action as keyof typeof colors] || colors.HOLD

    return (
      <div className="flex flex-col gap-1">
        <span className={`inline-flex items-center rounded border px-2 py-1 text-xs font-semibold ${color}`}>
          {displayText}
        </span>
        {position && (
          <span className="text-xs text-gray-500">
            {position}
          </span>
        )}
      </div>
    )
  }

  const getPnLColor = (pnl: number | undefined) => {
    if (pnl === undefined || pnl === null) return 'text-gray-600'
    return pnl >= 0 ? 'text-success' : 'text-danger'
  }

  return (
    <div className="card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Recent Trades</h3>
        <span className="text-sm text-gray-500">
          {displayTrades.length} / {trades.length} trades
        </span>
      </div>

      {displayTrades.length === 0 ? (
        <div className="py-8 text-center text-gray-500">
          No trades yet
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 text-left text-xs font-semibold uppercase text-gray-600">
                <th className="pb-3">Time</th>
                <th className="pb-3">Action</th>
                <th className="pb-3 text-right">Entry</th>
                <th className="pb-3 text-right">Exit/Current</th>
                <th className="pb-3 text-right">Leverage</th>
                <th className="pb-3 text-right">Liq Price</th>
                {showPnL && <th className="pb-3 text-right">P/L</th>}
                {showPnL && <th className="pb-3 text-right">P/L %</th>}
                <th className="pb-3 text-right">Balance</th>
                <th className="pb-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {displayTrades.map((trade, index) => {
                const displayPrice = trade.entry_price || trade.price || 0
                const currentPrice = trade.exit_price || trade.current_price || displayPrice

                return (
                  <tr key={trade.id || index} className="text-sm hover:bg-gray-50">
                    <td className="py-3 text-gray-600">
                      {formatTimestamp(trade.timestamp)}
                    </td>
                    <td className="py-3">
                      {getActionBadge(trade.action, trade.position)}
                    </td>
                    <td className="py-3 text-right font-mono text-gray-900">
                      {formatCurrency(displayPrice)}
                    </td>
                    <td className="py-3 text-right font-mono text-gray-900">
                      <div className="flex flex-col items-end">
                        <span>{formatCurrency(currentPrice)}</span>
                        {trade.take_profit && (
                          <span className="text-xs text-green-600" title="Take Profit">
                            TP: {formatCurrency(trade.take_profit)}
                          </span>
                        )}
                        {trade.stop_loss && (
                          <span className="text-xs text-red-600" title="Stop Loss">
                            SL: {formatCurrency(trade.stop_loss)}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 text-right font-mono text-purple-600 font-semibold">
                      {trade.leverage ? `${trade.leverage}x` : '-'}
                    </td>
                    <td className="py-3 text-right font-mono text-orange-600">
                      {trade.liquidation_price ? formatCurrency(trade.liquidation_price) : '-'}
                    </td>
                    {showPnL && (
                      <td className={`py-3 text-right font-mono font-semibold ${getPnLColor(trade.pnl)}`}>
                        {trade.pnl !== undefined && trade.pnl !== null
                          ? `${trade.pnl >= 0 ? '+' : ''}${formatCurrency(trade.pnl)}`
                          : '-'}
                      </td>
                    )}
                    {showPnL && (
                      <td className={`py-3 text-right font-mono font-semibold ${getPnLColor(trade.pnl)}`}>
                        {trade.pnl_pct !== undefined && trade.pnl_pct !== null
                          ? `${trade.pnl_pct >= 0 ? '+' : ''}${trade.pnl_pct.toFixed(2)}%`
                          : '-'}
                      </td>
                    )}
                    <td className="py-3 text-right font-mono text-gray-900">
                      {formatCurrency(trade.balance)}
                    </td>
                    <td className="py-3">
                      {trade.status && (
                        <span className={`inline-flex items-center rounded px-2 py-1 text-xs font-semibold ${
                          trade.status === 'OPEN'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-gray-100 text-gray-600'
                        }`}>
                          {trade.status}
                        </span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {trades.length > maxItems && (
        <div className="mt-4 text-center text-xs text-gray-500">
          Showing last {maxItems} trades. Total: {trades.length} trades.
        </div>
      )}
    </div>
  )
}
