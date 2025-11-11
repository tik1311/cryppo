'use client'

interface Trade {
  id?: string
  timestamp: string | number
  action: 'BUY' | 'SELL' | 'HOLD'
  price: number
  quantity?: number
  value?: number
  pnl?: number
  balance?: number
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

  const getActionBadge = (action: string) => {
    const colors = {
      BUY: 'bg-success/10 text-success border-success',
      SELL: 'bg-danger/10 text-danger border-danger',
      HOLD: 'bg-gray-100 text-gray-600 border-gray-300'
    }
    return (
      <span className={`inline-flex items-center rounded border px-2 py-1 text-xs font-semibold ${colors[action as keyof typeof colors] || colors.HOLD}`}>
        {action}
      </span>
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
                <th className="pb-3 text-right">Price</th>
                <th className="pb-3 text-right">Value</th>
                {showPnL && <th className="pb-3 text-right">P/L</th>}
                <th className="pb-3 text-right">Balance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {displayTrades.map((trade, index) => (
                <tr key={trade.id || index} className="text-sm hover:bg-gray-50">
                  <td className="py-3 text-gray-600">
                    {formatTimestamp(trade.timestamp)}
                  </td>
                  <td className="py-3">
                    {getActionBadge(trade.action)}
                  </td>
                  <td className="py-3 text-right font-mono text-gray-900">
                    {formatCurrency(trade.price)}
                  </td>
                  <td className="py-3 text-right font-mono text-gray-900">
                    {formatCurrency(trade.value)}
                  </td>
                  {showPnL && (
                    <td className={`py-3 text-right font-mono font-semibold ${getPnLColor(trade.pnl)}`}>
                      {trade.pnl !== undefined && trade.pnl !== null
                        ? `${trade.pnl >= 0 ? '+' : ''}${formatCurrency(trade.pnl)}`
                        : '-'}
                    </td>
                  )}
                  <td className="py-3 text-right font-mono text-gray-900">
                    {formatCurrency(trade.balance)}
                  </td>
                </tr>
              ))}
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
