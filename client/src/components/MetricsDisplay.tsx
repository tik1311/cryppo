'use client'

interface Metrics {
  balance?: number
  initial_balance?: number
  total_profit?: number
  total_return?: number
  win_rate?: number
  total_trades?: number
  winning_trades?: number
  losing_trades?: number
  max_drawdown?: number
  sharpe_ratio?: number
  current_position?: 'LONG' | 'SHORT' | 'NEUTRAL'
  position_size?: number
  unrealized_pnl?: number
}

interface MetricsDisplayProps {
  metrics: Metrics
  isLive?: boolean
}

export default function MetricsDisplay({ metrics, isLive = false }: MetricsDisplayProps) {
  const profitColor = (metrics.total_profit ?? 0) >= 0 ? 'text-success' : 'text-danger'
  const returnColor = (metrics.total_return ?? 0) >= 0 ? 'text-success' : 'text-danger'

  const formatCurrency = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A'
    return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const formatPercent = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A'
    const sign = value >= 0 ? '+' : ''
    return `${sign}${(value * 100).toFixed(2)}%`
  }

  const formatNumber = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A'
    return value.toLocaleString('en-US')
  }

  const getPositionBadge = () => {
    const position = metrics.current_position || 'NEUTRAL'
    const colors = {
      LONG: 'bg-success/10 text-success border-success',
      SHORT: 'bg-danger/10 text-danger border-danger',
      NEUTRAL: 'bg-gray-100 text-gray-700 border-gray-300'
    }
    return (
      <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${colors[position]}`}>
        {position}
      </span>
    )
  }

  return (
    <div className="space-y-4">
      {/* Live Indicator */}
      {isLive && (
        <div className="flex items-center gap-2 rounded-lg bg-primary-50 p-3">
          <div className="h-3 w-3 animate-pulse rounded-full bg-primary-500"></div>
          <span className="text-sm font-semibold text-primary-700">Live Trading Active</span>
        </div>
      )}

      {/* Main Metrics Grid */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* Balance */}
        <div className="card">
          <div className="text-sm text-gray-500">Current Balance</div>
          <div className="mt-1 text-2xl font-bold text-gray-900">
            {formatCurrency(metrics.balance)}
          </div>
          <div className="mt-1 text-xs text-gray-400">
            Initial: {formatCurrency(metrics.initial_balance)}
          </div>
        </div>

        {/* Total Profit */}
        <div className="card">
          <div className="text-sm text-gray-500">Total Profit/Loss</div>
          <div className={`mt-1 text-2xl font-bold ${profitColor}`}>
            {formatCurrency(metrics.total_profit)}
          </div>
          <div className={`mt-1 text-xs font-semibold ${returnColor}`}>
            {formatPercent(metrics.total_return)}
          </div>
        </div>

        {/* Win Rate */}
        <div className="card">
          <div className="text-sm text-gray-500">Win Rate</div>
          <div className="mt-1 text-2xl font-bold text-gray-900">
            {formatPercent(metrics.win_rate)}
          </div>
          <div className="mt-1 text-xs text-gray-400">
            {formatNumber(metrics.winning_trades)} / {formatNumber(metrics.total_trades)} trades
          </div>
        </div>

        {/* Sharpe Ratio */}
        <div className="card">
          <div className="text-sm text-gray-500">Sharpe Ratio</div>
          <div className="mt-1 text-2xl font-bold text-gray-900">
            {metrics.sharpe_ratio?.toFixed(3) ?? 'N/A'}
          </div>
          <div className="mt-1 text-xs text-gray-400">
            Max DD: {formatPercent(metrics.max_drawdown)}
          </div>
        </div>
      </div>

      {/* Position Information */}
      {metrics.current_position && (
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-500">Current Position</div>
              <div className="mt-2">{getPositionBadge()}</div>
            </div>
            {metrics.current_position !== 'NEUTRAL' && (
              <div className="text-right">
                <div className="text-sm text-gray-500">Position Size</div>
                <div className="mt-1 text-lg font-semibold text-gray-900">
                  {formatCurrency(metrics.position_size)}
                </div>
                <div className={`mt-1 text-sm font-semibold ${(metrics.unrealized_pnl ?? 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                  Unrealized P/L: {formatCurrency(metrics.unrealized_pnl)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Trading Statistics */}
      <div className="card">
        <h4 className="mb-3 text-sm font-semibold text-gray-700">Trading Statistics</h4>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="text-xs text-gray-500">Total Trades</div>
            <div className="mt-1 text-lg font-semibold text-gray-900">
              {formatNumber(metrics.total_trades)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Winning</div>
            <div className="mt-1 text-lg font-semibold text-success">
              {formatNumber(metrics.winning_trades)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Losing</div>
            <div className="mt-1 text-lg font-semibold text-danger">
              {formatNumber(metrics.losing_trades)}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
