'use client'

interface CurrentPositionProps {
  position: {
    direction: 'LONG' | 'SHORT' | null
    entry_price: number
    current_price: number
    quantity: number
    leverage: number
    liquidation_price: number
    take_profit: number
    stop_loss: number
    break_even: number
    unrealized_pnl: number
    unrealized_pnl_pct: number
    margin_used: number
    duration_hours: number
  } | null
}

export default function CurrentPosition({ position }: CurrentPositionProps) {
  if (!position || !position.direction) {
    return (
      <div className="card">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
          Current Position
        </h3>
        <div className="flex items-center justify-center py-12 text-gray-500">
          <div className="text-center">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
            </svg>
            <p className="mt-4 font-medium">No open position</p>
            <p className="mt-1 text-sm">Waiting for trading signal...</p>
          </div>
        </div>
      </div>
    )
  }

  const isProfitable = position.unrealized_pnl >= 0
  const pnlColor = isProfitable ? 'text-green-600' : 'text-red-600'
  const positionColor = position.direction === 'LONG' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'

  const formatCurrency = (value: number) => {
    return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const getDistancePercent = (price: number, current: number) => {
    return ((price - current) / current * 100).toFixed(2)
  }

  return (
    <div className="card bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">
          Current Position
        </h3>
        <span className={`px-3 py-1 rounded-full text-sm font-bold ${positionColor}`}>
          {position.direction} {position.leverage}x
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
        {/* Entry Price */}
        <div>
          <p className="text-xs text-gray-600 mb-1">Entry Price</p>
          <p className="text-lg font-bold text-gray-900">{formatCurrency(position.entry_price)}</p>
        </div>

        {/* Current Price */}
        <div>
          <p className="text-xs text-gray-600 mb-1">Current Price</p>
          <p className="text-lg font-bold text-blue-600">{formatCurrency(position.current_price)}</p>
        </div>

        {/* Unrealized P&L */}
        <div className="col-span-2 md:col-span-1">
          <p className="text-xs text-gray-600 mb-1">Unrealized P&L</p>
          <p className={`text-xl font-bold ${pnlColor}`}>
            {isProfitable ? '+' : ''}{formatCurrency(position.unrealized_pnl)}
          </p>
          <p className={`text-sm font-semibold ${pnlColor}`}>
            {isProfitable ? '+' : ''}{position.unrealized_pnl_pct.toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Price Levels */}
      <div className="bg-white rounded-lg p-4 space-y-3 mb-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Price Levels</h4>

        {/* Take Profit */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm font-medium text-gray-700">Take Profit</span>
          </div>
          <div className="text-right">
            <p className="text-sm font-bold text-gray-900">{formatCurrency(position.take_profit)}</p>
            <p className="text-xs text-green-600">+{Math.abs(parseFloat(getDistancePercent(position.take_profit, position.current_price)))}%</p>
          </div>
        </div>

        {/* Break Even */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-amber-500 rounded-full"></div>
            <span className="text-sm font-medium text-gray-700">Break Even</span>
          </div>
          <div className="text-right">
            <p className="text-sm font-bold text-gray-900">{formatCurrency(position.break_even)}</p>
            <p className="text-xs text-amber-600">{getDistancePercent(position.break_even, position.current_price)}%</p>
          </div>
        </div>

        {/* Stop Loss */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-sm font-medium text-gray-700">Stop Loss</span>
          </div>
          <div className="text-right">
            <p className="text-sm font-bold text-gray-900">{formatCurrency(position.stop_loss)}</p>
            <p className="text-xs text-red-600">{getDistancePercent(position.stop_loss, position.current_price)}%</p>
          </div>
        </div>

        {/* Liquidation */}
        <div className="flex items-center justify-between pt-2 border-t border-gray-200">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-700 rounded-full animate-pulse"></div>
            <span className="text-sm font-bold text-red-700">Liquidation</span>
          </div>
          <div className="text-right">
            <p className="text-sm font-bold text-red-700">{formatCurrency(position.liquidation_price)}</p>
            <p className="text-xs text-red-600">{getDistancePercent(position.liquidation_price, position.current_price)}%</p>
          </div>
        </div>
      </div>

      {/* Position Details */}
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div>
          <p className="text-xs text-gray-600 mb-1">Quantity</p>
          <p className="font-semibold text-gray-900">{position.quantity.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-600 mb-1">Margin Used</p>
          <p className="font-semibold text-gray-900">{formatCurrency(position.margin_used)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-600 mb-1">Duration</p>
          <p className="font-semibold text-gray-900">{position.duration_hours}h</p>
        </div>
      </div>
    </div>
  )
}
