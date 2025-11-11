'use client'

import { useState } from 'react'

interface Position {
  id: string
  direction: 'LONG' | 'SHORT'
  entry_price: number
  exit_price: number
  entry_time: number
  exit_time: number
  quantity: number
  leverage: number
  pnl: number
  pnl_pct: number
  duration_hours: number
  exit_reason: string
}

interface PositionHistoryProps {
  positions: Position[]
  maxItems?: number
}

export default function PositionHistory({ positions, maxItems = 20 }: PositionHistoryProps) {
  const [expanded, setExpanded] = useState<string | null>(null)

  const displayPositions = positions.slice(-maxItems).reverse()

  const formatCurrency = (value: number) => {
    return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })
  }

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'text-green-600' : 'text-red-600'
  }

  const getPositionBadge = (direction: string) => {
    return direction === 'LONG'
      ? 'bg-green-100 text-green-800 border-green-300'
      : 'bg-red-100 text-red-800 border-red-300'
  }

  const getExitReasonBadge = (reason: string) => {
    const badges: { [key: string]: string } = {
      'Take Profit': 'bg-green-100 text-green-700',
      'Stop Loss': 'bg-red-100 text-red-700',
      'Manual Close': 'bg-blue-100 text-blue-700',
      'Time Limit': 'bg-amber-100 text-amber-700',
      'Liquidated': 'bg-red-200 text-red-900'
    }
    return badges[reason] || 'bg-gray-100 text-gray-700'
  }

  const totalPnL = positions.reduce((sum, pos) => sum + pos.pnl, 0)
  const winningPositions = positions.filter(p => p.pnl > 0).length
  const winRate = positions.length > 0 ? (winningPositions / positions.length * 100).toFixed(1) : '0.0'

  return (
    <div className="card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Position History</h3>
        <div className="flex items-center gap-4 text-sm">
          <div className="text-right">
            <p className="text-xs text-gray-500">Total P&L</p>
            <p className={`font-bold ${getPnLColor(totalPnL)}`}>
              {totalPnL >= 0 ? '+' : ''}{formatCurrency(totalPnL)}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-500">Win Rate</p>
            <p className="font-bold text-blue-600">{winRate}%</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-500">Total</p>
            <p className="font-bold text-gray-700">{positions.length}</p>
          </div>
        </div>
      </div>

      {displayPositions.length === 0 ? (
        <div className="py-12 text-center text-gray-500">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <p className="mt-4">No closed positions yet</p>
        </div>
      ) : (
        <div className="space-y-2">
          {displayPositions.map((position) => {
            const isExpanded = expanded === position.id
            const isProfitable = position.pnl >= 0

            return (
              <div
                key={position.id}
                className={`border rounded-lg p-3 transition-all cursor-pointer hover:shadow-md ${
                  isProfitable ? 'border-green-200 bg-green-50/30' : 'border-red-200 bg-red-50/30'
                }`}
                onClick={() => setExpanded(isExpanded ? null : position.id)}
              >
                {/* Summary Row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-1 rounded text-xs font-bold border ${getPositionBadge(position.direction)}`}>
                      {position.direction} {position.leverage}x
                    </span>
                    <div className="text-sm">
                      <span className="font-medium text-gray-700">
                        {formatCurrency(position.entry_price)} → {formatCurrency(position.exit_price)}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <p className={`text-lg font-bold ${getPnLColor(position.pnl)}`}>
                        {isProfitable ? '+' : ''}{formatCurrency(position.pnl)}
                      </p>
                      <p className={`text-xs font-semibold ${getPnLColor(position.pnl)}`}>
                        {isProfitable ? '+' : ''}{position.pnl_pct.toFixed(2)}%
                      </p>
                    </div>
                    <svg
                      className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-gray-200 grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-xs text-gray-500">Entry Time</p>
                      <p className="font-medium text-gray-900">{formatTimestamp(position.entry_time)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Exit Time</p>
                      <p className="font-medium text-gray-900">{formatTimestamp(position.exit_time)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Quantity</p>
                      <p className="font-medium text-gray-900">{position.quantity.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Duration</p>
                      <p className="font-medium text-gray-900">{position.duration_hours}h</p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-xs text-gray-500">Exit Reason</p>
                      <span className={`inline-block mt-1 px-2 py-1 rounded text-xs font-semibold ${getExitReasonBadge(position.exit_reason)}`}>
                        {position.exit_reason}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {positions.length > maxItems && (
        <div className="mt-4 text-center text-xs text-gray-500">
          Showing last {maxItems} positions. Total: {positions.length} positions.
        </div>
      )}
    </div>
  )
}
