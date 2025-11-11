# Frontend - Phase 2 (Planned)

This directory will contain the Next.js 14 frontend for the RL Crypto Trading System.

## Planned Structure

```
client/
├── src/
│   ├── app/                    # Next.js 14 App Router
│   │   ├── page.tsx           # Home dashboard
│   │   ├── layout.tsx         # Root layout
│   │   ├── globals.css        # Global styles
│   │   ├── data/
│   │   │   └── page.tsx       # Data management
│   │   ├── train/
│   │   │   └── page.tsx       # Training interface
│   │   ├── backtest/
│   │   │   └── page.tsx       # Backtest interface
│   │   ├── paper-trade/
│   │   │   └── page.tsx       # Paper trading
│   │   ├── history/
│   │   │   └── page.tsx       # History viewer
│   │   └── system/
│   │       └── page.tsx       # System settings
│   │
│   ├── components/
│   │   ├── Chart.tsx          # Chart component
│   │   ├── TradingChart.tsx   # Trading visualization
│   │   ├── StatsCard.tsx      # Statistics card
│   │   └── TradesTable.tsx    # Trade history
│   │
│   └── lib/
│       ├── api.ts             # API client
│       └── websocket.ts       # WebSocket client
│
├── package.json
├── tsconfig.json
├── next.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Technology Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **State**: Zustand
- **Charts**: KLineChart / Recharts
- **UI Components**: shadcn/ui

## Setup (When Phase 2 starts)

```bash
cd client
npm install
npm run dev
```

## Status

🔄 **Phase 2 - Not Yet Implemented**

Current focus is on Phase 1 (Backend). Frontend development will begin once backend is stable and tested.
