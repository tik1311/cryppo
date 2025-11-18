# CRYPPO Training Guide

## 🎯 Quick Start

```bash
# 1. Pull latest code
git pull origin claude/update-readme-comprehensive-011CV5JnQFA2qfeYF1fouCCL

# 2. Check system
python cryppo.py check

# 3. Download data (7 years of BTC)
python cryppo.py data download --symbol BTCUSDT --start 2019-01-01 --end 2025-11-13

# 4. Verify data
python cryppo.py data list

# 5. Quick test (2 episodes, sample data, ~30 seconds)
python cryppo.py train --symbol BTCUSDT --episodes 2 --sample-data --reward profit

# 6. Full training with PROFIT reward (recommended for first training)
python cryppo.py train --symbol BTCUSDT --episodes 100 --lr 0.00025 --reward profit

# 7. List trained models
python cryppo.py models

# 8. Backtest
python cryppo.py backtest --model checkpoints/final_agent.pth
```

---

## 🏆 Reward Types Explained

### 1. **PROFIT** (Recommended for First Training)
```bash
python cryppo.py train --reward profit --episodes 100
```

**What it does:**
- Direct PnL reward: `reward = (balance_change / initial_balance)`
- Simple and intuitive
- Agent learns: "make money = good, lose money = bad"

**Best for:**
- First training
- Understanding agent behavior
- Quick iterations

**Expected results:**
- Agent will trade frequently
- May have high volatility
- Focus on absolute profit

---

### 2. **RETURN** (Percentage-based)
```bash
python cryppo.py train --reward return --episodes 100
```

**What it does:**
- Percentage return per step: `reward = (equity_change / previous_equity)`
- Normalized by current capital
- Good for comparing across different balances

**Best for:**
- Stable learning
- Consistent performance
- Risk-aware trading

**Expected results:**
- More conservative trading
- Better risk management
- Stable equity curve

---

### 3. **SHARPE** (Risk-Adjusted, Default)
```bash
python cryppo.py train --reward sharpe --episodes 100
```

**What it does:**
- Risk-adjusted return: `sharpe = mean(returns) / std(returns)`
- Balances profit vs volatility
- Penalizes risky strategies

**Best for:**
- Professional trading
- Low drawdown requirement
- Consistent returns

**Expected results:**
- Lower volatility
- Smoother equity curve
- May trade less frequently but more accurately

---

### 4. **SORTINO** (Downside Risk-Adjusted)
```bash
python cryppo.py train --reward sortino --episodes 100
```

**What it does:**
- Like Sharpe but only penalizes downside volatility
- `sortino = mean(returns) / std(negative_returns)`
- Allows upside volatility

**Best for:**
- Asymmetric strategies
- Trend-following
- High-risk-high-reward

**Expected results:**
- May allow big wins
- Penalizes losses more heavily
- Good for bull markets

---

## 📊 Training Parameters

### Full Command
```bash
python cryppo.py train \
    --symbol BTCUSDT \          # Trading pair
    --episodes 100 \            # Number of training episodes
    --lr 0.00025 \              # Learning rate (default: 0.00025)
    --reward profit \           # Reward type (profit/return/sharpe/sortino)
    --sample-data               # Use synthetic data (for testing)
```

### Recommended Settings

**Fast Testing (5 minutes):**
```bash
python cryppo.py train --episodes 10 --sample-data --reward profit
```

**Quick Training (30 minutes on RTX 2060):**
```bash
python cryppo.py train --episodes 50 --reward profit
```

**Full Training (2-3 hours):**
```bash
python cryppo.py train --episodes 200 --reward sharpe
```

**Aggressive Exploration:**
```bash
python cryppo.py train --episodes 100 --reward profit --lr 0.0005
```

---

## 🛡️ Advanced Risk Management (NEW!)

CRYPPO now includes **professional-grade risk management** based on industry best practices from 2024:

### ✅ Mandatory TP/SL + Trailing Stop Loss

**Every position automatically has:**
1. **ATR-Based Take Profit** (TP = Entry ± ATR × 2.0)
2. **ATR-Based Stop Loss** (SL = Entry ± ATR × 1.0)
3. **Trailing Stop Loss** (follows price with ATR × 0.5 distance)

**What is ATR?**
- ATR (Average True Range) = measure of market volatility over 14 periods
- Volatile market → wider TP/SL (gives price room to move)
- Calm market → tighter TP/SL (takes profits faster)
- **Adapts automatically** to market conditions!

### 📊 How Trailing Stop Loss Works

**For LONG positions:**
```
Entry: $30,000
Initial SL: $29,700 (ATR × 1.0 = $300 below)
TP: $30,600 (ATR × 2.0 = $600 above)
Trailing Distance: $150 (ATR × 0.5)

Step 1: Price → $30,200 (+0.67%)
  Trailing SL: $30,050 (moves UP!)

Step 2: Price → $30,500 (+1.67%)
  Trailing SL: $30,350 (moves UP again!)

Step 3: Price → $30,300 (drops)
  Trailing SL: stays at $30,350 (NEVER moves down!)

Step 4: Price → $30,340
  HIT Trailing SL! Auto close @ $30,340
  Profit: +1.13% (protected gains!)
```

**Key Benefits:**
- ✅ Locks in profits as price moves favorably
- ✅ Never moves against you (only improves)
- ✅ Automatically enforced (agent cannot override!)
- ✅ Based on volatility (not arbitrary percentages)

### 🚫 No Position Flipping

**Old Behavior (BAD):**
```
Step 72: OPEN_SHORT @ $30,885
Step 74: Agent says "OPEN_LONG"
  → Auto-close SHORT (loss: -$80)
  → Auto-open LONG (fee: $12)
  Total cost: $92 in unintended trades!
```

**New Behavior (GOOD):**
```
Step 72: OPEN_SHORT @ $30,885
Step 74: Agent says "OPEN_LONG"
  → BLOCKED! "Must CLOSE first!"
  → Penalty: -0.01

Step 75: Agent says "CLOSE"
  → Close SHORT properly

Step 76: Agent says "OPEN_LONG"
  → Open LONG (intentional)
```

**Benefits:**
- ✅ No accidental position flips
- ✅ Agent must be intentional
- ✅ Saves on fees ($24/trade → controlled)

### 🎯 Risk:Reward Ratio = 1:2

Every trade has:
- **Risk:** ATR × 1.0 (Stop Loss)
- **Reward:** ATR × 2.0 (Take Profit)
- **Ratio:** 1:2 (industry standard)

**What this means:**
- Win 50% of trades → Break even
- Win > 50% of trades → Profitable!
- Trailing SL improves this ratio even more

### 📈 Longer Episodes = Better Learning

**Old:** Episodes were random length (avg ~1000 steps)
**New:** Guaranteed minimum 1200 steps per episode (+20%)

**Benefits:**
- ✅ More data per episode
- ✅ Better pattern recognition
- ✅ Improved learning stability

---

## 📝 Detailed Episode Logging

CRYPPO now automatically saves **detailed step-by-step logs** for:
- ✅ **Last training episode** (final episode of training)
- ✅ **All test episodes** (final evaluation after training)
- ✅ **All backtest runs** (when you run backtest command)

### What's Logged?

Every single step is recorded with complete details:
```json
{
  "episode": 100,
  "step": 1234,
  "action": 1,
  "action_name": "LONG",
  "price": 45123.45,
  "position": "long",
  "balance": 10234.56,
  "equity": 10456.78,
  "unrealized_pnl": 222.22,
  "trade_reward": 0.0123,
  "exploration_bonus": 0.0,
  "position_penalty": 0.0,
  "total_reward": 0.0145,
  "total_trades": 12,
  "win_rate": 0.583,
  "terminated": false,
  "truncated": false
}
```

### Log Files Created

After training or backtesting, you'll find these files in `logs/`:

**1. JSON Lines Format** (for programmatic analysis)
```
logs/episode_0100_20251113_143022.jsonl
```
- Each line is a valid JSON object
- One step per line
- Easy to parse with scripts
- Can load into pandas: `pd.read_json('file.jsonl', lines=True)`

**2. Human-Readable Summary** (for quick review)
```
logs/episode_0100_summary.txt
```
Contains:
- Episode statistics (total reward, steps, trades, win rate)
- **First 10 steps** (initial behavior)
- **Last 10 steps** (final behavior)
- All rewards, actions, prices, balances

### Example Summary Format
```
Episode: 100
Total Steps: 5432
Total Reward: 234.56
Total Trades: 28
Win Rate: 57.14%
Final Balance: $11,234.56

========================================
FIRST 10 STEPS
========================================

Step 1 | HOLD | Price: $43,200.00 | Pos: None | Balance: $10,000.00
  Rewards: trade=0.0000, explore=0.0000, penalty=0.0000, total=0.0012
  Trades: 0, Win Rate: 0.00%

Step 2 | LONG | Price: $43,250.00 | Pos: None | Balance: $10,000.00
  [TRADE] Open LONG @ $43,250.00
  Rewards: trade=0.0000, explore=0.0010, penalty=0.0000, total=0.0023
  Trades: 1, Win Rate: 0.00%

...

========================================
LAST 10 STEPS
========================================
...
```

### How to Use These Logs

**1. Verify Calculations Are Correct**
```bash
# Check if balance changes match rewards
grep "balance" logs/episode_0100_*.jsonl | head -20
```

**2. Understand Why Agent Isn't Trading**
```bash
# Count action distribution
cat logs/episode_0100_*.jsonl | jq -r '.action_name' | sort | uniq -c
```

**3. Find Bad Trades**
```bash
# Find steps with negative rewards
cat logs/episode_0100_*.jsonl | jq 'select(.trade_reward < -0.01)'
```

**4. Track Balance Over Time**
```python
import pandas as pd
df = pd.read_json('logs/episode_0100_20251113_143022.jsonl', lines=True)
print(df[['step', 'balance', 'equity', 'action_name', 'total_reward']])
```

**5. Analyze Position Holding Time**
```python
# How long does agent hold positions?
positions = df[df['position'].notna()]
print(f"Average position duration: {len(positions) / df['total_trades'].iloc[-1]} steps")
```

---

## 🔍 Real-time Training Output

Training now shows **live progress** for every episode (no more waiting!):

```
Episode   1 | Reward:   -12.34 | Avg(10):    -12.34 | Loss: 0.0234 | Eps: 0.998 | Steps: 5432 | Trades: 12 | Win%:  50.0 | Balance: $10,234
Episode   2 | Reward:    23.45 | Avg(10):      5.56 | Loss: 0.0198 | Eps: 0.996 | Steps: 5431 | Trades: 15 | Win%:  53.3 | Balance: $10,456
Episode   3 | Reward:    45.67 | Avg(10):     18.93 | Loss: 0.0176 | Eps: 0.994 | Steps: 5430 | Trades: 18 | Win%:  55.6 | Balance: $10,789
...
```

**What each column means:**
- **Reward**: Total reward for this episode
- **Avg(10)**: Average reward over last 10 episodes (trend indicator)
- **Loss**: Neural network training loss
- **Eps**: Epsilon (exploration rate) - starts at ~1.0, decays to 0.2
- **Steps**: Number of steps taken in episode
- **Trades**: Number of trades executed
- **Win%**: Percentage of profitable trades
- **Balance**: Final balance after episode

**What to look for:**
- ✅ **Trades > 0** after first few episodes
- ✅ **Win% gradually increasing** (40% → 50% → 60%)
- ✅ **Avg(10) trending upward** over time
- ✅ **Eps gradually decreasing** (1.0 → 0.5 → 0.2)
- ❌ If Trades = 0 after 10 episodes → problem!

---

## 🐛 Troubleshooting

### Problem: Agent still not trading (Trades = 0)

**Check 1: Is data loaded?**
```bash
python cryppo.py data list
# Should show .parquet file
```

**Check 2: Run with debug**
```bash
python cryppo.py train --episodes 2 --sample-data --reward profit
# Watch for [ACTION] and [TRADE] logs
```

**Check 3: Check epsilon value**
- Should start at ~1.0 (100% exploration)
- Slowly decay to 0.2 (20% exploration)
- If < 0.1 too early → not enough exploration

**Check 4: Try PROFIT reward first**
```bash
python cryppo.py train --reward profit --episodes 50
```
- Simplest reward signal
- Should see trades immediately

**Check 5: Check for errors**
```bash
tail -f logs/training_*.log
```

---

### Problem: Agent trades too much (100+ trades per episode)

**Solution 1: Use SHARPE reward**
```bash
python cryppo.py train --reward sharpe --episodes 100
```
- Penalizes excessive volatility
- Encourages quality over quantity

**Solution 2: Reduce learning rate**
```bash
python cryppo.py train --lr 0.0001 --episodes 100
```
- Slower, more stable learning

---

### Problem: Poor backtest results (big losses)

**Likely causes:**
1. **Overfitting to training data**
   - Solution: Train longer (200+ episodes)
   - Use different reward type

2. **Not enough exploration**
   - Solution: Train new model from scratch
   - Old model may have learned bad policy

3. **Wrong reward type for your goal**
   - Profit: aggressive, high volatility
   - Sharpe: conservative, low volatility
   - Try different reward types

---

## 📈 Expected Results

### After 50 Episodes (Profit Reward):
```
Total Trades:     30-60
Win Rate:         45-55%
Total Return:     -10% to +15%
Sharpe Ratio:     0.3 - 1.5
```

### After 100 Episodes (Sharpe Reward):
```
Total Trades:     20-40
Win Rate:         50-60%
Total Return:     0% to +20%
Sharpe Ratio:     0.8 - 2.5
Max Drawdown:     -5% to -15%
```

### After 200 Episodes (Well-trained):
```
Total Trades:     30-50
Win Rate:         55-65%
Total Return:     +5% to +30%
Sharpe Ratio:     1.5 - 3.0
Max Drawdown:     -3% to -10%
```

---

## 🎓 Advanced Tips

### 1. Progressive Training
```bash
# Stage 1: Learn to trade (50 episodes, profit)
python cryppo.py train --reward profit --episodes 50

# Stage 2: Learn risk management (100 episodes, sharpe)
python cryppo.py train --reward sharpe --episodes 100

# Stage 3: Optimize (200 episodes, sortino)
python cryppo.py train --reward sortino --episodes 200
```

### 2. Multiple Symbols
```bash
# Train on BTC
python cryppo.py data download --symbol BTCUSDT --start 2019-01-01 --end 2025-11-13
python cryppo.py train --symbol BTCUSDT --episodes 100 --reward profit

# Train on ETH
python cryppo.py data download --symbol ETHUSDT --start 2019-01-01 --end 2025-11-13
python cryppo.py train --symbol ETHUSDT --episodes 100 --reward profit

# Compare backtests
python cryppo.py backtest --model checkpoints/final_agent.pth
```

### 3. Monitor Training
```bash
# Terminal 1: Run training
python cryppo.py train --episodes 200

# Terminal 2: Watch logs
tail -f logs/training_*.log | grep "Episode"

# Terminal 3: Watch GPU
watch -n 1 nvidia-smi
```

---

## 🚨 Critical Fixes Applied

### Bug #1: Quadratic Penalty Growth ✅ FIXED
- **Problem:** Penalty grew O(n²) instead of O(n)
- **Result:** -145,000 penalty, agent never trades
- **Fix:** Removed no-trade penalty entirely

### Bug #2: Insufficient Exploration ✅ FIXED
- **Problem:** Epsilon decay too fast
- **Result:** Agent converges to "do nothing"
- **Fix:** Slower decay (0.9999), higher end (0.2)

### Bug #3: Action Bias ✅ FIXED
- **Problem:** 25% random HOLD actions
- **Result:** Agent learns HOLD is safe
- **Fix:** Biased exploration (5% HOLD, 42% LONG, 42% SHORT)

---

## 📝 Summary

**TL;DR:**
1. Use `--reward profit` for first training
2. Agent should trade within first 10 episodes
3. Debug logs show [ACTION] and [TRADE] events
4. Expected: 20-60 trades, 45-60% win rate
5. If no trades → check debug output

**Next Steps:**
```bash
# Clean start
rm -rf checkpoints/*.pth logs/*.log

# Train
python cryppo.py train --reward profit --episodes 100

# If you see trades → good!
# If no trades → report with debug output
```
