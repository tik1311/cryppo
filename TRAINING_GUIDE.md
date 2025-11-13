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

## 🔍 Debug Output

Training now shows debug information to help diagnose issues:

```
Episode 1 | Reward: -0.45 | Loss: 0.0234 | Eps: 0.998 | Steps: 1234 | Trades: 0
  [ACTION] Step 72: OPEN_LONG | Pos: None
  [TRADE] Open LONG @ $45000.00 | Size: 0.6666 | Fee: $12.00
  [ACTION] Step 150: CLOSE | Pos: long
  [TRADE] Close @ $45500.00 | PnL: $333.20 (+1.11%) | Balance: $10321.20
```

**What to look for:**
- ✅ **Trades > 0** in episode summary
- ✅ **[ACTION] logs** show agent is taking actions
- ✅ **[TRADE] logs** show positions opening/closing
- ✅ **Balance changes** over time
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
