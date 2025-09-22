from pathlib import Path
from types import SimpleNamespace

from src.utils.config_manager import get_config
from simple_app.config import AppConfig
from src.cli.commands import train_commands as train_cli
from src.cli.commands import backtest_commands as backtest_cli

runtime_config = get_config()
runtime_config.reset()

cfg = AppConfig()
proc_root = Path("oneclick/processed")
parquet_files = sorted(proc_root.glob(f"{cfg.symbol.lower()}_{cfg.timeframes[0]}_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
if not parquet_files:
    raise SystemExit("No processed parquet files available; run processing first")

proc_file = parquet_files[0]
print(f"Using processed file: {proc_file}")
model_root = Path("oneclick/models")
model_root.mkdir(parents=True, exist_ok=True)
model_type = runtime_config.get("model.model_type", "double_dqn")

train_args = SimpleNamespace(
    input=str(proc_file),
    output=str(model_root),
    model_type=model_type,
    window_size=runtime_config.get("data.window_size"),
    batch_size=runtime_config.get("model.batch_size"),
    epochs=None,
    learning_rate=None,
    discount_factor=None,
    target_update=None,
    validation_ratio=0.1,
    test_ratio=0.1,
    use_gpu=None,
    tensorboard=False,
    seed=runtime_config.get("general.random_seed"),
    config=None,
)
train_cli.handle_model(train_args)

model_dir = sorted(model_root.glob(f"{model_type}_*"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
model_path = model_dir / "model.pt"
print(f"Model: {model_path}")

eval_args = SimpleNamespace(
    model=str(model_path),
    input=str(proc_file),
    output=str(Path("oneclick/reports") / "test_metrics.json"),
    batch_size=None,
    window_size=None,
    metrics="all",
    plot=False,
    use_gpu=None,
)
train_cli.handle_evaluate(eval_args)

backtest_root = Path("oneclick/backtest")
backtest_root.mkdir(parents=True, exist_ok=True)
backtest_args = SimpleNamespace(
    model=str(model_path),
    input=str(proc_file),
    output=str(backtest_root),
    start_date=None,
    end_date=None,
    initial_balance=10_000.0,
    leverage=None,
    fee_rate=None,
    stop_loss=None,
    take_profit=None,
    window_size=None,
    batch_size=None,
    use_gpu=None,
    plot=False,
    verbose=False,
    config=None,
)
backtest_cli.handle_run(backtest_args)
