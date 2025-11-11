"""
Database Schema for RL Crypto Trading
SQLite database for tracking training runs and backtest results
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
import json


class Database:
    """SQLite Database Manager"""

    def __init__(self, db_path: str = './data/trading.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.create_tables()

    def create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()

        # Training runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT NOT NULL,
                model_type TEXT NOT NULL,
                config TEXT,
                n_episodes INTEGER,
                best_reward REAL,
                final_reward REAL,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Training episodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                episode INTEGER NOT NULL,
                reward REAL,
                steps INTEGER,
                epsilon REAL,
                loss REAL,
                equity REAL,
                trades INTEGER,
                win_rate REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES training_runs (id)
            )
        ''')

        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                data_symbol TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                initial_balance REAL,
                final_balance REAL,
                total_return_pct REAL,
                total_trades INTEGER,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                results_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL UNIQUE,
                model_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                training_run_id INTEGER,
                validation_reward REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (training_run_id) REFERENCES training_runs (id)
            )
        ''')

        self.conn.commit()

    # ========== Training Runs ==========

    def create_training_run(
        self,
        run_name: str,
        model_type: str,
        config: dict,
        n_episodes: int
    ) -> int:
        """Create new training run"""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO training_runs (
                run_name, start_time, status, model_type, config, n_episodes
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            run_name,
            datetime.now(),
            'running',
            model_type,
            json.dumps(config),
            n_episodes
        ))

        self.conn.commit()
        return cursor.lastrowid

    def update_training_run(
        self,
        run_id: int,
        status: Optional[str] = None,
        end_time: Optional[datetime] = None,
        best_reward: Optional[float] = None,
        final_reward: Optional[float] = None,
        model_path: Optional[str] = None
    ):
        """Update training run"""
        cursor = self.conn.cursor()

        updates = []
        values = []

        if status:
            updates.append("status = ?")
            values.append(status)
        if end_time:
            updates.append("end_time = ?")
            values.append(end_time)
        if best_reward is not None:
            updates.append("best_reward = ?")
            values.append(best_reward)
        if final_reward is not None:
            updates.append("final_reward = ?")
            values.append(final_reward)
        if model_path:
            updates.append("model_path = ?")
            values.append(model_path)

        if updates:
            values.append(run_id)
            query = f"UPDATE training_runs SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, values)
            self.conn.commit()

    def add_training_episode(
        self,
        run_id: int,
        episode: int,
        reward: float,
        steps: int,
        epsilon: float,
        loss: float,
        equity: float,
        trades: int,
        win_rate: float
    ):
        """Add training episode data"""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO training_episodes (
                run_id, episode, reward, steps, epsilon, loss,
                equity, trades, win_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, episode, reward, steps, epsilon, loss, equity, trades, win_rate))

        self.conn.commit()

    def get_training_runs(self, limit: int = 100) -> List[Dict]:
        """Get recent training runs"""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT * FROM training_runs
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

        return [dict(row) for row in cursor.fetchall()]

    # ========== Backtest Results ==========

    def save_backtest_result(
        self,
        run_name: str,
        model_path: str,
        data_symbol: str,
        start_date: str,
        end_date: str,
        results: dict
    ) -> int:
        """Save backtest results"""
        cursor = self.conn.cursor()

        summary = results.get('summary', {})
        trades = results.get('trades', {})
        performance = results.get('performance', {})

        cursor.execute('''
            INSERT INTO backtest_results (
                run_name, model_path, data_symbol, start_date, end_date,
                initial_balance, final_balance, total_return_pct,
                total_trades, win_rate, sharpe_ratio, max_drawdown_pct,
                results_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_name,
            model_path,
            data_symbol,
            start_date,
            end_date,
            summary.get('initial_balance', 0),
            summary.get('final_balance', 0),
            summary.get('total_return_pct', 0),
            trades.get('total_trades', 0),
            trades.get('win_rate', 0),
            performance.get('sharpe_ratio', 0),
            performance.get('max_drawdown_pct', 0),
            json.dumps(results)
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_backtest_results(self, limit: int = 100) -> List[Dict]:
        """Get recent backtest results"""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT * FROM backtest_results
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

        results = []
        for row in cursor.fetchall():
            result_dict = dict(row)
            # Parse JSON results
            if result_dict.get('results_json'):
                result_dict['results'] = json.loads(result_dict['results_json'])
            results.append(result_dict)

        return results

    # ========== Models ==========

    def save_model(
        self,
        model_name: str,
        model_type: str,
        model_path: str,
        training_run_id: Optional[int] = None,
        validation_reward: Optional[float] = None,
        notes: Optional[str] = None
    ) -> int:
        """Save model information"""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO models (
                model_name, model_type, model_path, training_run_id,
                validation_reward, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_name, model_type, model_path, training_run_id, validation_reward, notes))

        self.conn.commit()
        return cursor.lastrowid

    def get_models(self, limit: int = 100) -> List[Dict]:
        """Get available models"""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT * FROM models
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection"""
        self.conn.close()
