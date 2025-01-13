import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import matplotlib.pyplot as plt
import traceback

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    transaction_cost: float

class BacktestEngine:
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Trade] = []
        
        self.portfolio_values = pd.Series(dtype=float)
        self.portfolio_returns = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        
    def simulate(self, prices_df, signals_df):
        """Run backtest simulation"""
        try:
            prices_df.index = pd.to_datetime(prices_df.index, utc=True)
            signals_df.index = pd.to_datetime(signals_df.index, utc=True)
            
            positions = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
            self.portfolio_returns = pd.Series(0.0, index=prices_df.index, dtype=float)
            self.portfolio_values = pd.Series(self.initial_capital, index=prices_df.index, dtype=float)
            
            portfolio_value = self.initial_capital
            active_positions = {}
            
            for i in range(1, len(prices_df)):
                current_date = prices_df.index[i]
                prev_date = prices_df.index[i-1]
                
                current_prices = prices_df.loc[current_date]
                current_signals = signals_df.loc[current_date]
                prev_signals = signals_df.loc[prev_date]
                
                for symbol in prices_df.columns:
                    prev_signal = prev_signals[symbol]
                    curr_signal = current_signals[symbol]
                    
                    if prev_signal != curr_signal:
                        # close previous position if exists
                        if symbol in active_positions:
                            entry_price = active_positions[symbol]['entry_price']
                            entry_time = active_positions[symbol]['entry_time']
                            position_size = active_positions[symbol]['size']
                            direction = active_positions[symbol]['direction']
                            
                            # calculate pnl and record trade
                            pnl = (current_prices[symbol] - entry_price) * position_size * direction
                            trade = Trade(
                                entry_time=entry_time,
                                exit_time=current_date,
                                symbol=symbol,
                                direction=direction,
                                entry_price=entry_price,
                                exit_price=current_prices[symbol],
                                size=position_size,
                                pnl=pnl,
                                transaction_cost=0
                            )
                            self.trades.append(trade)
                            del active_positions[symbol]
                        
                        if curr_signal != 0:
                            active_positions[symbol] = {
                                'entry_time': current_date,
                                'entry_price': current_prices[symbol],
                                'size': abs(curr_signal),
                                'direction': np.sign(curr_signal)
                            }
                
                positions.loc[current_date] = current_signals.astype(float)
                
                # calculate returns based on position weights and price changes
                daily_return = (positions.loc[prev_date] * (current_prices / prices_df.loc[prev_date] - 1)).sum()
                
                new_portfolio_value = portfolio_value * (1 + daily_return)
                self.portfolio_returns.loc[current_date] = new_portfolio_value/portfolio_value - 1
                self.portfolio_values.loc[current_date] = new_portfolio_value
                portfolio_value = new_portfolio_value
            
            print("\nSimulation completed")
            print(f"Final portfolio value: {portfolio_value:.2f}")
            print(f"Number of trades: {len(self.trades)}")
            print(f"Number of non-zero returns: {(self.portfolio_returns != 0).sum()}")
            
            return True
            
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def _execute_trade(self, symbol, signal, price, timestamp):
        """Execute a trade based on the signal"""
        try:
            if np.isnan(price) or np.isnan(signal) or price <= 0:
                return
            
            position_size = self._calculate_position_size(signal, price, self.current_capital)
            
            if symbol in self.positions:
                self._close_position(symbol, price, timestamp)
            
            # minimum signal threshold
            if abs(signal) >= 0.1:
                self._open_position(symbol, position_size, price, timestamp)
            
        except Exception as e:
            print(f"Error executing trade for {symbol}: {str(e)}")

    def _close_position(self, symbol, exit_price, timestamp):
        if symbol in self.positions:
            entry_price = self.positions[symbol]['entry_price']
            position_size = self.positions[symbol]['size']
            pnl = (exit_price - entry_price) * position_size
            
            transaction_cost = self._calculate_transaction_cost(position_size, exit_price)
            
            trade = Trade(
                entry_time=self.positions[symbol]['entry_time'],
                exit_time=timestamp,
                symbol=symbol,
                direction=self.positions[symbol]['direction'],
                entry_price=entry_price,
                exit_price=exit_price,
                size=position_size,
                pnl=pnl,
                transaction_cost=transaction_cost
            )
            self.trades.append(trade)
            
            self.current_capital += pnl - transaction_cost
            del self.positions[symbol]

    def _open_position(self, symbol, position_size, price, timestamp):
        """Open a new position"""
        if position_size != 0:
            self.positions[symbol] = {
                'entry_time': timestamp,
                'entry_price': price,
                'size': position_size,
                'direction': np.sign(position_size)
            }

    def calculate_metrics(self):
        """Calculate performance metrics"""
        try:
            if len(self.portfolio_returns) == 0:
                raise ValueError("No portfolio returns data available")
            
            final_value = float(self.portfolio_values.iloc[-1])
            initial_value = float(self.portfolio_values.iloc[0])
            
            total_return = (final_value / initial_value) - 1
            
            days = len(self.portfolio_returns)
            annualized_return = ((1 + total_return) ** (252 / days)) - 1
            
            daily_volatility = self.portfolio_returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            # calculate drawdown
            cumulative_returns = (1 + self.portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            risk_free_rate = 0.02
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
            # convert to percentages for display
            percentage_metrics = {
                'total_return': metrics['total_return'] * 100,
                'annualized_return': metrics['annualized_return'] * 100,
                'volatility': metrics['volatility'] * 100,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'] * 100
            }
            
            return percentage_metrics
            
        except Exception as e:
            print(f"\nError in calculate_metrics: {str(e)}")
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
    
    def plot_results(self):
        """Plot equity curve and drawdown"""
        try:
            # Create equity curve DataFrame
            self.equity_curve = pd.DataFrame({
                'Equity': self.portfolio_values
            })
            
            # Calculate drawdowns
            rolling_max = self.portfolio_values.expanding().max()
            self.drawdowns = self.portfolio_values / rolling_max - 1
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot equity curve
            self.equity_curve['Equity'].plot(ax=ax1, title='Equity Curve')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Portfolio Value')
            ax1.grid(True)
            
            # Plot drawdown
            self.drawdowns.plot(ax=ax2, title='Drawdown', color='red')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            print("Debug info:")
            print(f"Portfolio values shape: {self.portfolio_values.shape}")
            print(f"Portfolio values index type: {type(self.portfolio_values.index)}")
            print(f"First few values:\n{self.portfolio_values.head()}")

    def _calculate_portfolio_value(self, current_prices):
        """Calculate current portfolio value"""
        try:
            total_value = self.current_capital
            
            for symbol, position in self.positions.items():
                if position != 0 and symbol in current_prices.index:
                    price = current_prices[symbol]
                    total_value += position * price
            
            return total_value
            
        except Exception as e:
            print(f"Error calculating portfolio value: {str(e)}")
            return self.current_capital

    def _calculate_position_size(self, signal, price, available_capital):
        """Calculate position size based on signal and available capital"""
        if signal == 0 or np.isnan(signal) or np.isinf(signal):
            return 0
        
        # use smaller percentage of capital per trade for risk management
        capital_per_trade = available_capital * 0.05
        shares = max(int(capital_per_trade / price), 0)
        
        if signal < 0:
            shares = -shares
        
        # apply position size limits (max 10% of capital)
        max_shares = int(available_capital * 0.1 / price)
        shares = np.clip(shares, -max_shares, max_shares)
        
        return shares

    def _calculate_volatility(self, price_series, window=20):
        """Calculate rolling volatility"""
        if isinstance(price_series, pd.Series):
            returns = price_series.pct_change().dropna()
            return returns.rolling(window=window).std() * np.sqrt(252)
        return 0.0

    def _calculate_transaction_cost(self, position_size, price):
        """Calculate transaction cost based on position size and price"""
        commission_rate = 0.002  # 0.2% commission
        slippage_rate = 0.002   # 0.2% slippage
        
        commission = max(abs(position_size * price * commission_rate), 1.0)  # minimum $1
        slippage = abs(position_size * price * slippage_rate)
        
        return commission + slippage

    def get_monthly_returns(self):
        """Calculate monthly returns from portfolio values"""
        try:
            # Ensure we have a datetime index
            if not isinstance(self.portfolio_values.index, pd.DatetimeIndex):
                self.portfolio_values.index = pd.to_datetime(self.portfolio_values.index)
            
            # Resample to month-end and calculate returns
            monthly_values = self.portfolio_values.resample('ME').last()
            monthly_returns = monthly_values.pct_change()
            
            # Create a dictionary with month as key and return as value
            returns_dict = {}
            for date, ret in monthly_returns.items():
                if pd.notna(ret):  # Skip NaN values
                    month_key = date.strftime('%Y-%m')
                    returns_dict[month_key] = ret
            
            return returns_dict
            
        except Exception as e:
            print(f"Error calculating monthly returns: {str(e)}")
            print(f"Portfolio values index type: {type(self.portfolio_values.index)}")
            print(f"First few indices: {self.portfolio_values.index[:5]}")
            return None

    def get_total_trades(self):
        """Get total number of trades executed"""
        return len(self.trades)

    def get_win_rate(self):
        """Calculate win rate from trades"""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        return winning_trades / len(self.trades)

    def get_average_win(self):
        """Calculate average winning trade return"""
        winning_trades = [trade.pnl for trade in self.trades if trade.pnl > 0]
        if not winning_trades:
            return 0.0
        return sum(winning_trades) / len(winning_trades) / self.initial_capital

    def get_average_loss(self):
        """Calculate average losing trade return"""
        losing_trades = [trade.pnl for trade in self.trades if trade.pnl < 0]
        if not losing_trades:
            return 0.0
        return sum(losing_trades) / len(losing_trades) / self.initial_capital

    def get_profit_factor(self):
        """Calculate profit factor (gross profits / gross losses)"""
        gross_profits = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        gross_losses = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
        
        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 0.0
        
        return gross_profits / gross_losses

if __name__ == "__main__":
    pass