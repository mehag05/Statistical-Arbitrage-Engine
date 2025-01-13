# Statistical Arbitrage Trading System

A pairs trading system I built to explore statistical arbitrage opportunities between 5 major tech stocks. 
The project started as an experiment in applying machine learning to traditional stat arb strategies, 
but evolved into a more comprehensive trading platform.

## What it Does

The system looks for temporary price divergences between related stocks (like MSFT-GOOGL or AAPL-MSFT) and trades them 
when the spread becomes statistically significant. It's built around two main components:

- `strat.py` handles all the strategy logic - finding tradeable pairs, generating signals, and managing positions
- `backtest.py` lets me test and analyze the strategy's performance on historical data

### Key Features

- Uses machine learning to score and select the best pairs to trade (trains on the first six months of 2024)
- Adapts position sizes based on volatility and correlation
- Manages risk through position limits and stop losses
- Accounts for transaction costs and market impact
- Provides detailed performance analytics

## How Well Does it Work?

The backtest results were pretty encouraging:

- Made a 62.19% total return (over five months of testing period in 2024)
- Annualized to 14.78%
- Sharpe ratio of 1.02
- Max drawdown stayed under 10%

The strategy was consistently profitable across all test months in 2024:

| Month | Return |
|-------|---------|
| Aug | 0.04% |
| Sep | 13.20% |
| Oct | 10.91% |
| Nov | 10.64% |
| Dec | 8.36% |

Out of 279 total trades:
- 55.20% were winners
- Profit factor of 1.61
- Never used more than 3.68x leverage

While these are just backtest results, the consistent profitability and reasonable risk metrics suggest the strategy has potential. I'm continuing to refine the implementation and explore ways to make it more robust.

To run locally, you'll need to install the packages in the requirements.txt file.
Then run the initial data collection methods in the strat.py file or use the pre-existing data in the data folder.
(You can modify the data collection methods to pull data from different sources or different time periods)

Once you have  data, you can run the existing code in strat.py's main method to see the results.
