import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import requests
from alpha_vantage.timeseries import TimeSeries
import datetime as dt
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import json
import os
from hmmlearn import hmm
import time
import traceback
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from backtest import BacktestEngine
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class StatArbStrategy:
    def __init__(self, symbols, lookback_period=252):
        self.symbols = symbols
        self.lookback_period = lookback_period
        
        self.alphavantage_key = os.getenv('alphavantage_key')
        self.finnhub_key = os.getenv('finnhub_key')
        self.market_impact = {}
        self.liquidity_scores = {}
        # optional data cache
        self.collected_data = {
            'market_data': {},
            'alternative_data': {},
            'signals': {},
            'performance': {}
        }

    def fetch_market_data_init(self, symbol, start_date, end_date):
        """Fetch initial market data from Alpha Vantage and save to files"""
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        daily_file = f'{data_dir}/{symbol}_daily.csv'
        intraday_file = f'{data_dir}/{symbol}_intraday.csv'
        
        try:
            # Fetch daily data
            daily_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={self.alphavantage_key}'
            daily_response = requests.get(daily_url)
            daily_data = daily_response.json()
            
            if 'Time Series (Daily)' not in daily_data:
                print(f"Error fetching daily data for {symbol}: {daily_data.get('Note', 'Unknown error')}")
                time.sleep(12)
                return None
            
            daily_records = []
            for date_str, values in daily_data['Time Series (Daily)'].items():
                date = pd.to_datetime(date_str)
                if start_date <= date <= end_date:
                    daily_records.append({
                        'date': date_str,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(values['5. volume'])
                    })
            
            daily_df = pd.DataFrame(daily_records)
            if not daily_df.empty:
                daily_df.set_index('date', inplace=True)
                daily_df.sort_index(inplace=True)
                daily_df.to_csv(daily_file)
                print(f"Saved {len(daily_df)} daily records for {symbol}")
            
            time.sleep(12)
            
            # Fetch intraday data
            intraday_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&outputsize=full&apikey={self.alphavantage_key}'
            intraday_response = requests.get(intraday_url)
            intraday_data = intraday_response.json()
            
            if 'Time Series (60min)' not in intraday_data:
                print(f"Error fetching intraday data for {symbol}: {intraday_data.get('Note', 'Unknown error')}")
                return None
     
            intraday_records = []
            for datetime_str, values in intraday_data['Time Series (60min)'].items():
                datetime = pd.to_datetime(datetime_str)
                if start_date <= datetime <= end_date:
                    intraday_records.append({
                        'Datetime': datetime_str,
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': int(values['5. volume']),
                        'Dividends': 0.0,
                        'Stock Splits': 0.0
                    })
            
            intraday_df = pd.DataFrame(intraday_records)
            if not intraday_df.empty:
                intraday_df.set_index('Datetime', inplace=True)
                intraday_df.sort_index(inplace=True)
                intraday_df.to_csv(intraday_file)
                print(f"Saved {len(intraday_df)} intraday records for {symbol}")
            
            return {
                'daily': daily_df if not daily_df.empty else None,
                'intraday': intraday_df if not intraday_df.empty else None
            }
            
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {str(e)}")
            traceback.print_exc()
            return None
        
    def fetch_alternative_data_init(self, symbol):
        """Fetch sentiment/other data from various sources and save to files"""
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        sentiment_file = f'{data_dir}/{symbol}_sentiment.json'
        
        try:
            # Get news from multiple sources
            yf_sentiment = self._fetch_yahoo_finance_sentiment(symbol)
            reddit_sentiment = self.fetch_weekly_reddit_sentiment(symbol)
            av_sentiment = self.fetch_weekly_alpha_vantage_sentiment(symbol)
            
            daily_sentiments = {}
            weekly_sentiments = {}
            
            # Add Yahoo Finance data
            if yf_sentiment:
                daily_sentiments.update(yf_sentiment)
                
            # Add Reddit weekly data
            if reddit_sentiment:
                weekly_sentiments.update(reddit_sentiment)
                
            # Add Alpha Vantage weekly data
            if av_sentiment:
                weekly_sentiments.update(av_sentiment)
            
            # skip if no valid sentiment data was collected
            if not daily_sentiments and not weekly_sentiments:
                print(f"No valid sentiment data found for {symbol}")
                return None
            
            # Prepare sentiment data structure
            sentiment_data = {
                'daily_sentiment': daily_sentiments,
                'weekly_sentiment': weekly_sentiments,
                'metadata': {
                    'last_updated': dt.datetime.now().isoformat(),
                    'sources': {
                        'yahoo_finance': bool(yf_sentiment),
                        'reddit': bool(reddit_sentiment),
                        'alpha_vantage': bool(av_sentiment)
                    }
                }
            }
            
            # Save to file
            with open(sentiment_file, 'w') as f:
                json.dump(sentiment_data, f, indent=4)
            print(f"Saved sentiment data for {symbol} to {sentiment_file}")
            
            result = {
                'sentiment': sentiment_data,
                'fundamentals': {}
            }
            
            # Cache results if needed
            self.collected_data['alternative_data'][symbol] = result
            return result
            
        except Exception as e:
            print(f"Error fetching alternative data for {symbol}: {str(e)}")
            # traceback.print_exc()
            return None

    def _fetch_yahoo_finance_sentiment(self, symbol):
        """Helper method to fetch Yahoo Finance sentiment"""
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            if not news_data:
                print(f"No Yahoo Finance news data available for {symbol}")
                return {}
            
            sia = SentimentIntensityAnalyzer()
            daily_sentiments = {}
            
            for article in news_data:
                if 'content' in article and isinstance(article['content'], dict):
                    timestamp = pd.to_datetime(article.get('providerPublishTime', 0), unit='s')
                    
                    if timestamp.year < 2000:
                        continue
                    
                    date_str = timestamp.strftime('%Y-%m-%d')
                    
                    title = article['content'].get('title', '')
                    summary = article['content'].get('summary', '')
                    
                    title_scores = sia.polarity_scores(title)
                    summary_scores = sia.polarity_scores(summary)
                    
                    combined_score = {
                        'compound': title_scores['compound'] * 0.6 + summary_scores['compound'] * 0.4,
                        'pos': title_scores['pos'] * 0.6 + summary_scores['pos'] * 0.4,
                        'neg': title_scores['neg'] * 0.6 + summary_scores['neg'] * 0.4,
                        'neu': title_scores['neu'] * 0.6 + summary_scores['neu'] * 0.4
                    }
                    
                    if date_str not in daily_sentiments:
                        daily_sentiments[date_str] = {
                            'scores': [],
                            'articles': 0
                        }
                    
                    daily_sentiments[date_str]['scores'].append(combined_score)
                    daily_sentiments[date_str]['articles'] += 1
            
            return daily_sentiments
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance sentiment for {symbol}: {str(e)}")
            return {}
    
    def fetch_market_data(self, symbol, start_date, end_date):
        """Load market data from saved CSV files, run after initialization"""
        data_dir = 'data'
        daily_file = f'{data_dir}/{symbol}_daily.csv'
        intraday_file = f'{data_dir}/{symbol}_intraday.csv'
        
        try:
            start_ts = pd.to_datetime(start_date).tz_localize(None)
            end_ts = pd.to_datetime(end_date).tz_localize(None)
            
            # Load daily data
            if os.path.exists(daily_file):
                daily_data = pd.read_csv(daily_file)
                daily_data['Date'] = pd.to_datetime(daily_data.iloc[:, 0]).dt.tz_localize(None)
                daily_data.set_index('Date', inplace=True)
                daily_data = daily_data.sort_index()
                daily_data = daily_data[(daily_data.index >= start_ts) & (daily_data.index <= end_ts)]
            else:
                print(f"No saved daily data found for {symbol}")
                return None
            
            # Load intraday data
            if os.path.exists(intraday_file):
                intraday_data = pd.read_csv(intraday_file)
                intraday_data['Date'] = pd.to_datetime(intraday_data.iloc[:, 0], utc=True).dt.tz_localize(None)
                intraday_data.set_index('Date', inplace=True)
                intraday_data = intraday_data.sort_index()
                intraday_data = intraday_data[(intraday_data.index >= start_ts) & (intraday_data.index <= end_ts)]
            else:
                print(f"No saved intraday data found for {symbol}")
                intraday_data = None
            
            result = {
                'daily': daily_data,
                'intraday': intraday_data
            }
            return result
            
        except Exception as e:
            print(f"Error loading market data for {symbol}: {str(e)}")
            return None

    def fetch_alternative_data(self, symbol):
        """Load sentiment data from saved JSON file, run after initialization"""
        data_dir = 'data'
        sentiment_file = f'{data_dir}/{symbol}_sentiment.json'
        
        try:
            if os.path.exists(sentiment_file):
                with open(sentiment_file, 'r') as f:
                    sentiment_data = json.load(f)
                    
                result = {
                    'sentiment': sentiment_data,
                    'fundamentals': {}
                }
                return result
            else:
                print(f"No saved sentiment data found for {symbol}")
                return None
            
        except Exception as e:
            print(f"Error loading sentiment data for {symbol}: {str(e)}")
            return None
    
    def calculate_market_microstructure(self, symbol, tick_data):
        """Calculate market microstructure metrics"""
        # volume-weighted average price
        vwap = np.sum(tick_data['price'] * tick_data['volume']) / np.sum(tick_data['volume'])
        
        # Order Flow Imbalance
        bid_volume = tick_data[tick_data['side'] == 'buy']['volume'].sum()
        ask_volume = tick_data[tick_data['side'] == 'sell']['volume'].sum()
        order_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # kyle's Lambda (Price Impact)
        returns = tick_data['price'].pct_change()
        signed_volume = tick_data['volume'] * np.where(tick_data['side'] == 'buy', 1, -1)
        kyle_lambda = np.abs(np.corrcoef(returns[1:], signed_volume[:-1])[0,1])
        
        return {
            'vwap': vwap,
            'order_imbalance': order_imbalance,
            'kyle_lambda': kyle_lambda
        }
    
    def detect_regime_changes(self, returns_data):
        """Detect market regime changes using HMM"""
        try:
            X = np.array(returns_data).reshape(-1, 1)
            
            # Initialize and fit HMM
            model = hmm.GaussianHMM(
                n_components=2,  # Two regimes: high/low volatility
                covariance_type="full",
                n_iter=100,
                random_state=42,
                tol=0.01
            )
            
            # fit HMM
            model.fit(X)
            
            # predict states
            hidden_states = model.predict(X)
            
            # count regime changes and identify dominant regime
            num_changes = np.sum(np.diff(hidden_states) != 0)
            dominant_regime = stats.mode(hidden_states)[0]
            
            return {
                'dominant_regime': dominant_regime,
                'num_changes': int(num_changes)
            }
            
        except Exception as e:
            print(f"HMM failed: {str(e)}")
            # Return default single regime
            return {
                'dominant_regime': 0,
                'num_changes': 0
            }
        
    def calculate_adaptive_thresholds(self, zscore_series, market_regime):
        """Calculate regime-dependent trading thresholds"""
        volatility = zscore_series.rolling(20).std()
        base_threshold = 2.0
        
        # adjust thresholds based on regime
        regime_multiplier = np.where(market_regime == 0, 1.2, 0.8)  # more conservative in high vol regime
        return base_threshold * regime_multiplier * (1 + volatility)
    
    def kelly_position_sizing(self, win_prob, win_loss_ratio):
        """Calculate optimal position size using Kelly Criterion"""
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        return max(0, kelly_fraction)
    
    def calculate_optimal_hedge_ratio(self, price1, price2):
        """Calculate optimal hedge ratio using rolling regression"""
        from statsmodels.regression.rolling import RollingOLS
        import statsmodels.api as sm
        
        X = sm.add_constant(price2)
        
        # rolling regression
        rols = RollingOLS(endog=price1, exog=X, window=63)
        rolling_reg = rols.fit()
        
        return rolling_reg.params['x1']  # Return beta
    
    def detect_anomalies(self, spread_series):
        """Detect anomalies in spread series using Isolation Forest"""
        clf = IsolationForest(contamination=0.1, random_state=42)
        anomalies = clf.fit_predict(spread_series.reshape(-1, 1))
        return anomalies
    
    def calculate_transaction_costs(self, price, volume, side):
        """Estimate transaction costs including market impact, theoretical"""
        commission = 0.0001 * price * volume  # 1 bps
        
        spread_cost = 0.0001 * price * volume  # 1 bps
        
        market_impact = 0.1 * price * np.sqrt(volume / self.avg_daily_volume)
        
        total_cost = commission + spread_cost + market_impact
        
        return total_cost
    
    def optimize_execution(self, total_volume, price_series, urgency='medium'):
        """Optimize trade execution using TWAP/VWAP strategies"""
        urgency_params = {
            'high': {'intervals': 3, 'timing_factor': 0.8},
            'medium': {'intervals': 6, 'timing_factor': 1.0},
            'low': {'intervals': 12, 'timing_factor': 1.2}
        }
        
        params = urgency_params[urgency]

        interval_volume = total_volume / params['intervals']
        twap_schedule = [interval_volume] * params['intervals']
        
        volume_profile = self.get_historical_volume_profile()
        adjusted_schedule = [vol * profile for vol, profile in zip(twap_schedule, volume_profile)]
        
        return adjusted_schedule
    
    def risk_management(self, positions, market_data):
        """Comprehensive risk management module"""
        returns = market_data.pct_change()
        var_95 = np.percentile(returns, 5)
        
        # calculate expected shortfall
        es_95 = returns[returns < var_95].mean()
        
        # position limits based on risk metrics
        max_position = self.calculate_position_limit(var_95, es_95)
        
        # stop-loss triggers
        stop_loss_triggered = self.check_stop_loss(positions, market_data)
        
        return {
            'var_95': var_95,
            'es_95': es_95,
            'position_limit': max_position,
            'stop_loss_triggered': stop_loss_triggered
        }

    def run_strategy(self, start_date, end_date):
        """Run the statistical arbitrage strategy"""
        results = []
        
        for symbol in self.symbols:
            print(f"\nProcessing {symbol}...")
            
            # load sentiment data from file
            alt_data = self.fetch_alternative_data(symbol)
            sentiment_score = 0
            if alt_data and 'sentiment' in alt_data:
                sentiment_score = alt_data['sentiment'].get('compound_score', 0)
            
            try:
                # get market data
                market_data = self.fetch_market_data(symbol, start_date, end_date)
                if market_data is None or 'daily' not in market_data:
                    print(f"No market data available for {symbol}")
                    continue
                
                daily_data = market_data['daily']
                
                # calculate returns
                returns = daily_data['close'].pct_change().dropna()
                
                if len(returns) < 2:
                    print(f"Insufficient data points for {symbol}")
                    continue

                avg_return = returns.mean()
                volatility = returns.std()
                sharpe_ratio = (avg_return / volatility) if volatility != 0 else 0
                
                last_price = daily_data['close'].iloc[-1]
                price_change = last_price - daily_data['close'].iloc[0]
                
                regime_changes = self.detect_regime_changes(returns)
                if regime_changes is None:
                    regime_changes = {'dominant_regime': 0, 'num_changes': 0}
                
                results.append({
                    'symbol': symbol,
                    'avg_return': avg_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'sentiment_score': sentiment_score,
                    'data_points': len(returns),
                    'last_price': last_price,
                    'price_change': price_change,
                    'dominant_regime': regime_changes['dominant_regime'],
                    'regime_changes': regime_changes['num_changes']
                })
                
            except Exception as e:
                print(f"Error calculating mode for {symbol}: {str(e)}")
                continue
        
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)
            return results_df
        else:
            return pd.DataFrame()

    def generate_signals_from_factors(self, factor_results, dates):
        """Generate trading signals by combining momentum, mean reversion, and volatility factors"""
        try:
            dates = pd.to_datetime(dates)
            symbols = list(factor_results.keys())
            signals = pd.DataFrame(0, index=dates, columns=symbols)
            
            print("\nGenerating signals for each date...")
            signal_stats = {
                'momentum': [],
                'mean_reversion': [],
                'volatility': [],
                'final': []
            }
            
            for date in dates:
                regime = self.identify_regime(date)
                
                for symbol in symbols:
                    try:
                        mom = self._calculate_momentum_signal(symbol, date)
                        rev = self._calculate_mean_reversion_signal(symbol, date)
                        vol = self._calculate_volatility_signal(symbol, date)
                        
                        signal_stats['momentum'].append(mom)
                        signal_stats['mean_reversion'].append(rev)
                        signal_stats['volatility'].append(vol)
                        
                        vol_adjustment = 1.0 / (vol * np.sqrt(252)) if vol > 0 else 1.0
                        vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)  # Limit adjustment range
                        
                        if regime == 1:  # high volatility
                            signal = (0.3 * mom + 0.7 * rev) * vol_adjustment
                        else:  # low volatility
                            signal = (0.7 * mom + 0.3 * rev) * vol_adjustment
                        
                        # apply threshold and limits
                        if abs(signal) < 0.1:
                            signal = 0
                        signal = np.clip(signal, -1, 1)
                        
                        signal_stats['final'].append(signal)
                        signals.loc[date, symbol] = signal
                        
                        print(f"{symbol}: mom={mom:.3f}, rev={rev:.3f}, vol={vol:.3f}, final={signal:.3f}")
                        
                    except Exception as e:
                        print(f"Error processing {symbol} on {date}: {str(e)}")
                        signals.loc[date, symbol] = 0
            
            # print
            for signal_type, values in signal_stats.items():
                values = np.array(values)
            
            return signals
            
        except Exception as e:
            print(f"Error in signal generation: {str(e)}")
            return pd.DataFrame(0, index=dates, columns=symbols)

    def save_data_to_csv(self, start_date, end_date):
        """Save all collected data to CSV files, helper method"""
        for symbol in self.symbols:
            market_data = self.fetch_market_data(symbol, start_date, end_date)
            if market_data:
                self.collected_data['market_data'][symbol] = market_data
                
                if market_data['daily'] is not None:
                    market_data['daily'].to_csv(f'data/{symbol}_daily.csv')
                
                if market_data['intraday'] is not None:
                    market_data['intraday'].to_csv(f'data/{symbol}_intraday.csv')
            
            alt_data = self.fetch_alternative_data(symbol)
            self.collected_data['alternative_data'][symbol] = alt_data

            alt_df = pd.DataFrame([alt_data['sentiment']])
            alt_df.to_csv(f'data/{symbol}_alternative_data.csv')
        
        summary_data = []
        for symbol in self.symbols:
            symbol_data = {
                'symbol': symbol,
                'daily_records': len(self.collected_data['market_data'].get(symbol, {}).get('daily', pd.DataFrame())) if symbol in self.collected_data['market_data'] else 0,
                'intraday_records': len(self.collected_data['market_data'].get(symbol, {}).get('intraday', pd.DataFrame())) if symbol in self.collected_data['market_data'] else 0,
                'sentiment_score': self.collected_data['alternative_data'].get(symbol, {}).get('sentiment', {}).get('compound_score', 0),
                'news_count': self.collected_data['alternative_data'].get(symbol, {}).get('sentiment', {}).get('news_count', 0)
            }
            summary_data.append(symbol_data)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('data/data_summary.csv', index=False)
        
        return summary_df

    def clear_cache(self):
        """Clear all cached data, helper method"""
        self.collected_data = {
            'market_data': {},
            'alternative_data': {},
            'signals': {},
            'performance': {}
        }

    def calculate_factor_correlations(self, returns_df, market_returns, factors_dict):
        """Calculate correlations between alpha and various factors"""
        try:
            alpha = returns_df - market_returns
            
            # create a DataFrame with all factors
            factors_df = pd.DataFrame({
                'Alpha': alpha,
                'Market_Returns': market_returns,
                'Sentiment': factors_dict['sentiment'],
                'Regime': factors_dict['regime'],
                'Volatility': factors_dict['volatility'],
                'RSI': factors_dict['rsi'],
                'MACD': factors_dict['macd']
            })
            
            # calculate correlation matrix
            corr_matrix = factors_df.corr()
            
            # calculate factor betas with regression
            factor_betas = self.calculate_factor_betas(alpha, factors_df)
            
            return {
                'correlation_matrix': corr_matrix,
                'factor_betas': factor_betas
            }
        
        except Exception as e:
            print(f"Error in factor correlation analysis: {str(e)}")
            return None

    def calculate_factor_betas(self, alpha, factors):
        """Calculate factor betas with improved data handling"""
        try:
            alpha = pd.Series(alpha) if not isinstance(alpha, pd.Series) else alpha
            factors = pd.DataFrame(factors) if not isinstance(factors, pd.DataFrame) else factors
            
            combined = pd.concat([alpha, factors], axis=1, join='inner')
            alpha_clean = combined.iloc[:, 0]
            factors_clean = combined.iloc[:, 1:]
            
            # handle inf values by replacing them with NaN
            alpha_clean = alpha_clean.replace([np.inf, -np.inf], np.nan)
            factors_clean = factors_clean.replace([np.inf, -np.inf], np.nan)
            
            # drop NaN values
            valid_data = pd.concat([alpha_clean, factors_clean], axis=1).dropna()
            
            if len(valid_data) < 10:
                print("\nInsufficient valid data points after cleaning")
                print(f"Original data points: {len(alpha)}")
                print(f"Valid data points: {len(valid_data)}")
                return None
            
            y = valid_data.iloc[:, 0]
            X = valid_data.iloc[:, 1:]
            X = sm.add_constant(X)
            
            # run regression
            model = sm.OLS(y, X).fit()
            
            return {
                'betas': model.params,
                'p_values': model.pvalues,
                'r_squared': model.rsquared,
                't_stats': model.tvalues
            }
        
        except Exception as e:
            print(f"\nError in factor beta calculation: {str(e)}")
            print(f"Alpha shape: {alpha.shape if hasattr(alpha, 'shape') else 'not array-like'}")
            print(f"Factors shape: {factors.shape if hasattr(factors, 'shape') else 'not array-like'}")
            return None

    def analyze_regime_impact(self, returns, regime_changes):
        """Analyze the impact of regime changes on returns"""
        try:
            regime_periods = pd.Series(regime_changes)
            
            # calculate returns statistics for different regimes
            regime_stats = {}
            for regime in regime_periods.unique():
                regime_returns = returns[regime_periods == regime]
                regime_stats[regime] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if len(regime_returns) > 0 else 0,
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis()
                }
            
            return regime_stats
        
        except Exception as e:
            print(f"Error in regime impact analysis: {str(e)}")
            return None

    def run_factor_analysis(self, start_date, end_date):
        """Run comprehensive factor analysis"""
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # fetch SPY data
            spy = yf.download('SPY', start=start_date, end=end_date + pd.Timedelta(days=1))
            market_returns = spy['Close'].pct_change()
            
            print(f"\nMarket data range: {spy.index.min()} to {spy.index.max()}")
            
            factor_results = {}
            for symbol in ['AAPL', 'GOOGL', 'NVDA', 'META', 'MSFT']:
                print(f"\nAnalyzing factors for {symbol}...")

                # Load stock data
                stock_path = f'data/{symbol}_daily.csv'
                if not os.path.exists(stock_path):
                    print(f"Warning: Data file not found for {symbol}")
                    continue
                    
                stock_data = pd.read_csv(stock_path)
                stock_data.set_index(pd.to_datetime(stock_data.iloc[:, 0]), inplace=True)
                stock_data = stock_data.iloc[:, 1:]
                stock_returns = stock_data['close'].pct_change()
                
                # Load sentiment data
                sentiment_path = f'data/{symbol}_sentiment.json'
                if os.path.exists(sentiment_path):
                    with open(sentiment_path, 'r') as f:
                        sentiment_data = json.load(f)
                    sentiment_score = sentiment_data.get('compound_score', 0)
                else:
                    print(f"Warning: Sentiment data not found for {symbol}")
                    sentiment_score = 0
                
                # Align the market returns with stock returns
                aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner')
                aligned_data.columns = ['stock_returns', 'market_returns']
                
                # Add real sentiment data (broadcast the sentiment score across all dates)
                aligned_data['sentiment'] = sentiment_score
                
                # Add regime data using the existing identify_regime method
                aligned_data['regime'] = aligned_data.index.map(lambda x: self.identify_regime(x))
                
                # Perform factor analysis
                factor_analysis = self.analyze_factor_correlations(
                    aligned_data['stock_returns'],  # Alpha
                    pd.DataFrame({
                        'Market_Returns': aligned_data['market_returns'],
                        'Sentiment': aligned_data['sentiment'],
                        'Regime': aligned_data['regime']
                    })
                )
                
                # Analyze regime impact
                regime_impact = self.analyze_regime_impact(
                    aligned_data['stock_returns'],
                    aligned_data['regime']
                )
                
                # Store results
                factor_results[symbol] = {
                    'factor_analysis': factor_analysis,
                    'regime_impact': regime_impact
                }
                
            return factor_results
            
        except Exception as e:
            print(f"Error in factor analysis: {str(e)}")
            traceback.print_exc()
            return None

    def format_factor_analysis_results(self, results):
        """Format factor analysis results into a readable DataFrame"""
        formatted_results = []
        
        for symbol, analysis in results.items():
            if analysis['factor_analysis'] is None:
                continue
            
            factor_corr = analysis['factor_analysis']['correlation_matrix']
            factor_betas = analysis['factor_analysis']['factor_betas']
            
            row = {
                'Symbol': symbol,
                'Alpha_Market_Corr': factor_corr.loc['Alpha', 'Market_Returns'],
                'Alpha_Sentiment_Corr': factor_corr.loc['Alpha', 'Sentiment'],
                'Alpha_Regime_Corr': factor_corr.loc['Alpha', 'Regime'],
                'Sentiment_Beta': factor_betas['betas']['Sentiment'],
                'Regime_Beta': factor_betas['betas']['Regime'],
                'Model_R_Squared': factor_betas['r_squared'],
                'Regime_Impact': str(analysis['regime_impact'])
            }
            formatted_results.append(row)
        
        return pd.DataFrame(formatted_results)

    def analyze_factor_correlations(self, alpha, factors_df):
        """
        Analyze correlations between alpha and various factors
        Note:
        alpha : pd.Series of alpha values
        factors_df : pd.DataFrame containing factor data
        Returns: Dictionary containing correlation matrix and factor betas
        """
        try:
            # create a combined DataFrame with alpha and factors
            factors_dict = {
                'Alpha': alpha,
                'Market_Returns': factors_df['Market_Returns'],
                'Sentiment': factors_df['Sentiment'],
                'Regime': factors_df['Regime']
            }
            
            factors_df = pd.DataFrame(factors_dict)
            
            corr_matrix = factors_df.corr()
            
            # find factor betas using regression
            factor_betas = self.calculate_factor_betas(alpha, factors_df)
            
            return {
                'correlation_matrix': corr_matrix,
                'factor_betas': factor_betas
            }
        
        except Exception as e:
            print(f"Error in factor correlation analysis: {str(e)}")
            return None

    def save_factor_analysis_results(self, results, output_path='data/factor_analysis_results.csv'):
        """
        Save factor analysis results to CSV file
        """
        try:
            data = []
            
            for symbol, analysis in results.items():
                if analysis['factor_analysis'] is None:
                    continue
                    
                corr_matrix = analysis['factor_analysis']['correlation_matrix']
                regime_impact = analysis['regime_impact']
                
                row = {
                    'Symbol': symbol,
                    'Market_Correlation': corr_matrix.loc['Alpha', 'Market_Returns'],
                    'Sentiment_Correlation': corr_matrix.loc['Alpha', 'Sentiment'],
                    'Regime_Correlation': corr_matrix.loc['Alpha', 'Regime'],
                    'Regime_0_Return': regime_impact[0]['mean_return'],
                    'Regime_0_Sharpe': regime_impact[0]['sharpe'],
                    'Regime_0_Volatility': regime_impact[0]['volatility'],
                    'Regime_1_Return': regime_impact[1]['mean_return'],
                    'Regime_1_Sharpe': regime_impact[1]['sharpe'],
                    'Regime_1_Volatility': regime_impact[1]['volatility']
                }
                data.append(row)
            
            # create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df = df.round(4)
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
            return df
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return None

    def run_backtest(self):
        """Run complete backtest using BacktestEngine with 2024 split into train/test periods"""
        try:
            # define periods with timezone
            train_start = pd.Timestamp('2024-01-01').tz_localize('America/New_York')
            train_end = pd.Timestamp('2024-06-30').tz_localize('America/New_York')
            test_start = pd.Timestamp('2024-07-01').tz_localize('America/New_York')
            test_end = pd.Timestamp('2024-12-31').tz_localize('America/New_York')
            
            print(f"\nTraining period: {train_start} to {train_end}")
            print(f"Testing period: {test_start} to {test_end}")
            
            # load price data and split into train/test
            prices_df = pd.DataFrame()
            train_prices = pd.DataFrame()
            
            for symbol in self.symbols:
                # read data with parse_dates and set index directly
                df = pd.read_csv(f'data/{symbol}_intraday.csv', 
                               parse_dates=['Datetime'],
                               index_col='Datetime')
                
                print(f"\nLoaded data for {symbol}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                print(f"Number of records: {len(df)}")
                
                # Split into train (H1 2024) and test (H2 2024)
                mask_train = (df.index >= train_start) & (df.index <= train_end)
                mask_test = (df.index >= test_start) & (df.index <= test_end)
                
                train_prices[symbol] = df.loc[mask_train, 'Close']
                prices_df[symbol] = df.loc[mask_test, 'Close']
            
            print(f"\nTraining data shape: {train_prices.shape}")
            print(f"Testing data shape: {prices_df.shape}")
            
            # learn relationships from H1 2024
            pairs_data = self.analyze_pairs(self.symbols, prices=train_prices)
            
            # generate signals for H2 2024
            signals_df = self.generate_signals_for_backtest(pairs_data, prices_df.index)
            
            # run backtest on H2 2024
            backtest = BacktestEngine(initial_capital=1_000_000)
            backtest.simulate(prices_df, signals_df)
            
            # calculate and print metrics
            metrics = backtest.calculate_metrics()
            print("\nBacktest Results (H2 2024):")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            backtest.plot_results()
            
            return metrics, backtest
            
        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            traceback.print_exc()
            return None, None

    def analyze_pairs(self, symbols, prices=None):
        """Enhanced pair analysis with sector consideration"""
        try:
            # define sector groups
            sectors = {
                'TECH_MEGA': ['AAPL', 'MSFT'],
                'TECH_SEARCH': ['GOOGL'],
                'TECH_SOCIAL': ['META'],
                'TECH_CHIP': ['NVDA']
            }
            
            if prices is None:
                prices = pd.DataFrame()
                for symbol in symbols:
                    df = pd.read_csv(f'data/{symbol}_intraday.csv', 
                                   parse_dates=['Datetime'],
                                   index_col='Datetime')
                    prices[symbol] = df['Close']
            
            pairs_data = {}

            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    # calculate sector score (higher if same sector)
                    sector_score = 0
                    for sector in sectors.values():
                        if sym1 in sector and sym2 in sector:
                            sector_score = 1
                            break
                    
                    print(f"\nAnalyzing pair: {sym1}-{sym2}")
                    
                    price1 = prices[sym1]
                    price2 = prices[sym2]
                    
                    # returns
                    ret1 = price1.pct_change().dropna()
                    ret2 = price2.pct_change().dropna()
                    
                    # calculate cointegration
                    score, pvalue, _ = coint(price1, price2)
                    
                    # calculate hedge ratio using rolling OLS
                    model = sm.OLS(price1, price2)
                    results = model.fit()
                    beta = results.params[0]
                    
                    # calculate spread
                    spread = price1 - beta * price2
                    
                    # calculate spread statistics
                    mean_spread = spread.mean()
                    std_spread = spread.std()
                    
                    # correlation on returns
                    correlation = ret1.corr(ret2)
                    
                    # volatility ratio
                    vol_ratio = ret1.std() / ret2.std()
                    
                    # store results with enhanced metrics
                    pairs_data[(sym1, sym2)] = {
                        'price_data': {
                            'beta': beta,
                            'mean': mean_spread,
                            'std': std_spread,
                            'correlation': correlation,
                            'coint_score': score,
                            'coint_pvalue': pvalue,
                            'vol_ratio': vol_ratio,
                            'sector_score': sector_score
                        }
                    }
                    
                    print(f"Cointegration p-value: {pvalue:.4f}")
                    print(f"Correlation: {correlation:.4f}")
                    print(f"Hedge ratio: {beta:.4f}")
                    print(f"Volatility ratio: {vol_ratio:.4f}")
                    print(f"Sector score: {sector_score}")
            
            return pairs_data
            
        except Exception as e:
            print(f"Error in analyze_pairs: {str(e)}")
            traceback.print_exc()
            return None

    def generate_signals_for_backtest(self, pairs_data, test_dates, z_threshold=1.8, 
                                    max_leverage=2.0, top_n_pairs=3):
        """Generate signals with proper dtype handling"""
        try:
            signals_df = pd.DataFrame(index=test_dates, columns=self.symbols, dtype=float)
            signals_df.fillna(0.0, inplace=True)
            
            prices = pd.DataFrame()
            returns = pd.DataFrame()
            for symbol in self.symbols:
                df = pd.read_csv(f'data/{symbol}_intraday.csv', 
                               parse_dates=['Datetime'],
                               index_col='Datetime')
                prices[symbol] = df['Close'].astype(float)  # Ensure float type
                returns[symbol] = prices[symbol].pct_change()
            prices = prices.loc[test_dates]
            returns = returns.loc[test_dates]
            
            # select top N pairs
            pair_scores = []
            for (sym1, sym2), data in pairs_data.items():
                # sharpe-like ratio
                spread_returns = returns[sym1] - data['price_data']['beta'] * returns[sym2]
                spread_returns = returns[sym1] - data['price_data']['beta'] * returns[sym2]
                sharpe = np.sqrt(252) * spread_returns.mean() / spread_returns.std()
                
                # Composite score
                score = (
                    sharpe * 0.4 +
                    (1 - data['price_data']['coint_pvalue']) * 0.4 +
                    abs(data['price_data']['correlation']) * 0.2
                )
                
                pair_scores.append((sym1, sym2, data, score))

            top_pairs = sorted(pair_scores, key=lambda x: x[3], reverse=True)[:top_n_pairs]
            
            print(f"\nSelected top {top_n_pairs} pairs:")
            for sym1, sym2, data, score in top_pairs:
                print(f"\n{sym1}-{sym2}:")
                print(f"Score: {score:.4f}")
                print(f"Cointegration p-value: {data['price_data']['coint_pvalue']:.4f}")
                print(f"Correlation: {data['price_data']['correlation']:.4f}")
            
            # track active positions across all pairs
            active_positions = {}  # {(sym1, sym2): {'position': 1/-1, 'size': float}}
            
            for sym1, sym2, data, score in top_pairs:
                beta = float(data['price_data']['beta'])  # Ensure float type
                spread = prices[sym1] - beta * prices[sym2]
                
                # calculate indicators
                window_short = 8
                window_med = 21
                
                rolling_mean_short = spread.rolling(window=window_short).mean()
                rolling_std_short = spread.rolling(window=window_short).std()
                rolling_mean_med = spread.rolling(window=window_med).mean()
                
                for i, date in enumerate(test_dates):
                    try:
                        if i < window_med:
                            continue
                        
                        z_score_short = (spread[date] - rolling_mean_short[date]) / rolling_std_short[date]
                        trend = 1 if spread[date] > rolling_mean_med[date] else -1
                        
                        # available leverage
                        current_exposure = abs(signals_df.loc[date]).sum()
                        available_leverage = max(0, max_leverage - current_exposure)
                        
                        # position sizing with leverage
                        base_size = min(1.0, abs(z_score_short/z_threshold))
                        leverage_size = base_size * (available_leverage / top_n_pairs)
                        
                        pair_key = (sym1, sym2)
                        
                        if pair_key not in active_positions:
                            if z_score_short < -z_threshold and trend < 0:
                                signals_df.loc[date, sym1] = leverage_size
                                signals_df.loc[date, sym2] = -leverage_size * beta
                                active_positions[pair_key] = {'position': 1, 'size': leverage_size}
                                print(f"Long signal for {sym1}-{sym2} at {date}, size: {leverage_size:.2f}")
                            
                            elif z_score_short > z_threshold and trend > 0:
                                signals_df.loc[date, sym1] = -leverage_size
                                signals_df.loc[date, sym2] = leverage_size * beta
                                active_positions[pair_key] = {'position': -1, 'size': leverage_size}
                                print(f"Short signal for {sym1}-{sym2} at {date}, size: {leverage_size:.2f}")
                        
                        else:  
                            if (abs(z_score_short) < 0.3 or  # profit taking
                                (z_score_short > 0 and active_positions[pair_key]['position'] > 0) or
                                (z_score_short < 0 and active_positions[pair_key]['position'] < 0)):
                                
                                signals_df.loc[date, sym1] = 0
                                signals_df.loc[date, sym2] = 0
                                del active_positions[pair_key]
                                print(f"Exit signal for {sym1}-{sym2} at {date}")
                    
                    except Exception as e:
                        print(f"Error processing date {date}: {str(e)}")
                        continue
            
            # verify signal generation
            max_leverage_used = signals_df.abs().sum(axis=1).max()
            non_zero_signals = (signals_df != 0).sum().sum()
            
            print(f"\nStrategy Statistics:")
            print(f"Total non-zero signals: {non_zero_signals}")
            print(f"Maximum leverage used: {max_leverage_used:.2f}x")
            
            return signals_df
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            traceback.print_exc()
            return None

    def optimize_parameters(self, train_prices, test_prices, param_ranges=None):
        """
        Optimize strategy parameters using grid search
        """
        if param_ranges is None:
            param_ranges = {
                'z_threshold': [1.0, 1.5, 2.0, 2.5, 3.0],
                'min_coint_pvalue': [0.01, 0.05, 0.1],
                'min_correlation': [0.5, 0.6, 0.7, 0.8],
                'max_pairs': [3, 4, 5, 6]
            }
        
        best_sharpe = -np.inf
        best_params = None
        best_metrics = None
        results = []
        
        print("\nOptimizing Parameters...")
        total_combinations = np.prod([len(v) for v in param_ranges.values()])
        current = 0
        
        for z_threshold in param_ranges['z_threshold']:
            for min_coint_pvalue in param_ranges['min_coint_pvalue']:
                for min_correlation in param_ranges['min_correlation']:
                    for max_pairs in param_ranges['max_pairs']:
                        current += 1
                        print(f"\nTesting combination {current}/{total_combinations}")
                        print(f"z_threshold: {z_threshold}")
                        print(f"min_coint_pvalue: {min_coint_pvalue}")
                        print(f"min_correlation: {min_correlation}")
                        print(f"max_pairs: {max_pairs}")
                        
                        try:    
                            pairs_data = self.analyze_pairs(self.symbols, prices=train_prices)
                            
                            # filter pairs based on current parameters
                            filtered_pairs = {}
                            for (sym1, sym2), data in pairs_data.items():
                                if (data['price_data']['coint_pvalue'] < min_coint_pvalue and
                                    data['price_data']['correlation'] > min_correlation):
                                    filtered_pairs[(sym1, sym2)] = data
                            
                            # take only best pairs up to max_pairs
                            sorted_pairs = sorted(filtered_pairs.items(), 
                                               key=lambda x: x[1]['price_data']['correlation'],
                                               reverse=True)[:max_pairs]
                            filtered_pairs = dict(sorted_pairs)
                            
                            signals_df = self.generate_signals_for_backtest(
                                filtered_pairs, 
                                test_prices.index,
                                z_threshold=z_threshold
                            )
                            
                            # Run backtest
                            backtest = BacktestEngine(initial_capital=1_000_000)
                            backtest.simulate(test_prices, signals_df)
                            metrics = backtest.calculate_metrics()
                            
                            results.append({
                                'z_threshold': z_threshold,
                                'min_coint_pvalue': min_coint_pvalue,
                                'min_correlation': min_correlation,
                                'max_pairs': max_pairs,
                                'sharpe_ratio': metrics['sharpe_ratio'],
                                'total_return': metrics['total_return'],
                                'max_drawdown': metrics['max_drawdown']
                            })
                            
                            # Update best parameters if we found better Sharpe ratio
                            if metrics['sharpe_ratio'] > best_sharpe:
                                best_sharpe = metrics['sharpe_ratio']
                                best_params = {
                                    'z_threshold': z_threshold,
                                    'min_coint_pvalue': min_coint_pvalue,
                                    'min_correlation': min_correlation,
                                    'max_pairs': max_pairs
                                }
                                best_metrics = metrics
                                
                                print("\nNew best parameters found!")
                                print(f"Sharpe Ratio: {best_sharpe:.2f}")
                                print(f"Total Return: {metrics['total_return']:.2f}%")
                                print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
                        
                        except Exception as e:
                            print(f"Error testing combination: {str(e)}")
                            continue
        
        results_df = pd.DataFrame(results)
        
        return best_params, best_metrics, results_df
    
    def calculate_hurst_exponent(self, series, lags=range(2, 100)):
        """
        Calculate the Hurst exponent of a time series.
        
        The Hurst exponent (H) indicates:
        - H < 0.5: Mean-reverting series (good for pairs trading)
        - H = 0.5: Random walk
        - H > 0.5: Trending series
        
        Parameters:
        - series: Time series data
        - lags: Range of lag values to use
        
        Returns:
        - Hurst exponent value
        """
        try:
            series = np.array(series)
            returns = np.log(series[1:]) - np.log(series[:-1])
            
            tau = []  # Vector of tau values (time scale)
            lagvec = []  # Vector of lag values
            
            for lag in lags:
                pp = np.subtract(series[lag:], series[:-lag])
                
                # Calculate the variance of the difference vector
                tau.append(np.sqrt(np.std(pp)))
                lagvec.append(lag)
            
            # linear fit to double-log graph to get Hurst exponent
            #  use log of tau vs. log of lag
            lagvec = np.log(lagvec)
            tau = np.log(tau)
            
            # calculate Hurst exponent using linear regression
            poly = np.polyfit(lagvec, tau, 1)
            hurst = poly[0]
            
            # add some validation
            if not 0 < hurst < 1:
                print(f"Warning: Unusual Hurst value ({hurst:.4f}) - might indicate non-stationary data")
            
            return hurst
            
        except Exception as e:
                print(f"Error calculating Hurst exponent: {str(e)}")
                return None

    def analyze_pairs_with_ml(self, symbols, prices=None):
        """
        Analyze pairs using machine learning to predict successful pairs
        """
        try:
            
            if prices is None:
                prices = pd.DataFrame()
                for symbol in symbols:
                    df = pd.read_csv(f'data/{symbol}_intraday.csv', 
                                   parse_dates=['Datetime'],
                                   index_col='Datetime')
                    prices[symbol] = df['Close']
            
            # features for each pair
            pair_features = []
            pair_labels = []  # 1 for profitable pairs, 0 for unprofitable
            pairs_data = {}
            
            print("\nAnalyzing pairs with machine learning...")
            
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    print(f"\nAnalyzing pair: {sym1}-{sym2}")
                    
                    price1 = prices[sym1]
                    price2 = prices[sym2]
                    ret1 = price1.pct_change().dropna()
                    ret2 = price2.pct_change().dropna()
                    
                    score, pvalue, _ = coint(price1, price2)
                    correlation = ret1.corr(ret2)
                    
                    beta = sm.OLS(price1.values, price2.values).fit().params[0]
                    spread = price1 - beta * price2
                    
                    vol_ratio = ret1.std() / ret2.std()
                    spread_vol = spread.std() / spread.mean() if spread.mean() != 0 else 0
                    
                    hurst_exponent = self.calculate_hurst_exponent(spread)
                    if hurst_exponent is None:
                        hurst_exponent = 0.5  # default to random walk if calculation fails

                    rolling_corr = ret1.rolling(20).corr(ret2)
                    corr_stability = rolling_corr.std()
                    
                    features = {
                        'coint_score': score,
                        'coint_pvalue': pvalue,
                        'correlation': correlation,
                        'beta': beta,
                        'vol_ratio': vol_ratio,
                        'spread_vol': spread_vol,
                        'hurst_exponent': hurst_exponent,
                        'corr_stability': corr_stability
                    }

                    pairs_data[(sym1, sym2)] = {
                        'features': features,
                        'price_data': {
                            'beta': beta,
                            'mean': spread.mean(),
                            'std': spread.std(),
                            'correlation': correlation,
                            'coint_score': score,
                            'coint_pvalue': pvalue
                        }
                    }
                    
                    # add to training data
                    pair_features.append(list(features.values()))
                    
                    spread_returns = spread.pct_change().dropna()
                    sharpe = np.sqrt(252) * spread_returns.mean() / spread_returns.std()
                    pair_labels.append(1 if sharpe > 0.5 else 0)
            
            # train
            X = np.array(pair_features)
            y = np.array(pair_labels)
            
            # Standardize 
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train 
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Get pair scores
            pair_scores = model.predict_proba(X_scaled)[:, 1]
            
            # Add ML scores to pairs_data
            for (pair, score) in zip(pairs_data.keys(), pair_scores):
                pairs_data[pair]['ml_score'] = score
                print(f"\n{pair[0]}-{pair[1]} ML Score: {score:.4f}")
            
            return pairs_data
            
        except Exception as e:
            print(f"Error in analyze_pairs_with_ml: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_hurst_exponent(self, series, lags=range(2, 100)):
        """Calculate Hurst exponent with improved error handling"""
        try:
            # Convert series to numpy array and ensure positive values
            series = np.array(series)
            if len(series) < max(lags):
                return 0.5  # Return default if series is too short
            
            tau = []
            lagvec = []
            
            # Calculate tau for each lag
            for lag in lags:
                diff = np.subtract(series[lag:], series[:-lag])
                # Only proceed if we have valid differences
                if len(diff) > 0:
                    std = np.nanstd(diff)
                    if std > 0:
                        tau.append(np.sqrt(std))
                        lagvec.append(lag)
            
            if len(tau) < 2:
                return 0.5
            
            tau = np.array(tau)
            lagvec = np.array(lagvec)
            
            eps = 1e-10
            log_tau = np.log(tau + eps)
            log_lag = np.log(lagvec + eps)
            
            # linear regression
            poly = np.polyfit(log_lag, log_tau, 1)
            hurst = poly[0]
            
            # Hurst exponent
            if not 0 <= hurst <= 1:
                return 0.5
            
            return hurst
            
        except Exception as e:
            print(f"Error in Hurst calculation: {str(e)}")
            return 0.5

if __name__ == "__main__":
    # Initialize strategy
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
    strategy = StatArbStrategy(symbols)
    
    print("\nRunning Leveraged Multi-Pair Statistical Arbitrage Strategy")
    print("======================================================")
    
    # Load all price data
    prices = pd.DataFrame()
    for symbol in symbols:
        df = pd.read_csv(f'data/{symbol}_intraday.csv', 
                        parse_dates=['Datetime'],
                        index_col='Datetime')
        prices[symbol] = df['Close']
    
    # split into train and test with validation
    train_end = pd.Timestamp('2024-06-30').tz_localize('America/New_York')
    train_prices = prices[prices.index <= train_end]
    test_prices = prices[prices.index > train_end]
    
    # Continue with strategy...
    print("\nAnalyzing pairs with machine learning...")
    pairs_data = strategy.analyze_pairs_with_ml(symbols, prices=train_prices)
    
    if pairs_data:
        print("\nGenerating signals with leverage...")
        signals_df = strategy.generate_signals_for_backtest(
            pairs_data,
            test_prices.index,
            z_threshold=1.8,
            max_leverage=2.0,
            top_n_pairs=3
        )
        
        if signals_df is not None:
            print("\nRunning backtest...")
            backtest = BacktestEngine(initial_capital=1_000_000)
            backtest.simulate(test_prices, signals_df)
            metrics = backtest.calculate_metrics()
            
            print("\nBacktest Results:")
            print("================")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if 'return' in metric or 'drawdown' in metric:
                        print(f"{metric}: {value:.2f}%")
                    else:
                        print(f"{metric}: {value:.2f}")
            
            # Add monthly returns analysis
            print("\nMonthly Returns Analysis:")
            print("=======================")
            monthly_returns = backtest.get_monthly_returns()
            if monthly_returns is not None:
                print("\nMonthly Returns:")
                for month, ret in monthly_returns.items():
                    print(f"{month}: {ret*100:.2f}%")
                print(f"\nPositive months: {sum(r > 0 for r in monthly_returns.values())}")
                print(f"Negative months: {sum(r < 0 for r in monthly_returns.values())}")
            
            print("\nTrade Statistics:")
            print("================")
            print(f"Total trades: {backtest.get_total_trades()}")
            print(f"Win rate: {backtest.get_win_rate()*100:.2f}%")
            print(f"Average win: {backtest.get_average_win()*100:.2f}%")
            print(f"Average loss: {backtest.get_average_loss()*100:.2f}%")
            print(f"Profit factor: {backtest.get_profit_factor():.2f}")
            
            try:
                backtest.plot_results()
            except Exception as e:
                print(f"Error plotting results: {str(e)}")
        else:
            print("Error: Failed to generate signals")
    else:
        print("Error: Failed to analyze pairs")