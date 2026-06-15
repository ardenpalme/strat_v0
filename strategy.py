from vectorbtpro import *
import os
import sys

from zoneinfo import ZoneInfo
from datetime import datetime
from dateutil.relativedelta import relativedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gc
from ipywidgets import IntProgress
from IPython.display import display
import ipywidgets as widgets

'''
=================================
 Average Directional Index (ADX)
=================================
'''

@njit
def get_atr(high, low, close, period):
    close_shifted_1 = np.roll(close, 1)
    close_shifted_1[0] = np.nan
    
    tr0 = np.abs(high - low)
    tr1 = np.abs(high - close_shifted_1)
    tr2 = np.abs(low - close_shifted_1)
    tr = np.maximum(np.maximum(tr0, tr1), tr2)
    atr = vbt.nb.wwm_mean_1d_nb(tr, period)  

    alpha = 1 / period
    weights = (1 - alpha) ** np.arange(period - 1, -1, -1)  
    weights /= weights.sum()
    ewma = np.convolve(tr, weights, mode='full')[:len(close)] 
    ewma[:period - 1] = np.nan  

    return atr

@njit
def get_dm(high, low):
    high_shifted_1 = np.roll(high, 1)
    high_shifted_1[0] = np.nan
    plus_dm_cmp = high_shifted_1 - high
    plus_dm = np.where(plus_dm_cmp > 0, plus_dm_cmp, 0)
    
    low_shifted_1 = np.roll(low, 1)
    low_shifted_1[0] = np.nan
    minus_dm_cmp = low_shifted_1 - low
    minus_dm = np.where(minus_dm_cmp > 0, minus_dm_cmp, 0)
    return plus_dm, minus_dm

@njit
def get_adx_di(high, low, close, di_ma_len, adx_ma_len):
    plus_dm, minus_dm = get_dm(high, low)
    atr = get_atr(high, low, close, di_ma_len)
     
    conv_arr = np.ones(di_ma_len, dtype=np.float64)
    plus_dm_sma = np.convolve(plus_dm, conv_arr, 'full')[:-(di_ma_len-1)] / di_ma_len
    minus_dm_sma = np.convolve(minus_dm, conv_arr, 'full')[:-(di_ma_len-1)] / di_ma_len
    
    plus_di = 100 * (plus_dm_sma / atr)
    minus_di = 100 * (minus_dm_sma / atr)
    res = np.abs(plus_di - minus_di) / plus_di + minus_di

    conv_arr = np.ones(adx_ma_len, dtype=np.float64)
    adx = np.convolve(res, conv_arr, 'full')[:-(adx_ma_len-1)] / adx_ma_len
    
    return adx, plus_di, minus_di


ADX = vbt.IF(
    class_name = 'Average Directional Momentum Index',
    short_name = 'ADX',
    input_names = ['high', 'low', 'close'], 
    param_names = ['di_ma_len', 'adx_ma_len'],
    output_names = ['adx', 'plus_di', 'minus_di']
).with_apply_func(
    get_adx_di,
    takes_1d=True,
    di_ma_len=14,
    adx_ma_len=20
)

class ADX(ADX):
    def plot(self,
             column=None,
             plus_di_kwargs=None,
             minus_di_kwargs=None,
             adx_kwargs=None,
             fig=None,
             **layout_kwargs):
        
        plus_di_kwargs = plus_di_kwargs if plus_di_kwargs else {}
        minus_di_kwargs = minus_di_kwargs if minus_di_kwargs else {}
        adx_kwargs = adx_kwargs if adx_kwargs else {}

        if isinstance(self.adx.columns, pd.MultiIndex):
            plus_di = self.plus_di.xs(column, level='symbol', axis=1).squeeze().dropna()
            minus_di = self.minus_di.xs(column, level='symbol', axis=1).squeeze().dropna()
            adx = self.adx.xs(column, level='symbol', axis=1).squeeze().dropna()
        else:
            plus_di = self.plus_di[column].dropna()
            minus_di = self.minus_di[column].dropna()
            adx = self.adx[column].dropna()

        fig.add_trace(
            go.Scatter(x=adx.index,
                       y=adx.values, 
                       name="ADX",
                       **adx_kwargs),
            **layout_kwargs
        )
        fig.add_trace(
            go.Scatter(x=plus_di.index,
                       y=plus_di.values, 
                       name="+DI",
                       **plus_di_kwargs),
            **layout_kwargs
        )
        fig.add_trace(
            go.Scatter(x=minus_di.index,
                       y=minus_di.values, 
                       name="-DI",
                       **minus_di_kwargs),
            **layout_kwargs
        )

'''
==================================
 Kaufmann Adaptive Moving Average
==================================
'''

@njit
def get_sc(close, period, fast_ma_len, slow_ma_len):
    fastest_sc = 2 / (fast_ma_len + 1)
    slowest_sc = 2 / (slow_ma_len + 1)

    price_shifted_1 = np.roll(close,1)
    price_shifted_1[0] = np.nan
    diff_close = np.abs(close - price_shifted_1)

    conv_arr = np.ones(period,dtype=np.float64)
    denominator = np.convolve(diff_close, conv_arr, 'full')[:-(period-1)] 

    price_shifted_period = np.roll(close,period)
    price_shifted_period[0:period] = np.nan
    
    ER = np.abs(close - price_shifted_period) / denominator
    sc = (ER * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    return sc

@njit
def get_kama(close, period, fast_ma_len, slow_ma_len):
    sc = get_sc(close, period, fast_ma_len, slow_ma_len)
    kama = np.full((close.shape[0]), np.nan)
    for i in range(1,close.shape[0]):  
        if not np.isnan(sc[i]):
            if np.isnan(kama[i-1]):
                kama[i-1] = close[0]
            kama[i] = kama[i-1] + (sc[i] * (close[i] - kama[i-1]))
        else:
            kama[i] = np.nan
    return kama


Kama = vbt.IF(
    class_name = 'Kaufmann Adaptive Moving Average',
    short_name = 'KAMA',
    input_names = ['close'], 
    param_names = ['period', 'fast_ma_len', 'slow_ma_len'],
    output_names = ['kama']
).with_apply_func(
    get_kama,
    takes_1d=True,
    period=10,
    fast_ma_len=2,
    slow_ma_len=30
)

class KAMA(Kama):
    def plot(self,
             column=None,
             kama_kwargs=None,
             fig=None,
             **layout_kwargs
            ):
        kama_kwargs = kama_kwargs if kama_kwargs else {}
        if isinstance(self.kama.columns, pd.MultiIndex):
            kama = self.kama.xs(column, level='symbol', axis=1).squeeze().dropna()
        else:
            kama = self.kama[column].dropna()
        
        fig.add_trace(
            go.Scatter(x=kama.index,
                       y=kama.values, 
                       **kama_kwargs),
            **layout_kwargs
        )

'''
=================================
Signal Analysis Helper functions
=================================
'''



@njit(nogil=True)
def generate_signals(kama_fast, kama_slow, adx, adx_thrsh):
    "Takes only one asset column, iterate in caller"
    entries = vbt.nb.crossed_above_1d_nb(kama_fast, kama_slow) 
    exits = vbt.nb.crossed_below_1d_nb(kama_fast, kama_slow)

    adx_filter = np.where(adx > adx_thrsh, 1.0, 0.0)
    entries = entries * adx_filter
    # exits = exits * adx_filter # Do not supress exits on weak trends

    started = False  
    long = False     
    short = False    
    new_entries = np.full(kama_fast.shape[0], False, dtype=np.bool_)
    new_exits = np.full(kama_fast.shape[0], False, dtype=np.bool_)

    for i in range(entries.size):
        if not started:
            if exits[i]:  
                started = True
                long = False
                short = True
                new_exits[i] = True
            continue  
    
        if short and entries[i]:  
            short = False
            long = True
            new_entries[i] = True
        elif long and exits[i]:  
            long = False
            short = True
            new_exits[i] = True
    
    return (new_entries, new_exits)

def get_signals_analysis(kama_fast, kama_slow, adx, adx_thrsh):
    kama_values = kama_fast.kama.values
    dim = kama_values.shape 
    
    long_entries = np.empty(dim, dtype=np.bool_)
    long_exits = np.empty(dim, dtype=np.bool_)
    
    for col in range(dim[1]):
        signals = generate_signals(kama_fast.kama.values[:, col], 
                                   kama_slow.kama.values[:, col], 
                                   adx.adx.values[:, col],
                                   adx_thrsh)
        
        long_entries[:, col] = signals[0]
        long_exits[:, col] = signals[1]
    
    return (long_entries, long_exits)

dfl_params = dict(
    kama_fast_period = 5,
    kama_slow_period = 10,
    di_ma_len = 14,
    adx_ma_len = 20,
    adx_thrsh = 12
)
        
def plot_strategy(market_data, symbol, ind_params=dfl_params):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    ohlc_trace = market_data.plot(symbol=symbol, plot_volume=False, ohlc_trace_kwargs=dict(opacity=0.7)).data[0]
    ohlc_trace.name = symbol
    fig.add_trace(ohlc_trace, row=1, col=1)

    # compute indicators given parameters
    kama_fast = KAMA.run(market_data.get('Close'), period=ind_params['kama_fast_period'])
    kama_slow = KAMA.run(market_data.get('Close'), period=ind_params['kama_slow_period'])
    adx = ADX.run(market_data.get('High'), 
        market_data.get('Low'), 
        market_data.get('Close'), 
        di_ma_len=ind_params['di_ma_len'], 
        adx_ma_len=ind_params['adx_ma_len'],
    )
    
    kama_fast.plot(column=symbol, 
        kama_kwargs=dict(line_color='yellow', name=f'KAMA ({ind_params['kama_fast_period']})'),
        fig=fig, row=1, col=1
    )
    
    kama_slow.plot(column=symbol, 
        kama_kwargs=dict(line_color='blue', name=f'KAMA ({ind_params['kama_slow_period']})'),
        fig=fig, row=1, col=1
    )
    
    adx.plot(column=symbol, 
        adx_kwargs=dict(line_color='blue'),
        plus_di_kwargs=dict(line_color='orange'),
        minus_di_kwargs=dict(line_color='limegreen'),
        fig=fig, row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=adx.adx.index,
            y=[ind_params['adx_thrsh']]*len(adx.adx),
            mode='lines',
            name="ADX Threshold",
            line=dict(color='red', dash='dash')
        ),
        row=2, col=1
    )

    signals = get_signals_analysis(kama_fast, kama_slow, adx, ind_params['adx_thrsh'])
    entries = pd.DataFrame(signals[0], columns=market_data.columns, index=market_data.index)[symbol]
    exits = pd.DataFrame(signals[1], columns=market_data.columns, index=market_data.index)[symbol]
    entries.vbt.signals.plot_as_entries(y=market_data.get("Close", symbol), fig=fig)
    exits.vbt.signals.plot_as_exits(y=market_data.get("Close", symbol), fig=fig)

    fig.update(layout_xaxis_rangeslider_visible=False)

    return entries, exits, fig


@njit(nogil=True)
def pipeline_nb(high, low, close, 
                kama_fast_period=np.asarray([5]), 
                kama_slow_period=np.asarray([10]), 
                di_ma_len=np.asarray([14]),
                adx_ma_len=np.asarray([20]),
                adx_thrsh=np.asarray([9]),
                ann_factor=365):

    num_metrics = 4
    num_param_comb = kama_fast_period.size
    metrics = np.empty((num_param_comb * close.shape[1], num_metrics), dtype=np.float64)
    
    long_entries = np.empty(close.shape, dtype=np.bool)
    long_exits = np.empty(close.shape, dtype=np.bool)
    group_lens = np.full(close.shape[1], 1)
    k = 0

    for i in range(num_param_comb):
        for col in range(close.shape[1]): # for each asset
 
            kama_fast = get_kama(close[:, col], kama_fast_period[i], 2, 30)
            kama_slow = get_kama(close[:, col], kama_slow_period[i], 2, 30)
            adx_line, _, _ = get_adx_di(high[:, col], low[:, col], close[:, col], di_ma_len[i], adx_ma_len[i])
            
            signals = generate_signals(kama_fast, kama_slow, adx_line, adx_thrsh[i])
        
            long_entries[:, col] = signals[0]
            long_exits[:, col] = signals[1]

        sim_out = vbt.pf_nb.from_signals_nb(
            target_shape=close.shape,
            group_lens=group_lens,
            close=close,
            long_entries=long_entries,
            long_exits=long_exits,
            save_returns=True
        )
        num_trades = long_exits.sum(axis=0)
        
        returns = sim_out.in_outputs.returns
        sharpe = vbt.ret_nb.sharpe_ratio_nb(returns, ann_factor, ddof=1)
        sortino = vbt.ret_nb.sortino_ratio_nb(returns, ann_factor)
        dd = vbt.ret_nb.max_drawdown_nb(returns, ann_factor)

        metrics_vec = np.column_stack((sharpe, sortino, dd, num_trades))
        metrics_vec_len = len(metrics_vec)
        
        metrics[k:k + metrics_vec_len] = metrics_vec
        k += metrics_vec_len
            
    return metrics


def merge_func(arrs, ann_args, input_columns):
    arr = np.concatenate(arrs)

    param_idx = vbt.stack_indexes((
        pd.Index(ann_args['kama_fast_period']['value'], name='kama_fast_period'),
        pd.Index(ann_args['kama_slow_period']['value'], name='kama_slow_period'),
        pd.Index(ann_args['di_ma_len']['value'], name='di_ma_len'),
        pd.Index(ann_args['adx_ma_len']['value'], name='adx_ma_len'),
        pd.Index(ann_args['adx_thrsh']['value'], name='adx_thrsh')
    ))

    idx = vbt.combine_indexes((
        param_idx,
        input_columns
    ))
    
    return pd.DataFrame(arr, columns=['Sharpe', 'Sortino', 'DD', 'Num Trades'], index=idx)

nb_chunked = vbt.chunked(
    size=vbt.ArraySizer(arg_query='kama_fast_period', axis=0),
    arg_take_spec=dict(
        high=None,
        low=None,
        close=None,
        kama_fast_period=vbt.ArraySlicer(axis=0), 
        kama_slow_period=vbt.ArraySlicer(axis=0), 
        adx_ma_len=vbt.ArraySlicer(axis=0), 
        di_ma_len=vbt.ArraySlicer(axis=0), 
        adx_thrsh=vbt.ArraySlicer(axis=0), 
        ann_factor=None
    ),
    merge_func=merge_func,
    merge_kwargs=dict(
        # simply concatenate results from all chunks
        ann_args=vbt.Rep("ann_args") 
    )
)

def cross_validate(market_data, splitter, chunked_pipeline_nb, param_products):
    kama_fast_period_prod = np.asarray(param_products[0])
    kama_slow_period_prod = np.asarray(param_products[1])
    di_ma_len_prod = np.asarray(param_products[2])
    adx_ma_len_prod = np.asarray(param_products[3])
    adx_thrsh_prod = np.asarray(param_products[4])

    ticker_cols = market_data.columns
    results_rows = []
    max_dd = -0.4
    min_trades = 10
    logging = False

    prog_bar = IntProgress(min=0, )
    prog_bar = widgets.IntProgress(
        value=0, 
        min=0, 
        max=len(ticker_cols) * splitter.splits_arr.shape[0],
        layout=widgets.Layout(width='400px') # Adjust '400px' to your preference
    )
    prog_bar.style.description_width = 'initial' # space for description
    prog_label = widgets.Label('')
    prog_bar_ui = widgets.HBox([prog_bar, prog_label])
    display(prog_bar_ui)

    for symbol in ticker_cols:
        for split in splitter.splits_arr:
            is_split = split[0]
            oos_split = split[1]
            is_data = market_data[is_split]
            oos_data = market_data[oos_split]
            
            oos_period_str = f"Test: {oos_data.index[0].strftime('%Y-%m-%d')} to {oos_data.index[-1].strftime('%Y-%m-%d')}"
            is_period_str = f"Train: {is_data.index[0].strftime('%Y-%m-%d')} to {is_data.index[-1].strftime('%Y-%m-%d')}"
            
            ## IS Testing
            prog_label.value = f'{is_period_str} for {symbol}'
            
            res = chunked_pipeline_nb(
                is_data.get('High').values, is_data.get('Low').values, is_data.get('Close').values,
                kama_fast_period=kama_fast_period_prod, 
                kama_slow_period=kama_slow_period_prod, 
                adx_ma_len=adx_ma_len_prod,
                adx_thrsh=adx_thrsh_prod,
                di_ma_len=di_ma_len_prod,
                ann_factor=365,
                _execute_kwargs=dict(engine="dask"),
                _merge_kwargs=dict(input_columns=ticker_cols)
            )
            res = res.dropna()
            res = res[res['DD'] > max_dd] 
            res = res[res['Num Trades'] > min_trades] 
            if(res.shape[0] == 0):
                if(logging):
                    print(f"{is_period_str} for {symbol}: No strategy with: MDD > {max_dd} and Num Trades > {min_trades}")
                continue
            best_is_res = res.sort_values(by=['Sortino'], ascending=False)

            best_target_rows = best_is_res.loc[pd.IndexSlice[:, :, :, :, :, symbol], :]
            best_target_rows_indices = best_is_res.index.get_indexer(best_target_rows.index)
            best_target_row = best_is_res.iloc[best_target_rows_indices[0]]
            
            del res
            gc.collect()

            ## OOS Testing
            prog_label.value = f'{oos_period_str} for {symbol}'

            best_params = best_target_row.name
            kama_fast = KAMA.run(oos_data.get('Close'), period=best_params[0])
            kama_slow = KAMA.run(oos_data.get('Close'), period=best_params[1])
            adx = ADX.run(oos_data.get('High'), oos_data.get('Low'), oos_data.get('Close'), 
                          di_ma_len=best_params[2],
                          adx_ma_len=best_params[3])
            adx_thrsh= best_params[4]
            
            signals = get_signals_analysis(kama_fast, kama_slow, adx, adx_thrsh)
            pf = vbt.Portfolio.from_signals(
                oos_data.get('Close'),
                entries=signals[0],
                exits=signals[1]
            )
            stats = pf.stats(metrics=["sharpe_ratio", "sortino_ratio", "max_dd_duration", "max_dd", "total_trades", "total_return"], column="BTC-USD")
            
            if(logging):
                print(f"{is_period_str} for {symbol} yields: STR: {best_target_row['Sortino']:.2f}, MDD: {best_target_row['DD']*-100:.2f}%")
                print(f"{oos_period_str} for {symbol} yields: STR: {stats["Sortino Ratio"]:.2f}, MDD: {stats["Max Drawdown [%]"]:.2f}%")
                print("===========================================================\n")

            stats['oos_period_start'] = oos_data.index[0]
            stats['oos_period_end'] = oos_data.index[-1]
            
            stats['is_period_start'] = is_data.index[0]
            stats['is_period_end'] = is_data.index[-1]
            stats['train_best_params'] = best_params
            stats['train_best_sortino_ratio'] = best_target_row['Sortino']
            stats['symbol'] = symbol
            
            results_rows.append(stats)
            prog_bar.value += 1

    cv_results = pd.DataFrame(results_rows)
    return cv_results

