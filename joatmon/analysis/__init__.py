import numpy as np
import pandas as pd


# some data is pandas.series, some data is numpy array
# need to make sure all of them are the same type
def calculate_sma(df, window=3, inplace=True):
    """Simple Moving Average (SMA)"""
    if not inplace:
        df = df.copy()

    df['sma'] = df['close'].rolling(window=window).mean()
    return df


def calculate_ema(df, window=3, inplace=True):
    """Exponential Moving Average (EMA)"""
    if not inplace:
        df = df.copy()

    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()

    ema = np.convolve(df['close'], weights, mode='full')[:len(df)]
    ema[:window] = ema[window]
    df['ema'] = ema
    return df


def calculate_bollinger_bands(df, window=3, num_std=2, inplace=True):
    """Bollinger Bands"""
    if not inplace:
        df = df.copy()

    sma = calculate_sma(df, window, inplace=False)
    df['bollinger upper'] = sma['sma'] + (df['close'].rolling(window=window).std() * num_std)
    df['bollinger lower'] = sma['sma'] - (df['close'].rolling(window=window).std() * num_std)
    return df


def calculate_rsi(df, window=3, inplace=True):
    """Relative Strength Index (RSI)"""
    if not inplace:
        df = df.copy()

    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    return df


def calculate_macd(df, short_window=12, long_window=26, signal_window=9, inplace=True):
    """Moving Average Convergence Divergence (MACD)"""
    if not inplace:
        df = df.copy()

    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    df['macd_line'] = macd_line
    df['signal_line'] = signal_line
    df['macd_histogram'] = macd_histogram
    return df


def calculate_stochastic_oscillator(df, k_window=5, d_window=3, inplace=True):
    """Stochastic Oscillator"""
    if not inplace:
        df = df.copy()

    lowest_low = df['low'].rolling(window=k_window).min()
    highest_high = df['high'].rolling(window=k_window).max()
    df['%K'] = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    df['%D'] = df['%K'].rolling(window=d_window).mean()
    return df


def calculate_atr(df, window=14, inplace=True):
    """Average True Range (ATR)"""
    if not inplace:
        df = df.copy()

    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.DataFrame()
    tr['HL'] = high - low
    tr['HC'] = abs(high - close.shift())
    tr['LC'] = abs(low - close.shift())

    tr['TrueRange'] = tr.max(axis=1)

    atr = tr['TrueRange'].rolling(window=window).mean()

    df['atr'] = atr

    return df


def calculate_williams_r(df, window=14, inplace=True):
    """Williams %R"""
    if not inplace:
        df = df.copy()

    high = df['high']
    low = df['low']
    close = df['close']

    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()

    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100

    df['williams_r'] = williams_r

    return df


def calculate_cmf(df, window=20, inplace=True):
    """Chaikin Money Flow (CMF)"""
    if not inplace:
        df = df.copy()

    adl = ((2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])) * df['volume']
    cmf = adl.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()

    df['cmf'] = cmf
    return df


def calculate_obv(df, inplace=True):
    """On-Balance Volume (OBV)"""
    if not inplace:
        df = df.copy()

    obv = [0]  # Initialize OBV with 0 as the starting value

    for i in range(1, len(df['close'])):
        if df['close'].to_numpy()[i] > df['close'].to_numpy()[i - 1]:
            obv.append(obv[-1] + df['volume'].to_numpy()[i])  # If today's close > yesterday's close, add today's volume to OBV
        elif df['close'].to_numpy()[i] < df['close'].to_numpy()[i - 1]:
            obv.append(obv[-1] - df['volume'].to_numpy()[i])  # If today's close < yesterday's close, subtract today's volume from OBV
        else:
            obv.append(obv[-1])  # If today's close = yesterday's close, OBV remains unchanged

    df['obv'] = obv

    return df


def calculate_adl(df, inplace=True):
    """Accumulation/Distribution Line (ADL)"""
    if not inplace:
        df = df.copy()

    adl_values = [0]  # Initialize ADL with 0
    for i in range(1, len(df)):
        money_flow_multiplier = ((df['close'].to_numpy()[i] - df['low'].to_numpy()[i]) - (df['high'].to_numpy()[i] - df['close'].to_numpy()[i])) / (
                df['high'].to_numpy()[i] - df['low'].to_numpy()[i])
        money_flow_volume = money_flow_multiplier * df['volume'].to_numpy()[i]
        adl_values.append(adl_values[i - 1] + money_flow_volume)

    df['adl'] = adl_values
    return df


def calculate_cci(df, window=20, inplace=True):
    """Commodity Channel Index (CCI)"""
    if not inplace:
        df = df.copy()

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    moving_average = typical_price.rolling(window=window).mean()
    mean_deviation = np.abs(typical_price - moving_average).rolling(window=window).mean()

    cci = (typical_price - moving_average) / (0.015 * mean_deviation)
    df['cci'] = cci
    return df


def calculate_parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.2, inplace=True):
    """Parabolic SAR"""
    if not inplace:
        df = df.copy()

    high = df['high']
    low = df['low']

    af = af_start
    trend = 1  # 1 for uptrend, -1 for downtrend
    ep = low.iloc[0] if trend == 1 else high.iloc[0]
    sar_values = [0]

    for i in range(len(df)):
        if trend == 1:
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_step, af_max)
            sar = sar_values[-1] + af * (ep - sar_values[-1])
            if high.iloc[i] > ep:
                trend = -1
                sar = ep
                ep = high.iloc[i]
                af = af_start
        else:
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_step, af_max)
            sar = sar_values[-1] - af * (sar_values[-1] - ep)
            if low.iloc[i] < ep:
                trend = 1
                sar = ep
                ep = low.iloc[i]
                af = af_start
        sar_values.append(sar)

    df['sar'] = sar_values[1:]

    return df


def calculate_fibonacci_retracement(df, start_price=None, end_price=None, inplace=True):
    """Fibonacci Retracement"""
    if not inplace:
        df = df.copy()

    start_price = start_price or max(df['close'])
    end_price = end_price or min(df['close'])

    # Calculate the price range
    price_range = end_price - start_price

    # Fibonacci levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    # Calculate retracement levels
    retracement_levels = [end_price - level * price_range for level in fibonacci_levels]

    # Find the nearest data points to the calculated levels
    nearest_points = []
    for level in retracement_levels:
        nearest_price = min(df['close'], key=lambda x: abs(x - level))
        nearest_points.append((nearest_price, level))

    df['nearest price'] = list(map(lambda x: x[0], nearest_points))
    df['retracement level'] = list(map(lambda x: x[0], nearest_points))

    return df


def calculate_ichimoku_cloud(df, conversion_window=9, base_window=26, leading_span_b_window=52, lagging_span_window=26, inplace=True):
    """Ichimoku Cloud"""
    if not inplace:
        df = df.copy()

    df['Conversion_Line'] = (df['high'].rolling(window=conversion_window).max() + df['low'].rolling(window=conversion_window).min()) / 2
    df['Base_Line'] = (df['high'].rolling(window=base_window).max() + df['low'].rolling(window=base_window).min()) / 2
    df['Leading_Span_A'] = (df['Conversion_Line'] + df['Base_Line']) / 2
    df['Leading_Span_B'] = (df['high'].rolling(window=leading_span_b_window).max() + df['low'].rolling(window=leading_span_b_window).min()) / 2
    df['Lagging_Span'] = df['close'].shift(-lagging_span_window)

    return df


def calculate_aroon(df, window=5, inplace=True):
    """Aroon Indicator"""
    if not inplace:
        df = df.copy()

    aroon_up = [np.nan] * (window - 1)
    aroon_down = [np.nan] * (window - 1)

    for i in range(window, len(df) + 1):
        period = df[i - window: i].to_numpy()
        high_index = period.argmax()
        low_index = period.argmin()

        aroon_up.append(((window - high_index) / window) * 100)
        aroon_down.append(((window - low_index) / window) * 100)

    df['aaron_up'] = aroon_up
    df['aaron_down'] = aroon_down

    return df


def calculate_vwap(df, inplace=True):
    """Volume Weighted Average Price (VWAP)"""
    if not inplace:
        df = df.copy()

    # Make sure 'data' is a DataFrame with columns: ['price', 'volume']
    if not isinstance(df, pd.DataFrame) or 'close' not in df.columns or 'volume' not in df.columns:
        raise ValueError("Input data must be a DataFrame with 'price' and 'volume' columns.")

    # Calculate the cumulative sum of price times volume and cumulative sum of volume
    df['price_volume'] = df['close'] * df['volume']
    df['cumulative_price_volume'] = df['price_volume'].cumsum()
    df['cumulative_volume'] = df['volume'].cumsum()

    # Calculate VWAP
    df['vwap'] = df['cumulative_price_volume'] / df['cumulative_volume']

    # Drop intermediate columns
    df.drop(['price_volume', 'cumulative_price_volume', 'cumulative_volume'], axis=1, inplace=True)

    return df


def calculate_mfi(df, window=14, inplace=True):
    """Money Flow Index (MFI)"""
    if not inplace:
        df = df.copy()

    typical_prices = (df['high'].to_numpy() + df['low'].to_numpy() + df['close'].to_numpy()) / 3
    raw_money_flow = typical_prices * df['volume'].to_numpy()

    positive_money_flow = []
    negative_money_flow = []

    for i in range(1, len(df)):
        if typical_prices[i] > typical_prices[i - 1]:
            positive_money_flow.append(raw_money_flow[i])
            negative_money_flow.append(0)
        elif typical_prices[i] < typical_prices[i - 1]:
            negative_money_flow.append(raw_money_flow[i])
            positive_money_flow.append(0)
        else:
            positive_money_flow.append(0)
            negative_money_flow.append(0)

    positive_money_flow = pd.Series(positive_money_flow)
    negative_money_flow = pd.Series(negative_money_flow)

    positive_money_flow_avg = positive_money_flow.rolling(window=window).sum()
    negative_money_flow_avg = negative_money_flow.rolling(window=window).sum()

    mfi = 100 - (100 / (1 + (positive_money_flow_avg / negative_money_flow_avg)))

    df['mfi'] = mfi

    return mfi


def calculate_roc(df, window=3, inplace=True):
    """Rate of Change (ROC)"""
    if not inplace:
        df = df.copy()

    roc_values = (df['close'] - df['close'].shift(window)) / df['close'].shift(window) * 100

    df['roc'] = roc_values
    return df


def moving_average_ribbon(df, window=10, inplace=True):
    """Moving Average Ribbon"""
    if not inplace:
        df = df.copy()

    df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()

    return df


def calculate_trix(df, window=3, inplace=True):
    """Trix Indicator"""
    if not inplace:
        df = df.copy()

    ema1 = df['close'].ewm(span=window, min_periods=window).mean()
    ema2 = ema1.ewm(span=window, min_periods=window).mean()
    ema3 = ema2.ewm(span=window, min_periods=window).mean()

    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    df['trix'] = trix
    return df


def calculate_standard_deviation(df, window=3, inplace=True):
    """Standard Deviation"""
    if not inplace:
        df = df.copy()

    df = pd.DataFrame(df, columns=['close'])
    df['Std Dev'] = df['close'].rolling(window=window).std()
    return df


def calculate_momentum(df, window=3, inplace=True):
    """Momentum Indicator"""
    if not inplace:
        df = df.copy()

    if 'close' not in df.columns:
        raise ValueError("Data must contain a 'Close' column.")

    close_prices = df['close']
    momentum_values = close_prices.diff(window).fillna(0)
    df['momentum'] = momentum_values
    return df


def calculate_dpo(df, window=10, inplace=True):
    """Detrended Price Oscillator (DPO)"""
    if not inplace:
        df = df.copy()

    # Calculate the simple moving average (SMA) for the given period
    sma = df['close'].shift(window).rolling(window=window).mean()

    # Calculate the DPO
    dpo = df['close'] - sma.shift(int(window / 2) + 1)

    df['dpo'] = dpo

    return df


def calculate_eom(df, window=14, inplace=True):
    """Ease of Movement (EOM)"""
    if not inplace:
        df = df.copy()

    high = df['high']
    low = df['low']
    volume = df['volume']

    distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
    box_ratio = volume / 1000000 / (high - low)

    eom = distance_moved / box_ratio
    eom_smoothed = eom.rolling(window=window).mean()

    df['eom'] = eom_smoothed

    return df


def calculate_keltner_channels(df, ema_window=20, atr_window=10, atr_multiplier=2.0, inplace=True):
    """Keltner Channels"""
    if not inplace:
        df = df.copy()

    # Calculate EMA and ATR
    ema = calculate_ema(df, window=ema_window, inplace=False)
    atr = calculate_atr(df, window=atr_window, inplace=False)

    # Calculate Keltner Channels
    middle_line = ema['ema']
    upper_line = ema['ema'] + (atr['atr'] * atr_multiplier)
    lower_line = ema['ema'] - (atr['atr'] * atr_multiplier)

    df['keltner_middle'] = middle_line
    df['keltner_upper'] = upper_line
    df['keltner_lower'] = lower_line

    return df


def ultimate_oscillator(df, short_window=7, medium_window=14, long_window=28, inplace=True):
    """Ultimate Oscillator"""
    if not inplace:
        df = df.copy()

    # Ensure the input data is a pandas DataFrame with a 'high', 'low', and 'close' column
    if not isinstance(df, pd.DataFrame) or not {'high', 'low', 'close'}.issubset(df.columns):
        raise ValueError("Input data must be a pandas DataFrame with 'high', 'low', and 'close' columns.")

    # Calculate buying pressure and true range
    buying_pressure = df['close'] - pd.concat([df['low'], df['close'].shift()], axis=1).max(axis=1)
    true_range = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)

    # Calculate Average True Range (ATR) over short, medium, and long periods
    atr_short = true_range.rolling(window=short_window).mean()
    atr_medium = true_range.rolling(window=medium_window).mean()
    atr_long = true_range.rolling(window=long_window).mean()

    # Calculate raw ultimate oscillator values
    bp_sum_short = buying_pressure.rolling(window=short_window).sum()
    bp_sum_medium = buying_pressure.rolling(window=medium_window).sum()
    bp_sum_long = buying_pressure.rolling(window=long_window).sum()

    tr_sum_short = atr_short.rolling(window=short_window).sum()
    tr_sum_medium = atr_medium.rolling(window=medium_window).sum()
    tr_sum_long = atr_long.rolling(window=long_window).sum()

    ult_short = 4 * bp_sum_short / tr_sum_short
    ult_medium = 2 * bp_sum_medium / tr_sum_medium
    ult_long = bp_sum_long / tr_sum_long

    # Calculate the weighted Ultimate Oscillator
    weight_short = 4
    weight_medium = 2
    weight_long = 1

    ultimate = (weight_short * ult_short + weight_medium * ult_medium + weight_long * ult_long) / (weight_short + weight_medium + weight_long)

    df['ultimate'] = ultimate

    return df


def calculate_chaikin_oscillator(df, short_span=3, long_span=10, inplace=True):
    """Chaikin Oscillator"""
    if not inplace:
        df = df.copy()

    # Calculate Money Flow Multiplier
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])

    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * df['volume']

    # Calculate Accumulation/Distribution Line (ADL)
    adl = mf_volume.cumsum()

    # Calculate Chaikin Oscillator
    short_ema = adl.ewm(span=short_span, min_periods=1, adjust=False).mean()
    long_ema = adl.ewm(span=long_span, min_periods=1, adjust=False).mean()
    chaikin_oscillator = short_ema - long_ema

    df['chaikin'] = chaikin_oscillator

    return df


def calculate_adx(df, window=14, inplace=True):
    """Average Directional Index (ADX)"""
    if not inplace:
        df = df.copy()

    df['High-Low'] = df['high'] - df['low']
    df['High-PrevClose'] = abs(df['high'] - df['close'].shift(1))
    df['Low-PrevClose'] = abs(df['low'] - df['close'].shift(1))

    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['+DM'] = (df['high'] - df['high'].shift(1)).apply(lambda x: x if x > 0 else 0)
    df['-DM'] = (df['low'].shift(1) - df['low']).apply(lambda x: x if x > 0 else 0)

    df['+DI'] = (df['+DM'].rolling(window=window).sum() / df['TR'].rolling(window=window).sum()) * 100
    df['-DI'] = (df['-DM'].rolling(window=window).sum() / df['TR'].rolling(window=window).sum()) * 100

    df['DX'] = abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']) * 100
    df['ADX'] = df['DX'].rolling(window=window).mean()

    df.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', '+DM', '-DM', 'TR', '+DI', '-DI', 'DX'], axis=1, inplace=True)

    return df


def calculate_tma(df, window=3, inplace=True):
    """Triangular Moving Average (TMA)"""
    if not inplace:
        df = df.copy()

    sma = df.rolling(window=window, min_periods=1, center=True).mean()
    tma = sma['sma'].rolling(window=window, min_periods=1, center=True).mean()

    df['tma'] = tma
    return df
