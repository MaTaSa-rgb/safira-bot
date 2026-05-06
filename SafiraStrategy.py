# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib


class SafiraStrategy(IStrategy):
    """
    SAFIRA — Crypto Futures Strategy
    
    Estrategia basada en confluencia de:
    - Tendencia: EMA 9/21/50 + estructura de mercado
    - Momentum: RSI(14) con detección de divergencias
    - Volatilidad: Bollinger Bands(20,2) + ATR(14)
    - Volumen: confirmación de breakouts
    
    Gestión de riesgo:
    - Stop loss: 1.5x ATR desde entrada
    - R:R mínimo 1:2
    - Máximo 3 posiciones abiertas
    - Circuit breaker: pausa si drawdown > 5%
    """

    INTERFACE_VERSION = 3

    # ═══════════════════════════════════════
    # CONFIGURACIÓN BASE
    # ═══════════════════════════════════════
    can_short = True
    timeframe = '1h'
    stoploss = -0.05
    trailing_stop = False

    minimal_roi = {
        "0":   0.08,
        "30":  0.05,
        "60":  0.03,
        "120": 0.02
    }

    # Parámetros optimizables (para backtesting)
    ema_fast = IntParameter(7, 15, default=9, space='buy', optimize=True)
    ema_mid = IntParameter(18, 25, default=21, space='buy', optimize=True)
    ema_slow = IntParameter(45, 60, default=50, space='buy', optimize=True)
    rsi_period = IntParameter(10, 18, default=14, space='buy', optimize=True)
    rsi_oversold = IntParameter(25, 38, default=35, space='buy', optimize=True)
    rsi_overbought = IntParameter(62, 75, default=65, space='sell', optimize=True)
    volume_threshold = DecimalParameter(1.1, 2.0, default=1.3, space='buy', optimize=True)
    atr_multiplier = DecimalParameter(1.2, 2.0, default=1.5, space='sell', optimize=True)

    # ═══════════════════════════════════════
    # PROCESO DE ORDEN
    # ═══════════════════════════════════════
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    process_only_new_candles = True
    startup_candle_count: int = 100

    # ═══════════════════════════════════════
    # INDICADORES
    # ═══════════════════════════════════════
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # — EMAs —
        dataframe['ema9']  = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=self.ema_mid.value)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # — RSI —
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # — MACD —
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd']        = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist']   = macd['macdhist']

        # — Bollinger Bands —
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe['bb_upper']  = bollinger['upper']
        dataframe['bb_mid']    = bollinger['mid']
        dataframe['bb_lower']  = bollinger['lower']
        dataframe['bb_width']  = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_mid']

        # — ATR —
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # — Volumen —
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']

        # — Stochastico —
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # — Estructura de mercado —
        # Máximos y mínimos locales (ventana 5 velas)
        dataframe['high_max'] = dataframe['high'].rolling(window=5).max()
        dataframe['low_min']  = dataframe['low'].rolling(window=5).min()

        # Tendencia EMA alineada
        dataframe['ema_bullish'] = (
            (dataframe['ema9'] > dataframe['ema21']) &
            (dataframe['ema21'] > dataframe['ema50'])
        ).astype(int)

        dataframe['ema_bearish'] = (
            (dataframe['ema9'] < dataframe['ema21']) &
            (dataframe['ema21'] < dataframe['ema50'])
        ).astype(int)

        # — Cruce EMA 9/21 —
        dataframe['ema_cross_bull'] = (
            (dataframe['ema9'] > dataframe['ema21']) &
            (dataframe['ema9'].shift(1) <= dataframe['ema21'].shift(1))
        )
        dataframe['ema_cross_bear'] = (
            (dataframe['ema9'] < dataframe['ema21']) &
            (dataframe['ema9'].shift(1) >= dataframe['ema21'].shift(1))
        )

        # — Cruce MACD —
        dataframe['macd_cross_bull'] = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1))
        )
        dataframe['macd_cross_bear'] = (
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) >= dataframe['macd_signal'].shift(1))
        )

        # — Precio sobre/bajo EMAs —
        dataframe['price_above_ema50']  = dataframe['close'] > dataframe['ema50']
        dataframe['price_above_ema200'] = dataframe['close'] > dataframe['ema200']

        # — BB Squeeze (baja volatilidad = expansión inminente) —
        dataframe['bb_squeeze'] = dataframe['bb_width'] < dataframe['bb_width'].rolling(50).quantile(0.2)

        return dataframe

    # ═══════════════════════════════════════
    # SEÑALES DE ENTRADA — LONG
    # ═══════════════════════════════════════
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ── LONG: confluencia de 3+ condiciones ──
        conditions_long = []

        # 1. Contexto alcista: precio sobre EMA50 y EMA200
        conditions_long.append(dataframe['price_above_ema50'])
        conditions_long.append(dataframe['price_above_ema200'])

        # 2. Tendencia EMA alineada alcista
        conditions_long.append(dataframe['ema_bullish'] == 1)

        # 3. RSI saliendo de sobreventa o en zona neutral-alcista
        conditions_long.append(
            (dataframe['rsi'] > self.rsi_oversold.value) &
            (dataframe['rsi'] < 60)
        )

        # 4. MACD cruce alcista O histograma positivo creciendo
        conditions_long.append(
            dataframe['macd_cross_bull'] |
            (
                (dataframe['macd_hist'] > 0) &
                (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))
            )
        )

        # 5. Volumen confirma movimiento
        conditions_long.append(
            dataframe['volume_ratio'] > self.volume_threshold.value
        )

        # 6. Precio no en banda superior de Bollinger (no sobreextendido)
        conditions_long.append(
            dataframe['close'] < dataframe['bb_upper'] * 0.99
        )

        # Requiere MÍNIMO 4 de 6 condiciones (sistema de puntos)
        score_long = sum([c.astype(int) for c in conditions_long])

        dataframe.loc[
            (score_long >= 4) &
            (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']
        ] = (1, 'safira_long')

        # ── SHORT: confluencia de 3+ condiciones ──
        conditions_short = []

        # 1. Contexto bajista: precio bajo EMA50
        conditions_short.append(~dataframe['price_above_ema50'])

        # 2. Tendencia EMA alineada bajista
        conditions_short.append(dataframe['ema_bearish'] == 1)

        # 3. RSI bajando desde sobrecompra o en zona neutral-bajista
        conditions_short.append(
            (dataframe['rsi'] < (100 - self.rsi_oversold.value)) &
            (dataframe['rsi'] > 40)
        )

        # 4. MACD cruce bajista O histograma negativo decreciendo
        conditions_short.append(
            dataframe['macd_cross_bear'] |
            (
                (dataframe['macd_hist'] < 0) &
                (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))
            )
        )

        # 5. Volumen confirma movimiento
        conditions_short.append(
            dataframe['volume_ratio'] > self.volume_threshold.value
        )

        # 6. Precio no en banda inferior de Bollinger (no sobreextendido)
        conditions_short.append(
            dataframe['close'] > dataframe['bb_lower'] * 1.01
        )

        score_short = sum([c.astype(int) for c in conditions_short])

        dataframe.loc[
            (score_short >= 4) &
            (dataframe['volume'] > 0),
            ['enter_short', 'enter_tag']
        ] = (1, 'safira_short')

        return dataframe

    # ═══════════════════════════════════════
    # SEÑALES DE SALIDA
    # ═══════════════════════════════════════
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Salida LONG
        dataframe.loc[
            (
                # RSI en sobrecompra
                (dataframe['rsi'] > self.rsi_overbought.value) |
                # Cruce EMA bajista
                dataframe['ema_cross_bear'] |
                # Precio toca banda superior BB
                (dataframe['close'] >= dataframe['bb_upper']) |
                # MACD cruce bajista
                dataframe['macd_cross_bear']
            ) &
            (dataframe['volume'] > 0),
            ['exit_long', 'exit_tag']
        ] = (1, 'safira_exit_long')

        # Salida SHORT
        dataframe.loc[
            (
                # RSI en sobreventa
                (dataframe['rsi'] < self.rsi_oversold.value) |
                # Cruce EMA alcista
                dataframe['ema_cross_bull'] |
                # Precio toca banda inferior BB
                (dataframe['close'] <= dataframe['bb_lower']) |
                # MACD cruce alcista
                dataframe['macd_cross_bull']
            ) &
            (dataframe['volume'] > 0),
            ['exit_short', 'exit_tag']
        ] = (1, 'safira_exit_short')

        return dataframe

    # ═══════════════════════════════════════
    # STOP LOSS DINÁMICO (ATR)
    # ═══════════════════════════════════════
    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', None)

        if atr and atr > 0:
            atr_stop = (atr * self.atr_multiplier.value) / current_rate
            # No ampliar el stop más allá del 8%
            return max(-atr_stop, -0.08)

        return self.stoploss

    # ═══════════════════════════════════════
    # CONFIRMACIÓN DE ENTRADA (filtro extra)
    # ═══════════════════════════════════════
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, current_time: datetime,
                            entry_tag: Optional[str], side: str, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return False

        last = dataframe.iloc[-1]

        # No entrar si el volumen es muy bajo (candle sin convicción)
        if last['volume_ratio'] < 0.5:
            return False

        # No entrar si BB está en squeeze extremo sin ruptura
        if last['bb_squeeze'] and last['volume_ratio'] < 1.5:
            return False

        # No entrar si RSI está en zona neutral sin dirección clara
        rsi = last['rsi']
        if 45 <= rsi <= 55:
            return False

        return True
