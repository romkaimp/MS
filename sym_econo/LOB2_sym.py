import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_lob2_dummy_data(n_records=1000, levels=5, ticker="AAPL", start_price=150.0):
    """
    Генерация реалистичных dummy данных для стакана цен Level 2

    Parameters:
    n_records: количество записей
    levels: количество уровней глубины
    ticker: тикер инструмента
    start_price: начальная цена
    """

    np.random.seed(42)

    # Базовые параметры
    spread = 0.01  # начальный спред
    volatility = 0.001  # волатильность
    base_volume = 1000  # базовый объем

    # Временные метки
    start_time = datetime(2024, 1, 15, 9, 30, 0)
    timestamps = [start_time + timedelta(seconds=i * 10) for i in range(n_records)]

    # Генерация средней цены с случайным блужданием
    mid_prices = [start_price]
    for i in range(1, n_records):
        change = np.random.normal(0, volatility * mid_prices[-1])
        new_price = mid_prices[-1] + change
        mid_prices.append(new_price)

    data = []

    for i, (timestamp, mid_price) in enumerate(zip(timestamps, mid_prices)):
        record = {'timestamp': timestamp, 'mid_price': mid_price}

        # Динамический спред
        current_spread = spread * (1 + np.random.uniform(-0.3, 0.3))

        # Генерация уровней BID (покупатели)
        bid_prices = []
        bid_sizes = []
        current_bid = mid_price - current_spread / 2

        for level in range(1, levels + 1):
            # Цена уменьшается с каждым уровнем
            price_level = current_bid - (level - 1) * 0.01
            # Объем обычно уменьшается с глубиной
            size = max(100, int(base_volume * np.random.lognormal(0, 0.5) / (level ** 0.7)))

            bid_prices.append(round(price_level, 2))
            bid_sizes.append(size)

            record[f'bid_price_{level}'] = round(price_level, 2)
            record[f'bid_size_{level}'] = size

        # Генерация уровней ASK (продавцы)
        ask_prices = []
        ask_sizes = []
        current_ask = mid_price + current_spread / 2

        for level in range(1, levels + 1):
            # Цена увеличивается с каждым уровнем
            price_level = current_ask + (level - 1) * 0.01
            # Объем обычно уменьшается с глубиной
            size = max(100, int(base_volume * np.random.lognormal(0, 0.5) / (level ** 0.7)))

            ask_prices.append(round(price_level, 2))
            ask_sizes.append(size)

            record[f'ask_price_{level}'] = round(price_level, 2)
            record[f'ask_size_{level}'] = size

        # Расчет дополнительных метрик
        record['spread'] = round(current_spread, 4)
        record['total_bid_volume'] = sum(bid_sizes)
        record['total_ask_volume'] = sum(ask_sizes)
        record['volume_imbalance'] = record['total_bid_volume'] - record['total_ask_volume']

        # Визуальные индикаторы агрессии
        record['market_buy_pressure'] = np.random.exponential(1.0)
        record['market_sell_pressure'] = np.random.exponential(1.0)

        data.append(record)

    return pd.DataFrame(data)


def add_market_events(df, event_probability=0.02):
    """
    Добавление рыночных событий для реалистичности
    """
    df_with_events = df.copy()

    for i in range(1, len(df) - 1):
        if np.random.random() < event_probability:
            event_type = np.random.choice(['large_trade', 'spread_widening', 'liquidity_void'])

            if event_type == 'large_trade':
                # Большая сделка - резкое изменение объемов
                level = np.random.randint(1, 4)
                if np.random.random() < 0.5:  # buy trade
                    df_with_events.at[i, f'ask_size_{level}'] = int(df_with_events.at[i, f'ask_size_{level}'] * 0.3)
                else:  # sell trade
                    df_with_events.at[i, f'bid_size_{level}'] = int(df_with_events.at[i, f'bid_size_{level}'] * 0.3)

            elif event_type == 'spread_widening':
                # Расширение спреда
                df_with_events.at[i, 'ask_price_1'] += 0.05
                df_with_events.at[i, 'bid_price_1'] -= 0.03

            elif event_type == 'liquidity_void':
                # Исчезновение ликвидности на уровнях
                for level in [2, 3]:
                    df_with_events.at[i, f'bid_size_{level}'] = max(10, df_with_events.at[i, f'bid_size_{level}'] // 10)
                    df_with_events.at[i, f'ask_size_{level}'] = max(10, df_with_events.at[i, f'ask_size_{level}'] // 10)

    return df_with_events


# Генерация данных
print("Генерация LOB2 dummy данных...")
lob_data = generate_lob2_dummy_data(n_records=500, levels=5, ticker="GAZP", start_price=160.0)

# Добавление рыночных событий
lob_data_with_events = add_market_events(lob_data)

# Сохранение в CSV
lob_data_with_events.to_csv('lob2_dummy_data.csv', index=False)

print("Данные успешно сгенерированы!")
print(f"Размер датасета: {lob_data_with_events.shape}")
print("\nПервые 5 записей:")
print(lob_data_with_events.head())

print("\nИнформация о данных:")
print(lob_data_with_events.info())

def visualize_lob_data(df, n_samples=50):
    """
    Визуализация сгенерированных LOB2 данных
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Динамика средней цены и спреда
    sample_df = df.head(n_samples)

    axes[0, 0].plot(sample_df['timestamp'], sample_df['mid_price'],
                    label='Mid Price', linewidth=2, color='blue')
    axes[0, 0].set_title('Динамика средней цены')
    axes[0, 0].set_ylabel('Цена')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(sample_df['timestamp'], sample_df['spread'] * 100,
                    label='Spread (bps)', linewidth=2, color='red')
    axes[0, 1].set_title('Динамика спреда')
    axes[0, 1].set_ylabel('Спред (б.п.)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 2. Volume imbalance
    axes[1, 0].plot(sample_df['timestamp'], sample_df['volume_imbalance'],
                    label='Volume Imbalance', linewidth=2, color='green')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Дисбаланс объемов')
    axes[1, 0].set_ylabel('Разница объемов')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 3. Глубина стакана для одного снимка
    snapshot_idx = n_samples // 2
    snapshot = df.iloc[snapshot_idx]

    bid_prices = [snapshot[f'bid_price_{i}'] for i in range(1, 6)]
    bid_sizes = [snapshot[f'bid_size_{i}'] for i in range(1, 6)]
    ask_prices = [snapshot[f'ask_price_{i}'] for i in range(1, 6)]
    ask_sizes = [snapshot[f'ask_size_{i}'] for i in range(1, 6)]

    axes[1, 1].barh([f'Bid {i}' for i in range(1, 6)], bid_sizes,
                    color='green', alpha=0.6, label='Bid Size')
    axes[1, 1].barh([f'Ask {i}' for i in range(1, 6)], ask_sizes,
                    color='red', alpha=0.6, label='Ask Size')
    axes[1, 1].set_title(f'Глубина стакана (снимок {snapshot_idx})')
    axes[1, 1].set_xlabel('Объем')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def print_lob_snapshot(df, snapshot_idx=0):
    """
    Вывод красивого снимка стакана цен
    """
    snapshot = df.iloc[snapshot_idx]

    print(f"\n📊 Снимок стакана цен (время: {snapshot['timestamp']})")
    print("=" * 60)
    print(f"Mid Price: {snapshot['mid_price']:.2f} | Spread: {snapshot['spread']:.4f}")
    print(f"Bid Volume: {snapshot['total_bid_volume']} | Ask Volume: {snapshot['total_ask_volume']}")
    print("-" * 60)
    print(f"{'Уровень':<10} {'Цена BID':<12} {'Объем':<12} {'Цена ASK':<12} {'Объем':<12}")
    print("-" * 60)

    for level in range(1, 6):
        bid_price = snapshot[f'bid_price_{level}']
        bid_size = snapshot[f'bid_size_{level}']
        ask_price = snapshot[f'ask_price_{level}']
        ask_size = snapshot[f'ask_size_{level}']

        print(f"{level:<10} {bid_price:<12.2f} {bid_size:<12} {ask_price:<12.2f} {ask_size:<12}")

    print("=" * 60)


# Визуализация данных
print("Визуализация LOB2 данных...")
visualize_lob_data(lob_data_with_events)

# Вывод нескольких снимков
print_lob_snapshot(lob_data_with_events, 0)
print_lob_snapshot(lob_data_with_events, 50)
print_lob_snapshot(lob_data_with_events, 100)

# Статистика данных
print("\n📈 Статистика данных:")
print(lob_data_with_events[['mid_price', 'spread', 'total_bid_volume', 'total_ask_volume']].describe())

# Корреляционная матрица
print("\n🔗 Корреляционная матрица:")
numeric_cols = ['mid_price', 'spread', 'total_bid_volume', 'total_ask_volume', 'volume_imbalance']
correlation_matrix = lob_data_with_events[numeric_cols].corr()
print(correlation_matrix)


def create_advanced_lob_dataset():
    """
    Создание расширенного датасета LOB2 с дополнительными фичами
    """
    base_df = generate_lob2_dummy_data(n_records=1000, levels=5, ticker="GAZP", start_price=160.0)

    # Добавление производных фич
    df = base_df.copy()

    # Временные фичи
    df['time_index'] = range(len(df))
    df['minute_of_day'] = df['timestamp'].dt.minute + df['timestamp'].dt.hour * 60

    # Ценовые фичи
    df['price_change'] = df['mid_price'].diff()
    df['price_volatility'] = df['price_change'].rolling(window=5, min_periods=1).std()
    df['price_momentum'] = df['mid_price'].pct_change(periods=3)

    # Volume-based features
    for level in range(1, 6):
        df[f'bid_size_change_{level}'] = df[f'bid_size_{level}'].diff()
        df[f'ask_size_change_{level}'] = df[f'ask_size_{level}'].diff()

    # Spread features
    df['spread_change'] = df['spread'].diff()
    df['relative_spread'] = df['spread'] / df['mid_price']

    # Order book imbalance features
    df['depth_imbalance'] = (df['total_bid_volume'] - df['total_ask_volume']) / (
                df['total_bid_volume'] + df['total_ask_volume'])

    # Microprice calculation
    df['microprice'] = (df['bid_price_1'] * df['ask_size_1'] + df['ask_price_1'] * df['bid_size_1']) / (
                df['bid_size_1'] + df['ask_size_1'])

    # Заполнение NaN значений
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df


# Создание расширенного датасета
print("Создание расширенного датасета LOB2...")
advanced_lob_data = create_advanced_lob_dataset()

print("Расширенный датасет создан!")
print(f"Колонки: {list(advanced_lob_data.columns)}")
print(f"Размер: {advanced_lob_data.shape}")

# Сохранение
advanced_lob_data.to_csv('advanced_lob2_data.csv', index=False)
print("Расширенный датасет сохранен в 'advanced_lob2_data.csv'")