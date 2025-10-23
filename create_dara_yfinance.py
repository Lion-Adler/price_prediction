import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time

print("Start data save code")

# Списки самых важных тикеров по каждой категории
STOCKS = [
    # US Stocks (30)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
    'PG', 'UNH', 'HD', 'MA', 'DIS', 'ADBE', 'NFLX', 'PYPL', 'INTC', 'CMCSA',
    'PFE', 'CSCO', 'PEP', 'ABT', 'TMO', 'AVGO', 'ACN', 'CRM', 'TXN', 'QCOM',
    # International Stocks (20)
    'TSM', 'ASML', 'NSRGY', 'SAP', 'LIN', 'NVO', 'HSBC', 'SAN', 'AIR', 'BMW',
    'SIEGY', 'TTE', 'UL', 'AZN', 'RY', 'BHP', 'RIO', 'NVS', 'TM', 'SONY'
]

FOREX = [
    # Major Pairs (15)
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X',
    'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',
    'EURCHF=X', 'AUDJPY=X', 'USDCNY=X', 'USDHKD=X', 'USDSGD=X',
    # Minor Pairs (15)
    'EURCAD=X', 'EURAUD=X', 'EURSEK=X', 'EURNZD=X', 'GBPCAD=X',
    'GBPAUD=X', 'GBPCHF=X', 'AUDCAD=X', 'AUDNZD=X', 'CADJPY=X',
    'CHFJPY=X', 'NZDJPY=X', 'SGDJPY=X', 'TRYJPY=X', 'ZARJPY=X',
    # Exotic Pairs (20)
    'USDTRY=X', 'USDZAR=X', 'USDMXN=X', 'USDTHB=X', 'USDPLN=X',
    'USDCZK=X', 'USDHUF=X', 'USDDKK=X', 'USDNOK=X', 'USDSEK=X',
    'USDRUB=X', 'USDINR=X', 'USDBRL=X', 'USDKRW=X', 'USDTWD=X',
    'USDPHP=X', 'USDIDR=X', 'USDMYR=X', 'USDVND=X', 'USDARS=X'
]

COMMODITIES = [
    # Precious Metals (10)
    'GC=F', 'SI=F', 'PL=F', 'PA=F', 'HG=F',
    'GLD', 'SLV', 'PPLT', 'GDX', 'GDXJ',
    # Energy (15)
    'CL=F', 'BZ=F', 'NG=F', 'HO=F', 'RB=F',
    'UNG', 'USO', 'BNO', 'UCO', 'SCO',
    'DBO', 'XLE', 'XOP', 'VDE', 'IEZ',
    # Agriculture (15)
    'ZC=F', 'ZW=F', 'ZS=F', 'ZL=F', 'ZO=F',
    'KE=F', 'CC=F', 'KC=F', 'CT=F', 'SB=F',
    'LB=F', 'CORN', 'WEAT', 'SOYB', 'CANE',
    # Livestock & Other (10)
    'LE=F', 'HE=F', 'GF=F', 'ALI=F', 'CSC=F',
    'BAL', 'JO', 'NIB', 'COW', 'JJG'
]

CRYPTO = [
    # Major Cryptos (15)
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
    'SOL-USD', 'DOT-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD',
    'LTC-USD', 'LINK-USD', 'ATOM-USD', 'XLM-USD', 'ALGO-USD',
    # Mid Cap Cryptos (20)
    'ETC-USD', 'XMR-USD', 'EOS-USD', 'XTZ-USD', 'AAVE-USD',
    'UNI-USD', 'FIL-USD', 'THETA-USD', 'VET-USD', 'TRX-USD',
    'ICP-USD', 'FTM-USD', 'NEAR-USD', 'GRT-USD', 'SAND-USD',
    'AXS-USD', 'APE-USD', 'QNT-USD', 'RUNE-USD', 'KSM-USD',
    # Small Cap Cryptos (15)
    'CHZ-USD', 'ENJ-USD', 'MANA-USD', 'GALA-USD', 'CRV-USD',
    'COMP-USD', 'MKR-USD', 'SNX-USD', 'YFI-USD', 'UMA-USD',
    'BAT-USD', 'ZEC-USD', 'DASH-USD', 'NEO-USD', 'WAVES-USD'
]

ETFS = [
    # US Equity ETFs (10)
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
    'VOO', 'IVV', 'VUG', 'VTV', 'VOOG',
    # International ETFs (5)
    'VEA', 'VWO', 'EFA', 'EEM', 'IEMG',
    # Sector ETFs (5)
    'XLK', 'XLV', 'XLF', 'XLE', 'XLI',
    # Bond ETFs (5)
    'BND', 'AGG', 'TLT', 'IEF', 'LQD',
    # Commodity & Specialty ETFs (5)
    'GLD', 'SLV', 'USO', 'UNG', 'VNQ'
]

# Создаем словарь для удобного доступа к данным
categories = {
    'STOCKS': STOCKS,
    'FOREX': FOREX, 
    'COMMODITIES': COMMODITIES,
    'CRYPTO': CRYPTO,
    'ETFS': ETFS
}

def setup_logging():
    """Настройка логирования"""
    log_filename = f"download_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_filename, 'w', encoding='utf-8')
    log_file.write(f"Лог ошибок скачивания данных\n")
    log_file.write(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 50 + "\n\n")
    return log_file

def download_and_save_data(ticker, category, log_file, interval="1d", period="max"):
    """
    Функция для скачивания и сохранения данных по тикеру
    """
    try:
        print(f"Скачиваю {ticker} ({category})...")
        
        # Скачиваем данные
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            error_msg = f"Нет данных для {ticker} ({category})"
            print(f"  {error_msg}")
            log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
            return False
        
        # Создаем директорию, если она не существует
        os.makedirs(category, exist_ok=True)
        
        # Формируем имя файла
        filename = f"{category}/{ticker}.csv"
        
        # Сохраняем в CSV
        data.to_csv(filename)
        print(f"  Данные сохранены в {filename}")
        return True
        
    except Exception as e:
        error_msg = f"Ошибка при обработке {ticker} ({category}): {str(e)}"
        print(f"  {error_msg}")
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
        return False

# Основной цикл обработки
def main():
    print("Начинаю скачивание данных...")
    
    # Настраиваем логирование
    log_file = setup_logging()
    
    total_downloaded = 0
    total_errors = 0
    errors_by_category = {category: 0 for category in categories.keys()}
    
    try:
        for category, tickers in categories.items():
            print(f"\n=== Обрабатываю категорию: {category} ===")
            print(f"Количество тикеров: {len(tickers)}")
            
            for i, ticker in enumerate(tickers, 1):
                print(f"[{i}/{len(tickers)}] ", end="")
                success = download_and_save_data(ticker, category, log_file)
                
                if success:
                    total_downloaded += 1
                else:
                    total_errors += 1
                    errors_by_category[category] += 1
                
                # Небольшая задержка между запросами чтобы не получить бан
                time.sleep(0.1)
        
        # Записываем итоговую статистику в лог
        log_file.write("\n" + "=" * 50 + "\n")
        log_file.write("ИТОГОВАЯ СТАТИСТИКА:\n")
        log_file.write(f"Успешно скачано: {total_downloaded}\n")
        log_file.write(f"Ошибок: {total_errors}\n")
        log_file.write(f"Всего обработано: {total_downloaded + total_errors}\n\n")
        
        log_file.write("Ошибки по категориям:\n")
        for category, error_count in errors_by_category.items():
            log_file.write(f"{category}: {error_count} ошибок\n")
        
        print(f"\n=== Завершено ===")
        print(f"Успешно скачано: {total_downloaded}")
        print(f"Ошибок: {total_errors}")
        print(f"Всего обработано: {total_downloaded + total_errors}")
        print(f"\nЛог ошибок сохранен в файл: {log_file.name}")
        
    finally:
        # Всегда закрываем файл лога
        log_file.close()

if __name__ == "__main__":
    main()
