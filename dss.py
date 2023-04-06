import csv
import yahoo_fin.stock_info as si
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
# -*- coding: utf-8 -*-


def read_csv_file(file_name):
    # Read a CSV file and return a list of strings
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        symbols = []
        for row in reader:
            symbols.append(row[0])
    symbols.pop(0)
    return symbols


def get_symbol_data(symbols):
    symbol_data = []
    for symbol in symbols:
        try:
            si_data = si.get_quote_table(symbol)
            si_data.update({"Ticker Symbol": symbol})
            # Update Return on Equity info
            stats = si.get_stats(symbol)
            si_data.update({"Return on Equity": float(stats.iloc[34][1].replace("%", "")),
                            "Return on Assets": float(stats.iloc[33][1].replace("%", ""))
                            })
            symbol_data.append(si_data)
        except Exception as e:
            print(f"Mã chứng khoán {symbol} không tồn tại hoặc không có dữ liệu")
    return symbol_data


def get_number_market_cap(df):
    for market_cap in df["Market Cap"]:
        if market_cap[-1] == "T":
            df["Market Cap"] = df["Market Cap"].replace(market_cap, float(market_cap[:-1]) * 1e12)
        elif market_cap[-1] == "B":
            df["Market Cap"] = df["Market Cap"].replace(market_cap, float(market_cap[:-1]) * 1e9)
        elif market_cap[-1] == "M":
            df["Market Cap"] = df["Market Cap"].replace(market_cap, float(market_cap[:-1]) * 1e6)
        else:
            df["Market Cap"] = df["Market Cap"].replace(market_cap, float(market_cap[:-1]))
    return df


def preprocessing_symbol_data(df):
    # Chuyển đổi dữ liệu
    df = get_number_market_cap(df)
    df = df.fillna(0)

    # Xóa các dòng có giá trị thiếu
    df = df.dropna()

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()

    return scaler.fit_transform(df)


def get_average_cluster_data(df):
    cluter_data = {}
    for i in range(n_clusters):
        pe_ratio_sum = 0
        esp_sum = 0
        roe_sum = 0
        roa_sum = 0
        for j in range(len(df)):
            pe_ratio_sum += df.iloc[j]['PE Ratio (TTM)'] if df.iloc[j]['Cluster'] == i and not math.isnan(
                df.iloc[j]['PE Ratio (TTM)']) else 0
            esp_sum += df.iloc[j]['EPS (TTM)'] if df.iloc[j]['Cluster'] == i and not math.isnan(
                df.iloc[j]['EPS (TTM)']) else 0
            roe_sum += df.iloc[j]['Return on Equity'] if df.iloc[j]['Cluster'] == i and not math.isnan(
                df.iloc[j]['Return on Equity']) else 0
            roa_sum += df.iloc[j]['Return on Assets'] if df.iloc[j]['Cluster'] == i and not math.isnan(
                df.iloc[j]['Return on Assets']) else 0
        cluter_data.update({i: {'PE Ratio Avg': pe_ratio_sum / len(df[df['Cluster'] == i]),
                                'EPS Avg': esp_sum / len(df[df['Cluster'] == i]),
                                'ROE Avg': roe_sum / len(df[df['Cluster'] == i]),
                                'ROA Avg': roa_sum / len(df[df['Cluster'] == i])}})
    return cluter_data


def input_ticker_symbols():
    ticker_symbols = []
    print("Bạn có muốn nhập mã chứng khoán không?")
    print("1. Có")
    print("2. Không")
    print("3. Tải mã chứng khoán từ file CSV (99 mã chứng khoán)")
    choice = input("Nhập lựa chọn của bạn: ")
    while choice not in ["1", "2", "3"]:
        print("Lựa chọn không hợp lệ, vui lòng nhập lại")
        choice = input("Nhập lựa chọn của bạn: ")
    choice = int(choice)
    if choice == 1:
        print("Nhập mã chứng khoán của các công ty bạn muốn tìm hiểu (Nhập 0 để kết thúc): ")
        while True:
            ticker_symbol = input("Nhập mã chứng khoán (VD: AAPL): ").upper()

            # Xoá các mã chứng khoán trùng lặp
            ticker_symbols = list(dict.fromkeys(ticker_symbols))

            # Kết thúc khi nhập mã chứng khoán là 0
            if ticker_symbol == "0":
                break
            if not ticker_symbol:
                print("Mã chứng khoán không được để trống")
                print("Nhập lại mã chứng khoán của các công ty bạn muốn tìm hiểu (Nhập 0 để kết thúc): ")
                continue
            ticker_symbols.append(ticker_symbol)
        print("Mã chứng khoán bạn đã nhập: ", ticker_symbols)
        print("Các mã chứng khoán đang được phân tích...")
    elif choice == 3:
        ticker_symbols = read_csv_file("ticker_symbols.csv")
        print("Mã chứng khoán được tải từ file CSV: ", ticker_symbols)
        print("Các mã chứng khoán đang được phân tích...")
    return ticker_symbols


if __name__ == '__main__':
    # Nhập mã chứng khoán
    symbols = input_ticker_symbols()
    if symbols:
        # Lấy dữ liệu từ Yahoo Finance
        symbol_data = get_symbol_data(symbols)

        df = pd.DataFrame(symbol_data)

        features = ["1y Target Est", "Avg. Volume", "Beta (5Y Monthly)",
                    "EPS (TTM)", "Market Cap",
                    "Open", "PE Ratio (TTM)", "Previous Close", "Volume",
                    "Return on Equity", "Return on Assets"]
        df_original = df.copy()
        df = df[features]

        # Tiền xử lý dữ liệu
        df_scaled = preprocessing_symbol_data(df)

        # Áp dụng giải thuật K-Means với số cụm tối ưu
        n_clusters = 3 if len(df_scaled) > 3 else len(df_scaled)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(df_scaled)

        # Gán nhãn cho từng điểm dữ liệu
        df["Cluster"] = kmeans.labels_
        df["Ticker Symbol"] = df_original["Ticker Symbol"]
        # Tính trung bình các chỉ số của từng cụm
        cluter_data = get_average_cluster_data(df)

        # Hiển thị biểu đồ
        plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.show()

        # In mã chứng khoán nên đầu tư'
        with open("stock_recommend.txt", "a") as f:
            # Xoá dữ liệu cũ trong file
            f.truncate(0)
        for i in range(n_clusters):
            # Lọc ra các mã chứng khoán nên đầu tư
            df_recommend = df.copy()
            df_recommend = df_recommend[df_recommend['Cluster'] == i]
            df_recommend = df_recommend[df_recommend['PE Ratio (TTM)'] <= cluter_data[i]['PE Ratio Avg']]
            df_recommend = df_recommend[df_recommend['EPS (TTM)'] >= cluter_data[i]['EPS Avg']]
            df_recommend = df_recommend[df_recommend['Return on Equity'] >= cluter_data[i]['ROE Avg']]
            df_recommend = df_recommend[df_recommend['Return on Assets'] >= cluter_data[i]['ROA Avg']]
            with open("stock_recommend.txt", "a") as f:
                # Xoá dữ liệu cũ trong file
                f.write(f"Các mã chứng khoán nên đầu tư trong cụm {i}: {df_recommend['Ticker Symbol'].values}\n")
                f.write(f"Đặc điểm của cụm {i}:\n {df[df['Cluster'] == i].describe()}\n")
            print(f"Giá trị trung bình của các chỉ số của cụm {i}: {cluter_data[i]}")
            print(f"Các mã chứng khoán nên đầu tư trong cụm {i}: {df_recommend['Ticker Symbol'].values}")
            # Đặc điểm của từng cụm
            print(f"Đặc điểm của cụm {i}:\n {df[df['Cluster'] == i].describe()}")

        print("\nĐã lưu các mã chứng khoán nên đầu tư vào file stock_recommend.txt")
    else:
        print("\nKhông có mã chứng khoán nào được nhập vào!")
