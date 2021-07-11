import pandas
import matplotlib.pyplot as plt


def LoadOneDay():

    # Загружаем данные из файла
    data = pandas.read_csv(filepath_or_buffer='/media/vlad/GitHub/electricity/data.csv',
                           sep='\t')

    # Если файл загрузился не в DataFrame, то выдаём ошибку
    if not isinstance(data, pandas.DataFrame):
        print('--ERROR--\nCan not load file\n')
        return 0

    prepared_data = data.to_numpy()

    cur_date = prepared_data[0][0].split()
    max_price = prepared_data[0][1]
    min_price = prepared_data[0][1]

    dates = list()
    prices = list()
    max_prices = list()
    min_prices = list()

    for i, j in prepared_data:
        # Разделяем строку на дату и время
        i = i.split()

        if i[0] != cur_date[0]:
            print(cur_date[0])
            print(min_price)
            print(max_price)

            dates.append(i[0])
            prices.append(float(max_price.replace(',', '.'))-float(min_price.replace(',', '.')))
            max_prices.append(float(max_price.replace(',', '.')))
            min_prices.append(float(min_price.replace(',', '.')))

            min_price = j
            max_price = j
            cur_date = i

        if j > max_price and len(j) > 1:
            max_price = j

        if j < min_price and len(j) > 1:
            min_price = j

    print(cur_date[0])
    print(min_price)
    print(max_price)

    plt.plot(prices)
    plt.show()


if __name__ == '__main__':
    LoadOneDay()
