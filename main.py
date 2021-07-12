import pandas
import matplotlib.pyplot as plt
import numpy


def main():
    # Загружаем данные из файла
    data = pandas.read_csv(filepath_or_buffer='data.csv',
                           sep='\t')

    # Если файл загрузился не в DataFrame, то выдаём ошибку
    if not isinstance(data, pandas.DataFrame):
        print('--ERROR--\nCan not load file\n')
        return 0

    prepared_data = data.to_numpy()

    # ----Приведение данных к массивам (дата, максимальная, минимальная и разница цен)----
    cur_date = prepared_data[0][0].split()
    max_price = float(prepared_data[0][1].replace(',', '.'))
    min_price = float(prepared_data[0][1].replace(',', '.'))

    dates = list()
    prices = list()
    max_prices = list()
    min_prices = list()

    for i, j in prepared_data:
        if j != '-':
            # Разделяем строку на дату и время
            i = i.split()

            j = float(j.replace(',', '.'))

            if i[0] != cur_date[0]:
                dates.append(i[0])
                prices.append(max_price - min_price)
                max_prices.append(max_price)
                min_prices.append(min_price)

                min_price = j
                max_price = j
                cur_date = i

            if j > max_price:
                max_price = j

            if j < min_price:
                min_price = j
    # ---------------------------------------------------------------------------------------------

    # ----Нахождение коэффициента корреляции между временем суток и ценой на электроэнергию----
    time = list()
    prices_per_day = list()
    t_array = list()

    for i in range(0, 95):
        time.append(15 * i)
        for j in range(0, 31):
            t_array.append(float(prepared_data[(j * 96) + i][1].replace(',', '.')))

        print(t_array)
        prices_per_day.append(numpy.median(t_array))
        t_array.clear()

    print(numpy.corrcoef(prices_per_day, time))
    # ---------------------------------------------------------------------------------------------

    # ----Нахождение и построение скользящей средней(медиана) по разности максимальной и минимальной цен----
    average = 0
    average_thirty_prices = list()
    average_sixty_prices = list()
    average_ninety_prices = list()
    temp_list = list()

    for i in range(0, len(prices) - 31):
        for j in range(i, 29 + i):
            temp_list.append(prices[j])

        average_thirty_prices.append(numpy.median(temp_list))
        temp_list.clear()

    for i in range(0, len(prices) - 61):
        for j in range(i, 59 + i):
            temp_list.append(prices[j])

        average_sixty_prices.append(numpy.median(temp_list))
        temp_list.clear()

    for i in range(0, len(prices) - 91):
        for j in range(i, 89 + i):
            temp_list.append(prices[j])

        average_ninety_prices.append(numpy.median(temp_list))
        temp_list.clear()
    # ---------------------------------------------------------------------------------------------

    # for i in range(0, len(prices) - 31):
    #     for j in range(i, 29 + i):
    #         average += prices[j]
    #
    #     average_thirty_prices.append(average / 30)
    #     average = 0
    #
    # average = 0
    #
    # for i in range(0, len(prices) - 61):
    #     for j in range(i, 59 + i):
    #         average += prices[j]
    #
    #     average_sixty_prices.append(average / 60)
    #     average = 0
    #
    # average = 0
    #
    # for i in range(0, len(prices) - 91):
    #     for j in range(i, 89 + i):
    #         average += prices[j]
    #
    #     average_ninety_prices.append(average / 90)
    #     average = 0

    # ----Построение графиков по разности максимальной и минимальной цен----
    plt.plot(range(0, len(prices)), prices, label='Разница между max и min ценой')
    plt.plot(range(len(prices) - len(average_thirty_prices), len(prices)), average_thirty_prices,
             label='Скользящая средняя (30 дней)')
    plt.plot(range(len(prices) - len(average_sixty_prices), len(prices)), average_sixty_prices,
             label='Скользящая средняя (60 дней)')
    plt.plot(range(len(prices) - len(average_ninety_prices), len(prices)), average_ninety_prices,
             label='Скользящая средняя (90 дней)')
    plt.title('Разница между максимальной и минимальной ценой за электроэнергию в один день')
    plt.xlabel('Дни')
    plt.legend(loc='upper right')
    plt.ylim(-1.0, 200.0)
    plt.show()
    # plt.subplot(2, 2, 1)
    # plt.title('Разница между максимальной и минимальной ценой за электроэнергию в один день')
    # plt.xlabel('Дни')
    # plt.ylabel('Разница')
    # plt.plot(prices)
    # plt.subplot(2, 2, 2)
    # plt.title('Скользящая средняя за 30 дней')
    # plt.plot(average_thirty_prices)
    # plt.subplot(2, 2, 3)
    # plt.title('Скользящая средняя за 60 дней')
    # plt.plot(average_sixty_prices)
    # plt.subplot(2, 2, 4)
    # plt.title('Скользящая средняя за 90 дней')
    # plt.plot(average_ninety_prices)
    # plt.show()
    # ---------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
