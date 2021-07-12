import pandas
import matplotlib.pyplot as plt
import numpy
from datetime import date, datetime


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
    prices = list()
    dates = list()

    cur_date = prepared_data[0][0].split()
    max_price = float(prepared_data[0][1].replace(',', '.'))
    min_price = float(prepared_data[0][1].replace(',', '.'))

    dates.append(datetime.strptime(cur_date[0], '%m/%d/%Y'))

    for i, j in prepared_data:
        if j != '-':
            # Разделяем строку на дату и время
            i = i.split()

            j = float(j.replace(',', '.'))

            if i[0] != cur_date[0]:
                prices.append(max_price - min_price)
                dates.append(datetime.strptime(i[0], '%m/%d/%Y'))

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

        prices_per_day.append(numpy.median(t_array))
        t_array.clear()

    print(numpy.corrcoef(prices_per_day, time))
    # ---------------------------------------------------------------------------------------------

    # ----Нахождение скользящей средней для сезонов----
    summer = list([list(), list(), list()])
    winter = list([list(), list(), list()])
    spring = list([list(), list(), list()])
    fall = list([list(), list(), list()])

    temp_list = list()

    for i in range(0, len(prices)):
        if dates[i].month in (1, 2, 12):
            winter[dates[i].year - 2019].append(prices[i])

        if dates[i].month in (3, 4, 5):
            spring[dates[i].year - 2019].append(prices[i])

        if dates[i].month in (9, 10, 11):
            fall[dates[i].year - 2019].append(prices[i])

        if dates[i].month in (6, 7, 8):
            summer[dates[i].year - 2019].append(prices[i])

    average_summer = list([list(), list(), list()])
    average_winter = list([list(), list(), list()])
    average_spring = list([list(), list(), list()])
    average_fall = list([list(), list(), list()])

    for i in range(0, 3):
        for j in range(0, len(winter[i]) - 30):
            for k in range(j, 30 + j):
                temp_list.append(winter[i][j])

            average_winter[i].append(numpy.median(temp_list))
            temp_list.clear()

        temp_list.clear()

        for j in range(0, len(spring[i]) - 30):
            for k in range(j, 30 + j):
                temp_list.append(spring[i][j])

            average_spring[i].append(numpy.median(temp_list))
            temp_list.clear()

        temp_list.clear()

        for j in range(0, len(summer[i]) - 30):
            for k in range(j, 30 + j):
                temp_list.append(summer[i][j])

            average_summer[i].append(numpy.median(temp_list))
            temp_list.clear()

        temp_list.clear()

        for j in range(0, len(fall[i]) - 30):
            for k in range(j, 30 + j):
                temp_list.append(fall[i][j])

            average_fall[i].append(numpy.median(temp_list))
            temp_list.clear()

    # ----Нахождение и построение скользящей средней(медиана) по разности максимальной и минимальной цен----
    average_thirty_prices = list()
    average_sixty_prices = list()
    average_ninety_prices = list()
    temp_list.clear()

    for i in range(0, len(prices) - 30):
        for j in range(i, 30 + i):
            temp_list.append(prices[j])

        average_thirty_prices.append(numpy.median(temp_list))
        temp_list.clear()

    for i in range(0, len(prices) - 60):
        for j in range(i, 60 + i):
            temp_list.append(prices[j])

        average_sixty_prices.append(numpy.median(temp_list))
        temp_list.clear()

    for i in range(0, len(prices) - 90):
        for j in range(i, 90 + i):
            temp_list.append(prices[j])

        average_ninety_prices.append(numpy.median(temp_list))
        temp_list.clear()
    # ---------------------------------------------------------------------------------------------

    # ----Построение графиков по разности максимальной и минимальной цен----
    plt.figure(1)
    plt.title('Разница между максимальной и минимальной ценой за электроэнергию в один день')
    plt.plot(range(0, len(prices)), prices, label='Разница между max и min ценой')
    plt.plot(range(len(prices) - len(average_thirty_prices), len(prices)), average_thirty_prices,
             label='Скользящая средняя (30 дней)')
    plt.plot(range(len(prices) - len(average_sixty_prices), len(prices)), average_sixty_prices,
             label='Скользящая средняя (60 дней)')
    plt.plot(range(len(prices) - len(average_ninety_prices), len(prices)), average_ninety_prices,
             label='Скользящая средняя (90 дней)')
    plt.xlabel('Дни')
    plt.legend(loc='upper right')
    plt.ylim(-1.0, 200.0)

    plt.figure(2)
    plt.title('Скользящая средняя по временам года')
    plt.subplot(2, 2, 1)
    plt.title('Сравнение по зиме')
    plt.plot(range(0, len(average_winter[0])), average_winter[0], label='За зиму 2019 года')
    plt.plot(range(0, len(average_winter[1])), average_winter[1], label='За зиму 2020 года')
    plt.plot(range(0, len(average_winter[2])), average_winter[2], label='За зиму 2021 года')
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 2)
    plt.title('Сравнение по весне')
    plt.plot(range(0, len(average_spring[0])), average_spring[0], label='За весну 2019 года')
    plt.plot(range(0, len(average_spring[1])), average_spring[1], label='За весну 2020 года')
    plt.plot(range(0, len(average_spring[2])), average_spring[2], label='За весну 2021 года')
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 3)
    plt.title('Сравнение по лету')
    plt.plot(range(0, len(average_summer[0])), average_summer[0], label='За лето 2019 года')
    plt.plot(range(0, len(average_summer[1])), average_summer[1], label='За лето 2020 года')
    plt.plot(range(0, len(average_summer[2])), average_summer[2], label='За лето 2021 года')
    plt.ylim(0, 80)
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 4)
    plt.title('Сравнение по осени')
    plt.plot(range(0, len(average_fall[0])), average_fall[0], label='За осень 2019 года')
    plt.plot(range(0, len(average_fall[1])), average_fall[1], label='За осень 2020 года')
    plt.plot(range(0, len(average_fall[2])), average_fall[2], label='За осень 2021 года')
    plt.legend(loc='upper right')

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
