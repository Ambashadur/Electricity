import pandas
import matplotlib.pyplot as plt
import numpy
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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
    # Разница цен
    prices = list()
    dates = list()
    temp_list = list()

    cur_date = prepared_data[0][0].split()
    dates.append(datetime.strptime(cur_date[0], '%m/%d/%Y'))

    for i, j in prepared_data:
        if j != '-':
            # Разделяем строку на дату и время
            i = i.split()

            if i[0] != cur_date[0]:
                prices.append(max(temp_list) - min(temp_list))
                dates.append(datetime.strptime(i[0], '%m/%d/%Y'))

                cur_date = i
                temp_list.clear()

            temp_list.append(float(j.replace(',', '.')))
    # ---------------------------------------------------------------------------------------------

    # ----Нахождение коэффициента корреляции между временем суток и ценой на электроэнергию----
    time = list()
    # Массив медианных цен в определённый период за 1 месяц(пока январь 2019)
    average_prices_per_interval = list()
    temp_list.clear()

    for i in range(0, 95):
        time.append(15 * i)
        for j in range(0, 919):
            if prepared_data[(j * 96) + i][1] != '-':
                temp_list.append(float(prepared_data[(j * 96) + i][1].replace(',', '.')))

        average_prices_per_interval.append(numpy.median(temp_list))
        temp_list.clear()

    # Преоброзованние данных при помощи полиномиальной регрессии, для анализ через линейную регрессию на выявление
    # зависимости между временем суток и ценой на электроэнергию
    poly_reg_day = PolynomialFeatures(degree=9)
    X_poly = poly_reg_day.fit_transform(numpy.array(time).reshape((-1, 1)))
    lin_reg = LinearRegression().fit(X_poly, average_prices_per_interval)
    print(lin_reg.score(X_poly, average_prices_per_interval))
    print(lin_reg.intercept_)
    print(lin_reg.coef_)

    plt.figure(6)
    plt.title('Зависимость цены на электроэнергию от времени суток')
    plt.plot(time, average_prices_per_interval, label='Усреднённые данные')
    plt.plot(time, lin_reg.predict(X_poly), label='Данные полученные из регрессионного анализа')
    plt.legend(loc='upper right')
    # ---------------------------------------------------------------------------------------------

    # ----Нахождение скользящей средней для сезонов----
    summer = list([list(), list(), list()])
    winter = list([list(), list(), list()])
    spring = list([list(), list(), list()])
    fall = list([list(), list(), list()])

    temp_list.clear()

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
    # ---------------------------------------------------------------------------------------------

    # ----Построение графиков зависимости цен от потребления солнечной, ветренной энергий и потреблении энергии----
    energy_data = pandas.read_csv(filepath_or_buffer='energy_data.csv',
                                  sep=';')

    # Извлечение всех цен из файла (даже не зарегестрированных)
    price_per_interval = list()

    for i, j in prepared_data:
        if j != '-':
            price_per_interval.append(float(j.replace(',', '.')))
        else:
            price_per_interval.append(j)

    # Извлечение данных по солнечной и ветрянной электроэнергии
    prepared_data = energy_data.to_numpy()

    scholar_energy_per_interval = list()
    wind_energy_per_interval = list()
    energy_per_interval = list()

    # i - дата, j - потребление энергии, k - солнечная энергия, z - ветренная энергия
    for i, j, k, z in prepared_data:
        if k != '-':
            scholar_energy_per_interval.append(float(k.replace(',', '.')))
        else:
            scholar_energy_per_interval.append(k)

        if z != '-':
            wind_energy_per_interval.append(float(z.replace(',', '.')))
        else:
            wind_energy_per_interval.append(z)

        if j != '-':
            energy_per_interval.append(float(j.replace(',', '.')))
        else:
            energy_per_interval.append(j)

    # Удаляем все строки с незарегестрированными данными
    index = 0
    while index != len(price_per_interval):
        if price_per_interval[index] == '-' or scholar_energy_per_interval[index] == '-' \
                or wind_energy_per_interval[index] == '-' or energy_per_interval[index] == '-':
            price_per_interval.pop(index)
            scholar_energy_per_interval.pop(index)
            wind_energy_per_interval.pop(index)

            energy_per_interval.pop(index)
        else:
            index += 1

    # Удаляем все не 'лишние данные', они за те дни, в которые мы не знаем цену
    del scholar_energy_per_interval[len(price_per_interval):]
    del wind_energy_per_interval[len(price_per_interval):]
    del energy_per_interval[len(price_per_interval):]

    general_regression_list = list()

    # Регрессия для солнечной энергии
    plt.figure(3)
    plt.title('Зависимость цен на электреэнергию от потребления солнечной энергии')

    temp_list = numpy.array(scholar_energy_per_interval).reshape((-1, 1))
    general_regression_list.append(temp_list)
    model = LinearRegression().fit(temp_list, price_per_interval)
    print('\nРегрессионный анализ для цен и солнечной энергии')
    print('Коэффициент детерминированности (R^2) - ', model.score(temp_list, price_per_interval))
    print('b0 = ', model.intercept_)
    print('b1 = ', model.coef_)

    plt.scatter(scholar_energy_per_interval, price_per_interval, label='Реальные данные')
    plt.scatter(scholar_energy_per_interval, model.predict(temp_list),
                label='Данные полученные из регрессионной модели')
    plt.legend(loc='upper right')

    # Регрессия для ветрянной энергии
    plt.figure(4)
    plt.title('Зависимость цен на электроэнергию от потребления ветрянной энергии')

    temp_list = numpy.array(wind_energy_per_interval).reshape((-1, 1))
    general_regression_list.append(temp_list)
    wind_model = LinearRegression().fit(temp_list, price_per_interval)
    print('\nРегрессионный анализ для цен и ветрянной энергии')
    print('Коэффициент детерминированности (R^2) - ', wind_model.score(temp_list, price_per_interval))
    print('b0 = ', wind_model.intercept_)
    print('b1 = ', wind_model.coef_)

    plt.scatter(wind_energy_per_interval, price_per_interval, label='Реальные данные')
    plt.scatter(wind_energy_per_interval, wind_model.predict(temp_list),
                label='Данные полученные из регрессионной модели')
    plt.legend(loc='upper right')

    # Регрессия для общего потребления энергии
    plt.figure(5)
    plt.title('Зависимость цен на электроэнергию от потребление всей энергии')

    temp_list = numpy.array(energy_per_interval).reshape((-1, 1))
    general_regression_list.append(temp_list)
    energy_model = LinearRegression().fit(temp_list, price_per_interval)
    print('\nРегрессионный анализ для цен и потребления энергии')
    print('Коэффициент детерминированности (R^2) - ', energy_model.score(temp_list, price_per_interval))
    print('b0 = ', energy_model.intercept_)
    print('b1 = ', energy_model.coef_)

    plt.scatter(energy_per_interval, price_per_interval, label='Реальные данные')
    plt.scatter(energy_per_interval, energy_model.predict(temp_list),
                label='Данные полученные из регрессионной модели')
    plt.legend(loc='upper right')

    general_model = LinearRegression().fit(numpy.column_stack([general_regression_list[0], general_regression_list[1],
                                           general_regression_list[2]]), price_per_interval)
    print('\nОбщий регрессионный анализ')
    print('R^2 - ', general_model.score(numpy.column_stack([general_regression_list[0], general_regression_list[1],
                                        general_regression_list[2]]), price_per_interval))
    print('b0 = ', general_model.intercept_)
    print('b^ = ', general_model.coef_)

    temp_list = list()
    # ---------------------------------------------------------------------------------------------

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
    # ---------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
