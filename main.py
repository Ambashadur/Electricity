import pandas
import matplotlib.pyplot as plt
import numpy
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def find_difference_between_prices(path_to_file: str,
                                   index_name: str = 'timestamp',
                                   separator: str = ',',
                                   title: str = 'Name') -> None:

    data = pandas.read_csv(path_to_file, parse_dates=True, index_col=index_name, sep=separator)
    differences = list()
    temp_list = list()

    cur_date = data.index[0].day
    for i in data.index:
        if data.loc[i][data.columns[0]] != '-':
            if i.day != cur_date:
                differences.append(max(temp_list) - min(temp_list))
                cur_date = i.day
                temp_list.clear()

            temp_list.append(float(data.loc[i][data.columns[0]]))

    average_thirty_prices = list()
    average_sixty_prices = list()
    average_ninety_prices = list()
    temp_list.clear()

    for i in range(0, len(differences) - 30):
        for j in range(i, 30 + i):
            temp_list.append(differences[j])

        average_thirty_prices.append(numpy.median(temp_list))
        temp_list.clear()

    for i in range(0, len(differences) - 60):
        for j in range(i, 60 + i):
            temp_list.append(differences[j])

        average_sixty_prices.append(numpy.median(temp_list))
        temp_list.clear()

    for i in range(0, len(differences) - 90):
        for j in range(i, 90 + i):
            temp_list.append(differences[j])

        average_ninety_prices.append(numpy.median(temp_list))
        temp_list.clear()

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(pd.date_range(data.index[0].date(), periods=len(differences)),
            differences,
            label='Difference between the highest and the lowest price')
    ax.plot(pd.date_range(data.index[0].date(), periods=len(differences))[:len(average_thirty_prices)],
            average_thirty_prices,
            label='30 days moving average')
    ax.plot(pd.date_range(data.index[0].date(), periods=len(differences))[:len(average_sixty_prices)],
            average_sixty_prices,
            label='60 days moving average')
    ax.plot(pd.date_range(data.index[0].date(), periods=len(differences))[:len(average_ninety_prices)],
            average_ninety_prices,
            label='90 days moving average')
    ax.legend(loc='upper right')
    ax.set_ylabel('Difference (Euro)')
    ax.set_ylim(-1.0, 120.0)
    plt.show()


def main():
    # Загружаем данные из файла
    data = pandas.read_csv(filepath_or_buffer='data.csv',
                           sep='\t')
    find_difference_between_prices('data.csv',
                                   'DateTime',
                                   '\t',
                                   'The difference between the highest and the lowest price for electricity in one day (Belgium)')

    find_difference_between_prices('NL_price.csv',
                                   'timestamp',
                                   ',',
                                   'The difference between the highest and the lowest price for electricity in one day (Netherlands)')

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
    average_prices_per_interval = list()
    temp_list.clear()

    netherlands_prepared_data = pd.read_csv('NL_price.csv', parse_dates=True, index_col='timestamp').fillna(0.0).to_numpy()
    nl_average_prices_per_interval = list()

    for i in range(0, 95):
        time.append(15 * i)
        for j in range(0, 919):
            if prepared_data[(j * 96) + i][1] != '-':
                temp_list.append(float(prepared_data[(j * 96) + i][1].replace(',', '.')))

        average_prices_per_interval.append(numpy.median(temp_list))
        temp_list.clear()

    temp_list.clear()

    for i in range(0, 95):
        for j in range(0, 960):
            if netherlands_prepared_data[(j * 96) + i] != '-':
                temp_list.append(float(netherlands_prepared_data[(j * 96) + i]))

        nl_average_prices_per_interval.append(numpy.median(temp_list))
        temp_list.clear()

    # Преоброзованние данных при помощи полиномиальной регрессии, для анализ через линейную регрессию на выявление
    # зависимости между временем суток и ценой на электроэнергию
    poly_reg_day = PolynomialFeatures(degree=9)
    X_poly = poly_reg_day.fit_transform(numpy.array(time).reshape((-1, 1)))
    lin_reg = LinearRegression().fit(X_poly, average_prices_per_interval)
    print(lin_reg.score(X_poly, average_prices_per_interval))
    print(lin_reg.intercept_)
    print(lin_reg.coef_)

    time_in_minutes = list()
    for i in time:
        if i // 60 != 0 and i % 60 != 0:
            time_in_minutes.append(f'{i // 60}:{i % 60}')
        elif i // 60 == 0 and i % 60 == 0:
            time_in_minutes.append('00:00')
        elif i % 60 == 0:
            time_in_minutes.append(f'{i // 60}:00')
        else:
            time_in_minutes.append(f'00:{i % 60}')

    plt.figure(6)
    plt.title('Dependence og the price of electricity on the time of day')
    plt.plot(time_in_minutes, average_prices_per_interval, label='Belgium')
    plt.plot(time_in_minutes, nl_average_prices_per_interval, label='Netherlands')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.xticks(time_in_minutes[::5])
    plt.legend(loc='upper right')
    # ---------------------------------------------------------------------------------------------

    # ----Нахождение коэффициентов корреляции между ценами в разных странах
    europe_dataframe = pandas.read_csv(filepath_or_buffer='prices.csv', sep=',')

    one_shift_frame = europe_dataframe.copy()
    one_shift_frame.BE = one_shift_frame.BE.shift(-1)

    two_shift_frame = europe_dataframe.copy()
    two_shift_frame.BE = two_shift_frame.BE.shift(-2)

    three_shift_frame = europe_dataframe.copy()
    three_shift_frame.BE = three_shift_frame.BE.shift(-3)

    four_shift_frame = europe_dataframe.copy()
    four_shift_frame.BE = four_shift_frame.BE.shift(-4)

    five_shift_frame = europe_dataframe.copy()
    five_shift_frame.BE = five_shift_frame.BE.shift(-5)

    one_forward_shift_frame = europe_dataframe.copy()
    one_forward_shift_frame.BE = one_forward_shift_frame.BE.shift(1)

    two_forward_shift_frame = europe_dataframe.copy()
    two_forward_shift_frame.BE = two_forward_shift_frame.BE.shift(2)

    three_forward_shift_frame = europe_dataframe.copy()
    three_forward_shift_frame.BE = three_forward_shift_frame.BE.shift(3)

    four_forward_shift_frame = europe_dataframe.copy()
    four_forward_shift_frame.BE = four_forward_shift_frame.BE.shift(4)

    five_forward_shift_frame = europe_dataframe.copy()
    five_forward_shift_frame.BE = five_forward_shift_frame.BE.shift(5)

    # --- Back shift plot ---

    back_shift_df = pandas.DataFrame({'No shift': europe_dataframe.corr()['BE'][1:],
                                      '15 minutes': one_shift_frame.corr()['BE'][1:],
                                      '30 minutes': two_shift_frame.corr()['BE'][1:],
                                      '45 minutes': three_shift_frame.corr()['BE'][1:],
                                      '60 minutes': four_shift_frame.corr()['BE'][1:],
                                      '75 minutes': five_shift_frame.corr()['BE'][1:]})

    fig, ax = plt.subplots()
    ax.set_title('Shift forward in time')
    ax.imshow(back_shift_df.to_numpy(), cmap='Wistia')

    ax.set_xticks(numpy.arange(6))
    ax.set_yticks(numpy.arange(5))
    ax.set_xticklabels(back_shift_df.columns, fontsize=12)
    ax.set_yticklabels(['Poland', 'Spain', 'Germany', 'France', 'Netherlands'], fontsize=12)

    for i in range(5):
        for j in range(6):
            ax.text(j, i, round(back_shift_df.to_numpy()[i, j], 2),
                    ha='center', va='center', color='black', size='x-large')

    fig.tight_layout()

    # --- Forward shift plot

    forward_shift_df = pandas.DataFrame({'No shift': europe_dataframe.corr()['BE'][1:],
                                         '15 minutes': one_forward_shift_frame.corr()['BE'][1:],
                                         '30 minutes': two_forward_shift_frame.corr()['BE'][1:],
                                         '45 minutes': three_forward_shift_frame.corr()['BE'][1:],
                                         '60 minutes': four_forward_shift_frame.corr()['BE'][1:],
                                         '75 minutes': five_forward_shift_frame.corr()['BE'][1:]})

    fig, ax = plt.subplots()
    ax.set_title('Shift back in time')
    ax.imshow(forward_shift_df.to_numpy(), cmap='Wistia')

    ax.set_xticks(numpy.arange(6))
    ax.set_yticks(numpy.arange(5))
    ax.set_xticklabels(forward_shift_df.columns, fontsize=12)
    ax.set_yticklabels(['Poland', 'Spain', 'Germany', 'France', 'Netherlands'], fontsize=12)

    for i in range(5):
        for j in range(6):
            ax.text(j, i, round(forward_shift_df.to_numpy()[i, j], 2),
                    ha='center', va='center', color='black', size='x-large')

    fig.tight_layout()

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

    plt.figure(13)
    plt.plot(['Winter', 'Spring', 'Summer', 'Autumn'], [numpy.median(average_winter[0]), numpy.median(average_spring[0]),
                                                        numpy.median(average_summer[0]), numpy.median(average_fall[0])],
             label='2019')
    plt.plot(['Winter', 'Spring', 'Summer', 'Autumn'], [numpy.median(average_winter[1]), numpy.median(average_spring[1]),
                                                        numpy.median(average_summer[1]), numpy.median(average_fall[1])],
             label='2020')
    plt.plot(['Winter', 'Spring', 'Summer', 'Autumn'], [numpy.median(average_winter[2]), numpy.median(average_spring[2]),
                                                        numpy.median(average_summer[2]), numpy.median(average_fall[2])],
             label='2021')
    plt.legend(loc='upper right')
    print(numpy.median(average_spring[0]))
    print(numpy.median(average_spring[1]))
    print(numpy.median(average_spring[2]))
    print('------')
    print(numpy.median(average_summer[0]))
    print(numpy.median(average_summer[1]))
    print(numpy.median(average_summer[2]))
    print('------')
    print(numpy.median(average_fall[0]))
    print(numpy.median(average_fall[1]))
    print('------')
    print(numpy.median(average_winter[0]))
    print(numpy.median(average_winter[1]))
    print(numpy.median(average_winter[2]))
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

    # i - дата
    for i, total_energy, solar_energy, wind_energy in prepared_data:
        if solar_energy != '-':
            scholar_energy_per_interval.append(float(solar_energy.replace(',', '.')))
        else:
            scholar_energy_per_interval.append(solar_energy)

        if wind_energy != '-':
            wind_energy_per_interval.append(float(wind_energy.replace(',', '.')))
        else:
            wind_energy_per_interval.append(wind_energy)

        if total_energy != '-':
            energy_per_interval.append(float(total_energy.replace(',', '.')))
        else:
            energy_per_interval.append(total_energy)

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

    # ---- Коэффициенты корреляции
    coeff_df = pandas.DataFrame({'price': price_per_interval, 'solar': scholar_energy_per_interval,
                                 'wind': wind_energy_per_interval, 'total': energy_per_interval})

    one_shift_frame = coeff_df.copy()
    one_shift_frame.price = one_shift_frame.price.shift(-1)
    two_shift_frame = coeff_df.copy()
    two_shift_frame.price = two_shift_frame.price.shift(-2)
    three_shift_frame = coeff_df.copy()
    three_shift_frame.price = three_shift_frame.price.shift(-3)
    four_shift_frame = coeff_df.copy()
    four_shift_frame.price = four_shift_frame.price.shift(-4)
    five_shift_frame = coeff_df.copy()
    five_shift_frame.price = five_shift_frame.price.shift(-5)
    one_forward_shift_frame = coeff_df.copy()
    one_forward_shift_frame.price = one_forward_shift_frame.price.shift(1)
    two_forward_shift_frame = coeff_df.copy()
    two_forward_shift_frame.price = two_forward_shift_frame.price.shift(2)
    three_forward_shift_frame = coeff_df.copy()
    three_forward_shift_frame.price = three_forward_shift_frame.price.shift(3)
    four_forward_shift_frame = coeff_df.copy()
    four_forward_shift_frame.price = four_forward_shift_frame.price.shift(4)
    five_forward_shift_frame = coeff_df.copy()
    five_forward_shift_frame.price = five_forward_shift_frame.price.shift(5)

    fig, ax = plt.subplots()
    ax.set_title('Forward shift in time')

    back_shift_df = pd.DataFrame({'No shift': coeff_df.corr().price[1:],
                                  '15 minutes': one_shift_frame.corr().price[1:],
                                  '30 minutes': two_shift_frame.corr().price[1:],
                                  '45 minutes': three_shift_frame.corr().price[1:],
                                  '60 minutes': four_shift_frame.corr().price[1:],
                                  '75 minutes': five_shift_frame.corr().price[1:]})

    ax.imshow(back_shift_df.to_numpy(), cmap='Wistia')

    ax.set_xticks(numpy.arange(6))
    ax.set_yticks(numpy.arange(3))
    ax.set_xticklabels(back_shift_df.columns, fontsize=12)
    ax.set_yticklabels(['Total solar generation', 'Total wind generation', 'Total consumption'], fontsize=12)

    for i in range(3):
        for j in range(6):
            ax.text(j, i, round(back_shift_df.to_numpy()[i, j], 2),
                    ha='center', va='center', color='black', size='x-large')

    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.set_title('Back shift in time')

    forward_shift_df = pd.DataFrame({'No shift': coeff_df.corr().price[1:],
                                     '15 minutes': one_forward_shift_frame.corr().price[1:],
                                     '30 minutes': two_forward_shift_frame.corr().price[1:],
                                     '45 minutes': three_forward_shift_frame.corr().price[1:],
                                     '60 minutes': four_forward_shift_frame.corr().price[1:],
                                     '75 minutes': five_forward_shift_frame.corr().price[1:]})

    ax.imshow(forward_shift_df.to_numpy(), cmap='Wistia')

    ax.set_xticks(numpy.arange(6))
    ax.set_yticks(numpy.arange(3))
    ax.set_xticklabels(forward_shift_df.columns, fontsize=12)
    ax.set_yticklabels(['Total solar generation', 'Total wind generation', 'Total consumption'], fontsize=12)

    for i in range(3):
        for j in range(6):
            ax.text(j, i, round(forward_shift_df.to_numpy()[i, j], 2),
                    ha='center', va='center', color='black', size='x-large')

    fig.tight_layout()
    # ----------------------------------------------

    general_regression_list = list()

    # Регрессия для солнечной энергии
    temp_list = numpy.array(scholar_energy_per_interval).reshape((-1, 1))
    general_regression_list.append(temp_list)
    # model = LinearRegression().fit(temp_list, price_per_interval)

    temp_list = numpy.array(wind_energy_per_interval).reshape((-1, 1))
    general_regression_list.append(temp_list)
    # wind_model = LinearRegression().fit(temp_list, price_per_interval)

    # Регрессия для общего потребления энергии
    temp_list = numpy.array(energy_per_interval).reshape((-1, 1))
    general_regression_list.append(temp_list)
    # energy_model = LinearRegression().fit(temp_list, price_per_interval)

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
    plt.figure(2)
    plt.title('Moving average by seasons')
    plt.subplot(2, 2, 1)
    plt.title('Winter')
    plt.plot(range(0, len(average_winter[0])), average_winter[0], label='2019')
    plt.plot(range(0, len(average_winter[1])), average_winter[1], label='2020')
    plt.plot(range(0, len(average_winter[2])), average_winter[2], label='2021')
    plt.xlabel('Days')
    plt.ylabel('Difference (Euro)')
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 2)
    plt.title('Spring')
    plt.plot(range(0, len(average_spring[0])), average_spring[0], label='2019')
    plt.plot(range(0, len(average_spring[1])), average_spring[1], label='2020')
    plt.plot(range(0, len(average_spring[2])), average_spring[2], label='2021')
    plt.xlabel('Days')
    plt.ylabel('Difference (Euro)')
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 3)
    plt.title('Summer')
    plt.plot(range(0, len(average_summer[0])), average_summer[0], label='2019')
    plt.plot(range(0, len(average_summer[1])), average_summer[1], label='2020')
    plt.plot(range(0, len(average_summer[2])), average_summer[2], label='2021')
    plt.xlabel('Days')
    plt.ylabel('Difference (Euro)')
    plt.ylim(0, 80)
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 4)
    plt.title('Autumn')
    plt.plot(range(0, len(average_fall[0])), average_fall[0], label='2019')
    plt.plot(range(0, len(average_fall[1])), average_fall[1], label='2020')
    plt.plot(range(0, len(average_fall[2])), average_fall[2], label='2021')
    plt.xlabel('Days')
    plt.ylabel('Difference (Euro)')
    plt.legend(loc='upper right')

    plt.show()
    # ---------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
