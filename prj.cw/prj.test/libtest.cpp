//
// Created by hedge on 22.04.2023.
//

/**
 * @file libtest.cpp
 * \brief Тестовое приложение библиотеки DiabetesAI
 * \details В данном приложении мы тестируем основные функции и методы библиотеки, а именно: обработка датасета с помощью класса DiabetesData, обучение модели с помощью класса LogisticRegression по встроенному датасету 1 и кол-вом итераций = 10, вывод функции потерь в консоль, прогнозирование диабета по встроенному датасету 2 и вывод статистики модели c помощью класса Plot
 */

#include <regression/regression.hpp>
#include <diabetes_data/diabetes_data.hpp>
#include <plot/plot.hpp>

/**
 * @param argc Название .exe файла для запуска тестового приложения
 * @param argv полный путь до папки с датасетами вида "C:\...\data"
 *
 *
 * Обрабатываем встроенный датасет 1
 * \code
 *  DiabetesData a(argv[1],"dataset1");
    vector<vector<double>> X1 = a.get_X();
    vector<int> y1 = a.get_y();
    LogisticRegression lg1(X1, y1);
 * \endcode
 * Выводим функции потерь в консоль
 * \code
    vector<double> losses = lg1.fit(10);
    for (double losse: losses)
        cout << losse << endl;
 * \endcode
 * Обрабатываем встроенный датасет 2
 * \code
    DiabetesData b(argv[1],"dataset2");
    vector<vector<double>> X = a.get_X();
    vector<int> y = a.get_y();
 * \endcode
 *  Берем предсказания датасета 2 из обученной модели
 *  \code
    LogisticRegression lg2(X, y);
    vector<int> results = LogisticRegression::predict(X);
 *  \endcode
 *  Создаем statistic.tex файл
 * \code
    Plot::CreateLatexFile(results, y, 10);
 * \endcode
 */
void main(int argc, char* argv[]) {
    DiabetesData a(argv[1], "dataset1");
    vector<vector<double>> X1 = a.get_X();
    vector<int> y1 = a.get_y();
    LogisticRegression lg1(X1, y1);
    vector<double> losses = lg1.fit(5);
    for (double losse: losses)
        cout << losse << endl;

    DiabetesData b(argv[1], "dataset2");
    vector<vector<double>> X = a.get_X();
    vector<int> y = a.get_y();
    LogisticRegression lg2(X, y);
    vector<int> results = LogisticRegression::predict(X);
    Plot::CreateLatexFile(5, results, y);
}