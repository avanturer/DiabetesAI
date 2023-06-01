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
 * Обрабатываем встроенный датасет 1
 * \code
 *  DiabetesData a("C:\\Users\\hedge\\CLionProjects\\DiabetesAI\\prj.cw\\prj.lab\\data","dataset1");
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
    DiabetesData b("C:\\Users\\hedge\\CLionProjects\\DiabetesAI\\prj.cw\\prj.lab\\data","dataset2");
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
int main() {
    DiabetesData a("C:\\Users\\hedge\\CLionProjects\\DiabetesAI\\prj.cw\\prj.lab\\data", "dataset1");
    vector<vector<double>> X1 = a.get_X();
    vector<int> y1 = a.get_y();
    LogisticRegression lg1(X1, y1);
    vector<double> losses = lg1.fit(10);
    for (double losse: losses)
        cout << losse << endl;

    DiabetesData b("C:\\Users\\hedge\\CLionProjects\\DiabetesAI\\prj.cw\\prj.lab\\data", "dataset2");
    vector<vector<double>> X = a.get_X();
    vector<int> y = a.get_y();
    LogisticRegression lg2(X, y);
    vector<int> results = LogisticRegression::predict(X);
    Plot::CreateLatexFile(10, results, y);
}