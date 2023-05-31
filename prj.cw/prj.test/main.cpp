//
// Created by hedge on 22.04.2023.
//

/**
 * \brief Тестовое приложение библиотеки DiabetesAI
 */

#include <regression/regression.hpp>
#include <diabetes_data/diabetes_data.hpp>
#include <plot/plot.hpp>

/**
 * В данной функции мы проверяем основные методы нашей библиотеки
 */
int main() {
    DiabetesData a("C:\\Users\\hedge\\CLionProjects\\DiabetesAI\\prj.cw\\prj.lab\\data","dataset1");
    vector<vector<double>> X1 = a.get_X();
    vector<int> y1 = a.get_y();
    LogisticRegression lg1(X1, y1);
    vector<double> losses = lg1.fit(10);
    for (double losse: losses)
        cout << losse << endl;

    DiabetesData b("C:\\Users\\hedge\\CLionProjects\\DiabetesAI\\prj.cw\\prj.lab\\data","dataset2");
    vector<vector<double>> X = a.get_X();
    vector<int> y = a.get_y();
    LogisticRegression lg2(X, y);
    vector<int> results = LogisticRegression::predict(X);
    Plot::CreateLatexFile(results, y, 10);

}