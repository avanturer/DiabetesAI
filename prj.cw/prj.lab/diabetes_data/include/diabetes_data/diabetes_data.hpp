//
// Created by hedge on 13.05.2023.
//

#pragma once
#ifndef DIABETES_AI_HPP_09052023DDAT
#define DIABETES_AI_HPP_09052023DDAT

#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>


using namespace std;

/**
 * \brief Класс DiabetesData для подготовки датасета
 * \details Класс DiabetesData предоставляет возможность подготовить датасет для дальнейшей работы с классом LogisticRegression и Plot, путем разбиения датасета на двумерный массив double и одномерный массив int: массив X содержащий в себе строки датасета, не включая названия столбцов и исход(outcome), и у, содержащий в себе лишь исход(outcome)
 */
class DiabetesData {
public:

    /**
     * конструктор принимает на вход полный путь к директории с датасетами и название датасета, после чего он вызывает функцию ```load_data_from_file```
     * @param[in] file_path Полный путь к репозиторию с датасетами вида: "С:\\...\\data"
     * @param[in] data_name Название файла .csv с датасетои вида: "dataset"
     */

    explicit DiabetesData(const string &file_path, const string &data_name);

    /**
     * Этот конструктор принимает на вход двумерный вектор double и нормализует его с помощью ```data_normalization```
     * @param[in] features двумерный вектор double состоящий из одной строчки и n столбцов (n = кол-ву столбцов датасета  - 1)
     */

    explicit DiabetesData(const vector<vector<double>> &features);

    void load_data_from_file(const string &file_path, const string &data_name);

    static vector<vector<double>> data_normalization(vector<vector<double>> X);

    /**
     * Функция возвращающая приватный член Х_
     * @return двумерный вектор double
     */
    vector<vector<double>> get_X ();

    vector<int> get_y ();

private:

    vector<vector<double>> features_;
    vector<vector<string>> dataset_;
    /**
     * двумерный вектор double содержащий в себе датасет не включающий итог(outсome) и названия столбцов
     */
    vector<vector<double>> X_;
    /**
     * вектор int содержащий в себе итог(outсome)
     */
    vector<int> y_;
};

#endif
