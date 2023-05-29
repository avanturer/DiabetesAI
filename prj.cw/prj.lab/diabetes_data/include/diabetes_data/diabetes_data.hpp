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
 * \brief Класс DiabetesData предназначен для подготовки датасета
 * \details Класс DiabetesData предоставляет возможность подготовить датасет для дальнейшей работы с классом LogisticRegression и Plot, путем разбиения датасета на двумерный вектор double и одномерный вектор int: вектор X_ содержит в себе строки датасета, не включая названия столбцов и исход(outcome), и у_, содержит в себе лишь исход(outcome)
 */
class DiabetesData {
public:
    /**
     * Конструктор по умолчанию для создания объекта класса
     */
    DiabetesData() = default;
    /**
     * Конструктор принимает на вход полный путь к директории с датасетами и название датасета, после чего он вызывает функцию ```load_data_from_file```
     * @param[in] file_path Полный путь к репозиторию с датасетами вида: "С:\\...\\data"
     * @param[in] data_name Название файла .csv с датасетои вида: "dataset"
     */

    explicit DiabetesData(const string &file_path, const string &data_name);

    /**
     * Конструктор принимает на вход выборку и нормализует её с помощью ```data_normalization```
     * @param[in] features двумерный вектор double состоящий из одной строчки и n столбцов (n = кол-ву столбцов датасета  - 1)
     */

    explicit DiabetesData(const vector<vector<double>> &features);
    /**
     * Функция принимает на вход полный путь к директории с датасетами и название датасета, обрабатывает его, и записывает результат в X_ и y_
     * @param file_path Полный путь к репозиторию с датасетами вида: "С:\\...\\data"
     * @param data_name Название файла .csv с датасетои вида: "dataset"
     */
    void load_data_from_file(const string &file_path, const string &data_name);
    /**
     * Функция принимает на вход выборку Х_ и нормализует её методом z-масштабирования
     * @param X Двумерный вектор типа double, содержащий в себе выборку
     */
    static vector<vector<double>> data_normalization(vector<vector<double>> X);

    /**
     * Функция возвращающая приватный член Х_
     * @return двумерный вектор типа double
     */
    vector<vector<double>> get_X ();
    /**
     * Функция возвращающая приватный член y_
     * @return вектор типа int
     */
    vector<int> get_y ();

private:
    /**
     * Двумерный вектор типа double, содержащий в себе выборку для использования на уже обученной модели
     */
    vector<vector<double>> features_;
    /**
     * Двумерный вектор типа double, содержащий в себе выборку
     */
    vector<vector<string>> dataset_;
    /**
     * Двумерный вектор типа double, содержащий в себе __нормализованную__ выборку
     */
    vector<vector<double>> X_;
    /**
     * Вектор типа int, содержащий в себе итог(outсome)
     */
    vector<int> y_;
};

#endif
