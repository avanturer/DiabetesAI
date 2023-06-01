//
// Created by hedge on 09.05.2023.
//
#pragma once
#ifndef DIABETES_AI_HPP_09052023PLOT
#define DIABETES_AI_HPP_09052023PLOT

#include <iostream>
#include <vector>
#include <fstream>
#include <string>


using namespace std;

/**
 * \brief Класс Plot предназначен для создания .tex файла с информацией о модели
 * \details Класс Plot предоставляет возможность вывести данные о текущей модели в виде .tex файла, а именно: функция потерь и confusion matrix.
 */

class Plot {
public:
    /**
    * Конструктор по умолчанию для создания объекта класса
    */
    Plot() = default;
    /**
     * Функция создает statistic.tex файл по результатам работы модели модели, содержащий в себе функцию потерь и confusion matrix
     * @param max_iter переменная типа int, содержащая в себе кол-во итераций, которая совершила модель при обучении (по умолчанию 100)
     * @param results вектор типа int, содержащий в себе результаты предсказаний модели (по умолчанию пустой)
     * @param y вектор типа int, содержащий в себе действительный результат (по умолчанию пустой)
     * Пример:
     *
     * ![Пример statistic.tex](C:\Users\hedge\CLionProjects\DiabetesAI\prj.cw\prj.lab\data\img_1.png)
     *
     * ![Пример statistic.tex](C:\Users\hedge\CLionProjects\DiabetesAI\prj.cw\prj.lab\data\img_2.png)
     */
    static void CreateLatexFile(int max_iter = 100, vector<int> results = {}, vector<int> y = {});
private:
};

#endif
