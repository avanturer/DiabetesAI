//
// Created by hedge on 09.05.2023.
//

#pragma once
#ifndef DIABETES_AI_HPP_09052023REG
#define DIABETES_AI_HPP_09052023REG

#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

using namespace std;

/**
 * \brief Класс LogisticRegression предназначен для обучения модели на основе лог.регрессии
 * \details Класс LogisticRegression предоставляет возможность обучить модель, основанную на лог.регрессии, путем градиентного спуска по функции потерь с использованием R2 - регуляризации
 */
class LogisticRegression {
public:
    /**
     * Конструктор сохраняет поступившие данные и рандомит веса для дальнейшей работы
     * @param X Двумерный вектор типа double, содержащий в себе __нормализованную__ выборку
     * @param y Вектор типа int, содержащий в себе итог(outсome)
     */
    LogisticRegression(const vector<vector<double>> &X, const vector<int> &y);

    /**
     * Функция подсчитывает логиты всей выборки по текущим весам и возвращает их
     * @param X __Транспонированный__ двумерный вектор типа double, содержащий в себе выборку
     * @param w Вектор весов типа double
     * @return Двумерный вектор логитов типа double
     * \f[
     * \sum_{i=0}^{n} (b + w_i*x_i))^2
     * \f]
     */
    static vector<vector<double>> logit(vector<vector<double>> X, vector<double> w);

    /**
     * Функция подсчитывает сигмоиды по полученным логитам
     * @param logits Двумерный вектор логитов типа double
     * @return двумерный вектор типа double, содержащий в себе сигмоиды для каждого логита
     */
    static vector<vector<double>> sigmoid(vector<vector<double>> logits);

    vector<double> fit(int max_iter = 100, double lr = 0.1);

    /**
     * Функция подсчитывает "функцию потерь" нашей логистической регрессии с использованием l2 - регуляризации, основываясь на у исходе(outcome) и сигмоидах.
     * @param y Вектор типа int, содержащий в себе итог(outсome) для нашей выборки
     * @param z Двумерный вектор типа double, содержащий в себе сигмоиды для каждого логита
     * @return Число типа double показывающее текущее значение функции потерь
     */
    double loss(vector<int> y, vector<vector<double>> z);

    static void save_weights(const vector<double> &weights);

    static vector<vector<double>> predict_proba(vector<vector<double>> feauters);

    static vector<int> predict(const vector<vector<double>> &feauters, double threshold = 0.5);

    static int model_accuracy(vector<int> results, vector<int> y);

    static void saveLossToCSV(const vector<double> &losses);

private:
    /**
     * Двумерный вектор типа double, содержащий в себе __нормализованную__ выборку
     */
    vector<vector<double>> X_;
    /**
     * Вектор типа int, содержащий в себе итог(outсome)
     */
    vector<int> y_;
    /**
     * Вектор типа double, содержащий в себе веса наней модели
     */
    vector<double> w_;
    /**
     * Вектор типа double, содержащий в себе результаты функции потерь на каждой итерации модели при обучении
     */
    vector<double> losses_;
    /**
     * Переменная типа int, содержащая в себе кол-во итераций, которые будет совершать модель при обучении
     */
    int max_iter_ = 0;
    /**
     * Переменная типа double, содержащая в себе коэффициент скорости обучения модели
     *
     * ![Пример learning rate](C:\Users\hedge\CLionProjects\DiabetesAI\prj.cw\prj.lab\data\img.png)
     */
    double lr_ = 0;
};

#endif
