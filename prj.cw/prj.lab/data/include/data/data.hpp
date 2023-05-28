//
// Created by hedge on 13.05.2023.
//

#pragma once
#ifndef DIABETES_AI_HPP_09052023
#define DIABETES_AI_HPP_09052023

#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>


using namespace std;

class DiabetesData {
public:

    explicit DiabetesData(const string &data_name) {
        load_data_from_file(data_name);
    }

    void load_data_from_file(const string &data_name) {
        fstream fin;
        fin.open(data_name + ".csv", ios::in);
        string line;
        vector<vector<string>> parsedCsv;
        if (fin.fail()) {
            cout << "NOT OPEN";
        }
        while (getline(fin, line)) {
            stringstream lineStream(line);
            string cell;
            vector<string> parsedRow;
            while (getline(lineStream, cell, ',')) {
                parsedRow.push_back(cell);
            }

            parsedCsv.push_back(parsedRow);
        }
        fin.close();
        dataset_ = parsedCsv;

        for (int i = 1; i < dataset_.size(); i++) {
            vector<double> cell;
            for (int j = 0; j < dataset_[1].size(); j++) {
                if (j != dataset_[1].size() - 1) {
                    cell.push_back(stod(dataset_[i][j]));
                } else if (j == dataset_[1].size() - 1) {
                    y_.push_back(stoi(dataset_[i][j]));
                }
            }
            X_.push_back(cell);
        }
        X_ = data_normalization(X_);
    }

    explicit DiabetesData(const vector<vector<double>> &features) {
        if (features.size() != dataset_[0].size() || features.empty())
            throw runtime_error(
                    "The number of parameters does not match the number of dataset parameters, check the correctness of the entered data and check which dataset you submitted");

        features_ = data_normalization(features);
    }


    static vector<vector<double>> data_normalization(vector<vector<double>> X) {

        vector<double> avarage;
        for (int j = 0; j < X[0].size(); j++) {
            double summ = 0;
            for (auto &i: X) {
                summ += i[j];
            }
            avarage.push_back(summ / X.size());
        }

        vector<double> deviation;
        for (int j = 0; j < X[0].size(); j++) {
            double summ = 0;
            for (auto &i: X)
                summ += pow((i[j] - avarage[j]), 2);

            deviation.push_back(sqrt(summ / (X.size() - 1)));
        }

        for (int j = 0; j < X[0].size(); j++) {
            for (auto &i: X) {
                i[j] = (i[j] - avarage[j]) / deviation[j];
                if (i[j] > 3 * deviation[j]) {
                    i[j] = 0;
                }
            }
        }
        return X;
    }

    vector<vector<double>> features_;
    vector<vector<string>> dataset_;
    vector<vector<double>> X_;
    vector<int> y_;
private:

};

#endif
