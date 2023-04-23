//
// Created by hedge on 22.04.2023.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>

using namespace std;

class DiabetesData {
public:
    // Constructor
    DiabetesData() = default;

    DiabetesData(double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8) {
        feature1 = f1;
        feature2 = f2;
        feature3 = f3;
        feature4 = f4;
        feature5 = f5;
        feature6 = f6;
        feature7 = f7;
        feature8 = f8;
    }

    void load_data_from_file() {
        fstream fin;
        fin.open("dataset.csv", ios::in);
        string line;
        std::vector<std::vector<std::string> > parsedCsv;
        if (fin.fail()) {
            std::cout << "NOT OPEN";
        }
        while (std::getline(fin, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            std::vector<std::string> parsedRow;
            while (std::getline(lineStream, cell, ',')) {
                parsedRow.push_back(cell);
            }

            parsedCsv.push_back(parsedRow);
        }
        fin.close();
        dataset = parsedCsv;

        for (int i = 1; i < dataset.size(); i++) {
            vector<double> cell;
            for (int j = 0; j < 9; ++j) {
                if (j != 8) {
                    cell.push_back(stod(dataset[i][j]));
                } else {
                    y.push_back(stoi(dataset[i][j]));
                }
            }
            X.push_back(cell);
        }
        data_normalization();
    }

    void data_normalization() {

        vector<double> avarage;
        for (int i = 0; i < X.size(); i++) {
            double summ = 0;
            for (int j = 0; j < X[0].size(); ++j) {
                summ += X[i][j];
            }
            avarage.push_back(summ / X[0].size());
        }

        vector<double> deviation;
        for (int i = 0; i < X.size(); i++) {
            double summ = 0;
            for (int j = 0; j < X[0].size(); ++j) {
                summ += pow((X[i][j] - avarage[i]), 2);
            }
            deviation.push_back(sqrt(summ / X[0].size()));
        }

        for (int i = 0; i < X.size(); ++i) {
            for (int j = 0; j < X[0].size(); ++j) {
                X[i][j] = (X[i][j] - avarage[i]) / deviation[i];
            }
        }
    }

    double feature1;
    double feature2;
    double feature3;
    double feature4;
    double feature5;
    double feature6;
    double feature7;
    double feature8;
    std::vector<std::vector<string>> dataset;
    std::vector<std::vector<double>> X;
    std::vector<int> y;
private:

};

class LogisticRegression {
public:

    LogisticRegression(std::vector<std::vector<double>> X, std::vector<int> y) {
        this->X = X;
        this->y = y;
        for (int i = 0; i < X.size(); ++i) {
            double a = (rand() % 1000);
            w.push_back(a / 1000);
        }

    }

    std::vector<std::vector<double>> X;
    std::vector<int> y;
    vector<double> w;

};

int main() {

}