//
// Created by hedge on 22.04.2023.
//

#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>

using namespace std;

class DiabetesData {
public:
    // Constructor
    DiabetesData() {
        load_data_from_file();
    }

    DiabetesData(double f1, double f2, double f3, double f4, double f5, double f6, double f7, double f8) {
        features = {{f1},
                    {f2},
                    {f3},
                    {f4},
                    {f5},
                    {f6},
                    {f7},
                    {f8}};
        features = data_normalization(features);
    }

    void load_data_from_file() {
        fstream fin;
        fin.open("dataset.csv", ios::in);
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
        dataset = parsedCsv;

        for (int i = 1; i < dataset.size(); i++) {
            vector<double> cell;
            for (int j = 0; j < 9; j++) {
                if (j != 8) {
                    cell.push_back(stod(dataset[i][j]));
                } else {
                    y.push_back(stoi(dataset[i][j]));
                }
            }
            X.push_back(cell);
        }
        X = data_normalization(X);
    }

    vector<vector<double>> data_normalization(vector<vector<double>> X) {

        vector<double> avarage;
        for (int i = 0; i < X.size(); i++) {
            double summ = 0;
            for (int j = 0; j < X[0].size(); j++) {
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

        for (int i = 0; i < X.size(); i++) {
            for (int j = 0; j < X[0].size(); j++) {
                X[i][j] = (X[i][j] - avarage[i]) / deviation[i];
            }
        }
        return X;
    }

    vector<vector<double>> features;
    vector<vector<string>> dataset;
    vector<vector<double>> X;
    vector<int> y;
private:

};

class LogisticRegression {
public:
    vector<vector<double>> X;
    vector<int> y;
    vector<double> w;
    vector<double> losses;

    LogisticRegression(const vector<vector<double>> X, const vector<int> y) {
        this->X = X;
        this->y = y;
        for (int i = 0; i < X[0].size(); i++) {
            double a = (rand() % 1000);
            w.push_back(a / 1000);
        }

    }

    vector<vector<double>> logit(vector<vector<double>> X, vector<double> w) {
        int m = X.size();
        int n = X[0].size();
        vector<vector<double>> logits(m, vector<double>(1, 0.0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < 1; j++) {
                for (int k = 0; k < n; k++) {
                    logits[i][j] += X[i][k] * w[k];
                }
            }
        }
        return logits;
    }

    vector<vector<double>> sigmoid(vector<vector<double>> logits) {
        vector<vector<double>> sigmoids;
        for (int i = 0; i < logits.size(); i++) {
            vector<double> sigmoid;
            for (int j = 0; j < logits[0].size(); j++) {
                sigmoid.push_back(1 / (1 + exp(-logits[i][j])));
            }
            sigmoids.push_back(sigmoid);
        }
        return sigmoids;
    }

    vector<double> fit(int max_iter = 30, double lr = 0.05) {
        vector<vector<double>> X_train = X;

        for (int i = 0; i < X_train.size(); i++) {
            X_train[i].insert(X_train[i].begin(), 1);
        }

        vector<vector<double>> X_trainT(X_train[0].size(), vector<double>(X_train.size(), 0.0));
        for (int i = 0; i < X_train.size(); i++) {
            for (int j = 0; j < X_train[0].size(); j++) {
                X_trainT[j][i] = X_train[i][j];
            }
        }

        for (int iter = 0; iter <= max_iter; iter++) {
            vector<vector<double>> z;
            for (int i = 0; i < sigmoid(logit(X, w)).size(); i++) {
                z.push_back(sigmoid(logit(X, w))[i]);
            }
            int m = X_trainT.size();
            int n = X_trainT[0].size();
            int p = z[0].size();

            vector<vector<double>> grad(m, vector<double>(p, 0.0));
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < p; j++) {
                    for (int k = 0; k < n; k++) {
                        grad[i][j] += (X_trainT[i][k] * (z[k][0] - y[k]));
                    }
                    grad[i][j] /= y.size();
                }
            }

            for (int i = 0; i < w.size(); i++) {
                w[i] -= (grad[i][0] * lr);
            }
            if (losses.size() != 0 and (abs(loss(y, z)) < abs(*min_element(losses.begin(), losses.end())))) {
                save_weights(w);
            }
            losses.push_back(loss(y, z));
        }
        return losses;
    }

    double loss(vector<int> y, vector<vector<double>> z) {
        double loss = 0;
        for (int i = 0; i < y.size(); i++) {
            loss += (y[i] * (log(z[i][0])) + (1 - y[i]) * log(1 - z[i][0]) + pow(w[i], 2));
        }
        return loss;
    }

    void save_weights(std::vector<double> weights) {
        // проверяем, существует ли файл weights.txt
        ofstream outfile("weights.txt", ios::out | ios::trunc);
        bool file_exists = outfile.good();

        // если файл не существует, создаем его
        if (!file_exists) {
            std::ofstream outfile("weights.txt");
            outfile.close();
        }

        // открываем файл weights.txt для записи
        std::ofstream file("weights.txt", ios_base::app);

        // записываем значения в файл
        for (double w: weights) {
            file << w << std::endl;
        }
        file.close();
    }

    vector<vector<double>> predict_proba(vector<vector<double>> feauters) {
        vector<vector<double>> _f = std::move(feauters);
        vector<double> w;
        for (int i = 0; i < _f.size(); i++) {
            _f[i].insert(_f[i].begin(), 1);
        }
        fstream fin;
        fin.open("weights.txt", ios::in);
        string line;
        while (getline(fin, line)) {
            stringstream lineStream(line);
            string cell;
            getline(lineStream, cell);
            w.push_back(stod(cell));
        }
        fin.close();
        return sigmoid(logit(_f, w));
    }

    bool predict(vector<vector<double>>feauters, double threshold=0.5) {
        return (predict_proba(X)[0][0] >= threshold);
    }

};

int main() {
    DiabetesData a;
    vector<vector<double>> X = a.X;
    vector<int> y = a.y;
    DiabetesData b(8,183,64,0,0,23.3,0.672,32);

    LogisticRegression lg(X, y);
    vector<double> losses = lg.fit();
    for (int i = 0; i < losses.size(); i++) {
        std::cout << losses[i] << endl;
    }
    std::cout<<lg.predict(b.features);
}