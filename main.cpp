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
#include <cmath>

using namespace std;

class DiabetesData {
public:
    // Constructor
    explicit DiabetesData(const string &data_name) {
        load_data_from_file(data_name);
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
        dataset = parsedCsv;

        for (int i = 1; i < dataset.size(); i++) {
            vector<double> cell;
            for (int j = 0; j < dataset[1].size(); j++) {
                if (j != dataset[1].size() - 1) {
                    cell.push_back(stod(dataset[i][j]));
                } else if (j == dataset[1].size() - 1) {
                    y.push_back(stoi(dataset[i][j]));
                }
            }
            X.push_back(cell);
        }
        X = data_normalization(X);
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
            for (auto &i: X)
                i[j] = (i[j] - avarage[j]) / deviation[j];

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
        for (int i = 0; i < X[0].size() + 1; i++) {
            double a = (rand() % 1000);
            w.push_back(a / 1000);
        }

    }

    static vector<vector<double>> logit(vector<vector<double>> X, vector<double> w) {
        unsigned long long int m = X.size();
        unsigned long long int n = X[0].size();
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

    static vector<vector<double>> sigmoid(vector<vector<double>> logits) {
        vector<vector<double>> sigmoids;
        for (int i = 0; i < logits.size(); i++) {
            vector<double> sigmoid;
            sigmoid.reserve(logits[0].size());
            for (int j = 0; j < logits[0].size(); j++)
                sigmoid.push_back(1 / (1 + exp(-logits[i][j])));

            sigmoids.push_back(sigmoid);
        }
        return sigmoids;
    }

    vector<double> fit(int max_iter = 10, double lr = 0.3) {
        vector<vector<double>> X_train = X;

        for (auto &i: X_train)
            i.insert(i.begin(), 1);


        vector<vector<double>> X_trainT(X_train[0].size(), vector<double>(X_train.size(), 0.0));
        for (int i = 0; i < X_train.size(); i++) {
            for (int j = 0; j < X_train[0].size(); j++) {
                X_trainT[j][i] = X_train[i][j];
            }
        }

        for (int iter = 0; iter <= max_iter; iter++) {
            vector<vector<double>> z;
            for (int i = 0; i < sigmoid(logit(X_train, w)).size(); i++) {
                z.push_back(sigmoid(logit(X_train, w))[i]);
            }
            unsigned long long int m = X_trainT.size();
            unsigned long long int n = X_trainT[0].size();
            unsigned long long int p = z[0].size();

            vector<vector<double>> grad(m, vector<double>(p, 0.0));
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < p; j++) {
                    for (int k = 0; k < n; k++) {
                        grad[i][j] += ((X_trainT[i][k] * (z[k][0] - y[k])) + 2*w[i]);
                    }
                    grad[i][j] /= y.size();
                }
            }

            for (int i = 0; i < w.size(); i++) {
                w[i] -= (grad[i][0] * lr);
            }
            if ((!losses.empty()) && (loss(y, z) < *min_element(losses.begin(), losses.end()))) {
                save_weights(w);
            }
            losses.push_back(loss(y, z));
        }
        return losses;
    }

    double loss(vector<int> y, vector<vector<double>> z) {
        double loss = 0;
        for (int i = 0; i < y.size(); i++) {
            loss += (y[i] * (log(z[i][0])) + (1 - y[i]) * log(1 - z[i][0]));
        }
        for (double i: w)
            loss += pow(i, 2);

        loss /= y.size();
        loss *= -1;
        return loss;
    }

    static void save_weights(const std::vector<double> &weights) {
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

    static vector<vector<double>> predict_proba(vector<vector<double>> feauters) {
        vector<vector<double>> _f = std::move(feauters);
        vector<double> w;
        for (auto &i: _f)
            i.insert(i.begin(), 1);

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

    static vector<int> predict(const vector<vector<double>> &feauters, double threshold = 0.5) {
        vector<int> results;
        for (int i = 0; i < predict_proba(feauters).size(); i++) {
            results.push_back(predict_proba(feauters)[i][0] >= threshold);
        }
        return results;
    }

    static int model_accuracy(vector<int> results, vector<int> y) {
        double accuracy = 0;
        double YES = 0;
        double NO = 0;
        for (int i = 0; i < results.size(); i++) {
            if (results[i] == y[i]) {
                YES++;
            } else {
                NO++;
            }
        }
        accuracy = YES / (YES + NO);
        return round(accuracy * 100);
    }


};

int main() {
    DiabetesData a("dataset1");
    vector<vector<double>> X1 = a.X;
    vector<int> y1 = a.y;
    LogisticRegression lg1(X1, y1);
    vector<double> losses = lg1.fit();
    for (double losse: losses)
        cout << losse << endl;

    DiabetesData b("dataset2");
    vector<vector<double>> X = b.X;
    vector<int> y = b.y;
    LogisticRegression lg2(X, y);
    vector<int> results = LogisticRegression::predict(X);
    cout << to_string(LogisticRegression::model_accuracy(results, b.y)) + "%";


}