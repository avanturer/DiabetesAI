//
// Created by hedge on 09.05.2023.
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

class LogisticRegression {
public:
    vector<vector<double>> X_;
    vector<int> y_;
    vector<double> w_;
    vector<double> losses_;
    vector<vector<double>> logloss_;
    int max_iter_ = 0;
    double lr_ = 0;


    LogisticRegression(const vector<vector<double>> &X, const vector<int> &y) {
        X_ = X;
        y_ = y;
        for (int i = 0; i < X[0].size() + 1; i++) {
            double a = rand() % 1000;
            w_.push_back(a / 1000);
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

    vector<double> fit(int max_iter = 100, double lr = 0.1) {
        if (max_iter <= 0 || lr <= 0) {
            throw runtime_error("maximum iterations or learning rate < 0");
        }
        max_iter_ = max_iter;
        lr_ = lr;
        vector<vector<double>> X_train = X_;
        for (auto &i: X_train)
            i.insert(i.begin(), 1);


        vector<vector<double>> X_trainT(X_train[0].size(), vector<double>(X_train.size(), 0.0));
        for (int i = 0; i < X_train.size(); i++) {
            for (int j = 0; j < X_train[0].size(); j++) {
                X_trainT[j][i] = X_train[i][j];
            }
        }
        for (int iter = 0; iter <= max_iter_; iter++) {
            vector<vector<double>> z;

            for (int i = 0; i < sigmoid(logit(X_train, w_)).size(); i++) {
                z.push_back(sigmoid(logit(X_train, w_))[i]);
            }
            unsigned long long int m = X_trainT.size();
            unsigned long long int n = X_trainT[0].size();
            unsigned long long int p = z[0].size();

            vector<vector<double>> grad(m, vector<double>(p, 0.0));
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < p; j++) {
                    for (int k = 0; k < n; k++) {
                        grad[i][j] += ((X_trainT[i][k] * (z[k][0] - y_[k])) + 2 * w_[i]);
                    }
                    grad[i][j] /= y_.size();
                }
            }

            for (int i = 0; i < w_.size(); i++) {
                w_[i] -= (grad[i][0] * lr_);
            }
            losses_.push_back(loss(y_, z));
            if ((!losses_.empty()) && (loss(y_, z) < *min_element(losses_.begin(), losses_.end()))) {
                save_weights(w_);
            }
        }
        saveLossToCSV(losses_);
        return losses_;
    }

    double loss(vector<int> y, vector<vector<double>> z) {
        double loss = 0;
        for (int i = 0; i < y.size(); i++) {
            loss += (y[i] * (log(z[i][0])) + (1 - y[i]) * log(1 - z[i][0]));
        }
        for (double i: w_)
            loss += pow(i, 2);

        loss /= y.size();
        loss *= -1;
        return loss;
    }

    static void save_weights(const vector<double> &weights) {
        ofstream outfile("weights.txt", ios::out | ios::trunc);
        bool file_exists = outfile.good();

        if (!file_exists) {
            ofstream outfile("weights.txt");
            outfile.close();
        }

        ofstream file("weights.txt", ios_base::app);

        for (double w: weights) {
            file << w << endl;
        }
        file.close();
    }

    static vector<vector<double>> predict_proba(vector<vector<double>> feauters) {
        vector<vector<double>> _f = move(feauters);
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

    static void saveLossToCSV(const vector<double> &losses) {
        const string filename = "losses_.csv";
        ofstream outputFile(filename);
        if (outputFile.is_open()) {
            outputFile << 1 << "," << 0 << endl;
            for (int i = 0; i < losses.size(); i++) {
                outputFile << losses[i] << "," << i + 1 << endl;
            }
            outputFile.close();
            cout << "Data saved to losses_.csv successfully." << endl;
        } else {
            cerr << "Unable to open file: " << filename << endl;
        }
    }

};

#endif