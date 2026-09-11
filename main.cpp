#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
using namespace std; 

//Data Loading 
struct DataPoint {
    vector<double> features;
    double label;
}; 

vector<DataPoint> loadCSV(const string& path) {
    vector<DataPoint> data; 
    ifstream file(path); 

    if(!file.is_open()) {
        cerr << "could not open file: " << path << endl;
        return data;
    }

    string line; 

    getline(file, line); // skip header

    while (getline(file, line)){
        if(line.empty()) 
            continue;

        if(!line.empty() && line.back() == '\r') {
            line.pop_back(); // remove carriage return if present
        }

        stringstream ss(line);
        string cell;
        vector<double> values; 

        while (getline(ss, cell, ',')) {
            values.push_back(stod(cell)); // convert string to double and add to values
        }

        DataPoint dp;
        dp.label = values.back(); // last value is the label
        values.pop_back(); // remove label from features
        dp.features = values; // remaining values are features

        data.push_back(dp); // add DataPoint to data vector
    }
    return data; 
}

double euclideanDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double classifyKNN(const DataPoint& query, const vector<DataPoint>& trainingData, int k) {
    vector<pair<double, double>> distances; // pair of distance and label
    for(const auto& trainPoint : trainingData) {
        double dist = euclideanDistance(query.features, trainPoint.features);
        distances.push_back({dist, trainPoint.label});
    }

    sort (distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first < b.first; // sort by distance
    });

    map<double, int> votes;
    for(int i = 0; i < k; i++) {
        double neighborLabel = distances[i].second; 
        votes[neighborLabel]++;
    }

    double bestLabel = -1;
    int bestCount = -1;
    for(const auto& v : votes) {
        if(v.second > bestCount) {
            bestCount = v.second;
            bestLabel = v.first;
        }
    }
    return bestLabel; 
}

double evaluateKNN (const vector<DataPoint>& testData, const vector<DataPoint>& trainingData, int k) {
    int correct = 0; 

    for(const auto& testPoint : testData) {
        double predictedLabel = classifyKNN(testPoint, trainingData, k);
        if(predictedLabel == testPoint.label) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / testData.size();
    cout << "KNN k=" 
         << k 
         << " accuracy: " 
         << accuracy * 100 
         << "% "<< correct 
         << "/" 
         << testData.size() 
         << " Correct, " 
         << "Error: " 
         << (1 - accuracy) * 100 
         << "%" 
         << endl;

    return accuracy;
}

struct Perceptron{
    vector<double> weights;
    double bias;
    double learningRate;

    Perceptron(int numFeatures, double lr) {
        weights = vector<double>(numFeatures, 0.0);
        bias = 0.0;
        learningRate = lr;
    }

    int predict(const vector<double>& x){
        double sum = bias; 
        for (size_t i = 0; i < x.size(); i++) {
            sum += weights[i] * x[i];
        }
        return (sum >= 0) ? 1 : 0;
    }

    vector<int> train (vector<DataPoint>& trainingData, int epochs) {
        vector<int> errorsPerEpoch; 

        for(int epoch = 0; epoch < epochs; epoch++) {
            int errors = 0; 

            for(const auto& dataPoint : trainingData) {
                int prediction = predict(dataPoint.features);
                int actual = static_cast<int>(dataPoint.label);
                int error = actual - prediction;

                if(error != 0) {
                    errors++;
                    for(size_t i = 0; i < weights.size(); i++) {
                        weights[i] += learningRate * error * dataPoint.features[i];
                    }
                    bias += learningRate * error;
                }
            }
            errorsPerEpoch.push_back(errors);
            cout << "Epoch " << epoch + 1 << ": " << errors << " misclassifications" << endl;
        }
        return errorsPerEpoch;
    }
};


int main() {

    //KNN - MNIST Dataset

    auto mnistTrain = loadCSV("Data_uppgift1/mnist_train.csv");
    auto mnistTest = loadCSV("Data_uppgift1/mnist_test.csv");

    evaluateKNN(mnistTest, mnistTrain, 1);
    evaluateKNN(mnistTest, mnistTrain, 3);
    evaluateKNN(mnistTest, mnistTrain, 5);
    evaluateKNN(mnistTest, mnistTrain, 10);

    //Perceptron - OR Gate

    auto orTrain = loadCSV("Data_uppgift1/OR_train.csv");
    auto orTest = loadCSV("Data_uppgift1/OR_test.csv");

    Perceptron p(2, 0.1);

    p.train(orTrain, 10);

    int correct = 0;

    for (const auto& dataPoint : orTest) {
        int prediction = p.predict(dataPoint.features);

        if (prediction == static_cast<int>(dataPoint.label)) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / orTest.size();

    cout << "Perceptron test accuracy: "
         << accuracy * 100
         << "% "
         << correct
         << "/"
         << orTest.size()
         << " Correct, "
         << "Error: "
         << (1 - accuracy) * 100
         << "%"
         << endl;

    return 0;
}