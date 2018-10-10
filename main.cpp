#include <bits/stdc++.h> 
#include <fstream>
#include <iostream>
#include <string>
#include <sstream> 
#include <math.h> 
#include <time.h>

using namespace std;

int NUM_FEATURES = 48 * 48 + 1;
int NUM_TRAIN_OBSERVATIONS = 3995 + 4097 + 1; //emotion 0 + emotion 2
int NUM_TEST_OBSERVATIONS = 958 + 1054 + 1;
int NUM_EPOCHS = 100;
int COST_THRESHOLD = 0.001;
float LEARNING_RATE = 0.01;

float **allocMatrix(int rows, int cols){
    float **matrix = (float **)malloc(rows * sizeof(float *)); 
    for (int i=0; i<rows; i++) 
         matrix[i] = (float *)malloc(cols * sizeof(float)); 

    return matrix;
}

void freeMatrix(float **matrix, int rows, int cols){
    for(int i=0; i<rows; i++) 
        free(matrix[i]);
    free(matrix);
}

void parsePixels(string pixels_str, float *pixels){
    stringstream ss(pixels_str);
    float pixel;
    int i=0;
    
    while(ss >> pixel){
        pixels[i] = pixel / 255;
        i++;
    }
}

void addToDataset(float **X, float *y, int index, int emotion, float* pixels, string usage){
    X[index][0] = 1;
    y[index] = emotion;

    for (int i = 1; i < NUM_FEATURES; i++){
        X[index][i] = pixels[i-1];
    }
}

float sigmoid(float z){
    return 1.0/(1.0+exp(-z));
}

float hypothesis(float *weights, float *observation){ //observation == xi
    float z = 0;

    for (int i = 0; i < NUM_FEATURES; i++){
        z += (weights[i] * observation[i]);
    }

    return sigmoid(z);
}

float cost_function(float **X_train, float *y_train, float *predictions){
    float cost = 0;
    float h_xi;

    for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        h_xi = predictions[i];
        float p1 = y_train[i] * log(h_xi);
        float p2 = (1-y_train[i]) * log(1-h_xi);
        cost += (-p1-p2);
    }

    return  cost/NUM_TRAIN_OBSERVATIONS;
}

float gradient(float **X_train, float *y_train, float *predictions, int j){
    float h_xi = 0;
    float *xi;
    float sum = 0;
    
    for(int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        xi = X_train[i];
        h_xi = predictions[i];
        sum += (h_xi - y_train[i])*xi[j];
    }
    return (LEARNING_RATE/NUM_TRAIN_OBSERVATIONS) * sum;
}

void updateWeights(float **X_train, float *y_train, float *weights, float *predictions){    
    for(int j = 0; j < NUM_FEATURES; j++){
	    weights[j] -= gradient(X_train, y_train, predictions, j);
    }
}

void saveEpoch(int epoch, ofstream &outputFile, float *predictions, float *y, int size, float cost, long time){
    float accuracy, precision, recall, f1;
    float tp = 0, tn = 0, fp = 0, fn = 0;
    int pred, real;

    for(int i = 0; i < size; i++){
        pred = round(predictions[i]);
        real= round(y[i]);

        if(pred == 0 && real == 0)
                tn++;
        else if(pred == 0 && real == 1)
                fn++;
        else if(pred == 1 && real == 0)
                fp++;
        else
                tp++;
    }
    
    accuracy = (tp + tn)/(tp + fp + fn + tn);
    precision = tp/(tp + fp);
    recall = tp/(tp + fn);
    f1 = (2*recall*precision)/(recall + precision);

    outputFile << epoch <<  "," << accuracy <<  "," << precision << "," << recall <<  "," << f1 <<  "," << cost << "," <<time << endl;
}

void parse_args(int argc, char**argv){
    if(argc < 4){
        return;
    }

    NUM_TRAIN_OBSERVATIONS = atof(argv[0]) + 1;
    NUM_TEST_OBSERVATIONS = atof(argv[1]) + 1;
    NUM_EPOCHS = atof(argv[2]);
    LEARNING_RATE = atof(argv[4]);
}

int main(int argc, char** argv){
    parse_args(argc, argv);

    float **X_train, **X_test, *weights, *newWeights;
    float *y_train, *y_test;
    X_train = allocMatrix(NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    X_test = allocMatrix(NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    y_train = (float *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(float));
    y_test = (float *)malloc(NUM_TEST_OBSERVATIONS * sizeof(float));

    weights = (float *) malloc (NUM_FEATURES * sizeof(float));
    for(int i = 0; i < NUM_FEATURES; i++){
        weights[i] = 0;//distribution(generator);
    }

    ifstream inputFile;
    ofstream outputFile;
    inputFile.open("data/images.csv");
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    string outputFileName = "results/output_" +to_string(NUM_EPOCHS) + "epochs_" + to_string(NUM_TRAIN_OBSERVATIONS) + "train_" + to_string(NUM_TEST_OBSERVATIONS) + "test___"+to_string(seed)+".txt";
    outputFile.open(outputFileName);
    outputFile << "epoch,accuraloadscy,precision,recall,f1,cost,time" << endl;
    string line;
    
    int index_train = 0, index_test = 0;
    float pixels[48*48];
    getline(inputFile, line); // skip first line
    
    // Reading input
    while(getline(inputFile, line)){
        int emotion = line[0] - '0';
        
        if(emotion == 0 || emotion == 2){
                int lastPixelIndex = line.find(",", 2);
                string pixels_str = line.substr(2, line.find(",", 2)-2);
                string usage = line.substr(lastPixelIndex+1);
                parsePixels(pixels_str, pixels);
                emotion = round(emotion / 2);
            if(usage.compare("Training") != 0 && index_test < NUM_TEST_OBSERVATIONS){
                addToDataset(X_test, y_test, index_test, emotion, pixels, usage);
                index_test ++;
            }else if (index_train < NUM_TRAIN_OBSERVATIONS){
                addToDataset(X_train, y_train, index_train, emotion, pixels, usage);
                index_train ++;
            }
        }
    }

    inputFile.close();
    
    int epoch = 1;
    while(epoch <= NUM_EPOCHS){
        float predictions[NUM_TRAIN_OBSERVATIONS], pred;
        auto start = chrono::system_clock::now();

        for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
            pred = hypothesis(weights, X_train[i]);
            predictions[i] = pred;
        }

        updateWeights(X_train, y_train, weights, predictions);

        auto end = chrono::system_clock::now();
        long elapsed = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float cost = cost_function(X_train, y_train, predictions);
        
        saveEpoch(epoch, outputFile, predictions, y_train, NUM_TRAIN_OBSERVATIONS, cost, elapsed);
        epoch ++;
    }

    // Run tests
    float predictions[NUM_TEST_OBSERVATIONS], pred;
    int correct = 0;
    for (int i = 0; i < NUM_TEST_OBSERVATIONS; i++){
        pred = hypothesis(weights, X_test[i]);
        predictions[i] = pred;
        if(pred == y_test[i]){
            correct ++;
        }
    }

    saveEpoch(-1, outputFile, predictions, y_test, NUM_TEST_OBSERVATIONS, -1, -1);


    outputFile.close();
    freeMatrix(X_train, NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    freeMatrix(X_test, NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    free(y_train);
    free(y_test);
    free(weights);
    free(newWeights);
}
