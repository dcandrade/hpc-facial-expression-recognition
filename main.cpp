#include <bits/stdc++.h> 
#include <fstream>
#include <iostream>
#include <string>
#include <sstream> 
#include <math.h> 
#include <time.h>

using namespace std;

const int NUM_FEATURES = 48 * 48 + 1;
const int NUM_TRAIN_OBSERVATIONS = 3995 + 4097 + 1; //emotion 0 + emotion 2
const int NUM_TEST_OBSERVATIONS = 958 + 1054 + 1;
const int NUM_EPOCHS = 100;
const int COST_THRESHOLD = 0.001;
const double LEARNING_RATE = 0.01; // TODO: check value

double **allocMatrix(int rows, int cols){
    double **matrix = (double **)malloc(rows * sizeof(double *)); 
    for (int i=0; i<rows; i++) 
         matrix[i] = (double *)malloc(cols * sizeof(double)); 

    return matrix;
}

void freeMatrix(double **matrix, int rows, int cols){
    for(int i=0; i<rows; i++) 
        free(matrix[i]);
    free(matrix);
}

void parsePixels(string pixels_str, double *pixels){
    stringstream ss(pixels_str);
    double pixel;
    int i=0;
    
    while(ss >> pixel){
        pixels[i] = pixel / 255;
        i++;
    }
}

void addToDataset(double **X, double *y, int index, int emotion, double* pixels, string usage){
    X[index][0] = 1;
    y[index] = emotion;

    for (int i = 1; i < NUM_FEATURES; i++){
        X[index][i] = pixels[i-1]; // pixels between 0 and 1
    }
}

void initWeights(double *weights){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    uniform_real_distribution<> distribution(0.0001, 0.1); // min = 0, max = 1

    for(int i = 0; i < NUM_FEATURES; i++){
        weights[i] = distribution(generator);
    }
}

double sigmoid(double z){
    return 1.0/(1.0+exp(-z));
}

double hipothesys(double *weights, double *observation){ //observation == xi
    double z = 0;

    for (int i = 0; i < NUM_FEATURES; i++){
        z += (weights[i] * observation[i]);
    }

    return sigmoid(z);
}

double cost_function(double **X_train, double *y_train, double *weights){
    double cost = 0;
    double h_xi;

    for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        h_xi = hipothesys(weights, X_train[i]);
        cost += ((-y_train[i]*log(h_xi)) - ((1-y_train[i])*log(1-h_xi)));
    }

    return  -cost;
}

double gradient(double **X_train, double *y_train, double *weights, int j){
    double h_xi = 0;
    double *xi;
    double sum = 0;
    
    for(int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        xi = X_train[i];
        h_xi = hipothesys(weights, xi);
        sum += (h_xi - y_train[i])*xi[j];
    }
    return (LEARNING_RATE/NUM_TRAIN_OBSERVATIONS) * sum;
}

void updateWeights(double **X_train, double *y_train, double *weights, double *newWeights){
    
    for(int j = 0; j < NUM_FEATURES; j++){
	    newWeights[j] = weights[j] - gradient(X_train, y_train, weights, j);
    }

    for(int j = 0; j < NUM_FEATURES; j++){
        weights[j] = newWeights[j];
    }
}

void saveEpoch(int epoch, ofstream &outputFile, int *predictions, double *y, int size){
    double accuracy, precision, recall, f1;
    int tp = 0, tn = 0, fp = 0, fn = 0;
    
    for(int i = 0; i < size; i++){
        if(predictions[i] == 0 && y[i] == 0)
                tn++;
        else if(predictions[i] == 0 && y[i] == 1)
                fn++;
        else if(predictions[i] == 1 && y[i] == 0)
                fp++;
        else
                tp++;
    }
    
    accuracy = (tp + tn)/(tp + fp + fn + tn);
    precision = tp/(tp + fp);
    recall = tp/(tp + fn);
    f1 = (2*recall*precision)/(recall + precision);
    
    outputFile << epoch << ',' << accuracy << ',' << precision << ',' << recall << ',' << f1 << ',' << 1-accuracy << endl;
}

int main(){
    double **X_train, **X_test, *weights, *newWeights;
    double *y_train, *y_test;
    X_train = allocMatrix(NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    X_test = allocMatrix(NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    y_train = (double *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(double));
    y_test = (double *)malloc(NUM_TEST_OBSERVATIONS * sizeof(double));

    weights = (double *) malloc (NUM_FEATURES * sizeof(double));
    newWeights = (double *) malloc (NUM_FEATURES * sizeof(double));
    initWeights(weights);

    ifstream inputFile;
    ofstream outputFile;
    inputFile.open("data/images.csv");
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    outputFile.open("results/output" + seed);
    outputFile << "#epoch accuracy precision recall f1 cost" << endl;
    string line;
    
    int index_train = 0, index_test = 0;
    double pixels[48*48];
    getline(inputFile, line); // skip first line
    
    // Reading input
    while(getline(inputFile, line)){
        int lastPixelIndex = line.find(",", 2);
        int emotion = line[0] - '0';
        string pixels_str = line.substr(2, line.find(",", 2)-2);
        string usage = line.substr(lastPixelIndex+1);
        parsePixels(pixels_str, pixels);

        if(emotion == 0 || emotion == 2){
                emotion = round(emotion / 2);
            if(usage.compare("Training") != 0){
                addToDataset(X_test, y_test, index_test, emotion, pixels, usage);
                index_test ++;
            }else{
                addToDataset(X_train, y_train, index_train, emotion, pixels, usage);
                index_train ++;
            }
        }
    }

    inputFile.close();
    double cost = INT_MAX;
    double newCost = cost_function(X_train, y_train, weights);
    int epoch = 0;
    
    while(epoch < NUM_EPOCHS){
        int predictions[NUM_TRAIN_OBSERVATIONS], pred;
        int correct = 0;
        int h;

        for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
            h = hipothesys(weights, X_train[i]);
            pred = round(h);
            predictions[i] = pred;
            if(pred == y_train[i]){
                correct ++;
            }
        }

        cout<<"Correct: "<<correct<<", Wrong: "<<NUM_TRAIN_OBSERVATIONS-correct << ", Accuracy: " <<((float)correct/NUM_TRAIN_OBSERVATIONS)*100<<"%, Cost: "<<cost<<endl;
        cout << "Processando época: " << epoch << endl;
        
        saveEpoch(epoch, outputFile, predictions, y_train, NUM_TRAIN_OBSERVATIONS);

        updateWeights(X_train, y_train, weights, newWeights);
        cost = newCost;
        newCost = cost_function(X_train, y_train, weights);
        epoch ++;
    }

    // Run tests
    int predictions[NUM_TEST_OBSERVATIONS], pred;
    int correct = 0;
    int h;
    for (int i = 0; i < NUM_TEST_OBSERVATIONS; i++){
        h = hipothesys(weights, X_test[i]);
        pred = round(h);
        predictions[i] = pred;
        if(pred == y_test[i]){
            correct ++;
        }
    }

    saveEpoch(-1, outputFile, predictions, y_test, NUM_TEST_OBSERVATIONS);


    outputFile.close();

    freeMatrix(X_train, NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    freeMatrix(X_test, NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    free(y_train);
    free(y_test);
    free(weights);
    free(newWeights);
}
