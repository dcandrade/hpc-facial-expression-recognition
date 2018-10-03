#include <bits/stdc++.h> 
#include <fstream>
#include <iostream>
#include <string>
#include <sstream> 

using namespace std;

const int NUM_FEATURES = 48 * 48 + 1;
const int NUM_TRAIN_OBSERVATIONS = 28709 + 1;
const int NUM_TEST_OBSERVATIONS = 3589*2 + 1;
const float LEARNING_RATE = 0.01; // TODO: check value

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

void parsePixels(string pixels_str, int *pixels){
    stringstream ss(pixels_str);
    int pixel, i=0;
    
    while(ss >> pixel){
        pixels[i] = pixel;
        i++;
    }
}

void initWeights(float *weights){
    default_random_engine generator;
    normal_distribution<float> distribution(0.0, 1.0); // mean = 0, stdev = 1

    for(int i = 0; i < NUM_FEATURES; i++){
        weights[i] = distribution(generator);
    }
}

float sigmoid(float z){
    return 1/(1+exp(-z));
}

float hipothesys(float weights[NUM_FEATURES], float observation[NUM_FEATURES]){ //observation == xi
    float sumOfWeightedFeatures = 0;

    for (int i = 0; i < NUM_FEATURES; i++){
        int weight = weights[i];
        int feature = observation[i];

        sumOfWeightedFeatures += (weight * feature);
    }

    return sigmoid(sumOfWeightedFeatures);
}

void addToDataset(float **X, int *y, int index, int emotion, int* pixels, string usage){
    X[index][0] = 1;
    y[index] = emotion;

    for (int i = 1; i <= NUM_FEATURES; i++){
        X[index][i] = pixels[i-1] / 256; // pixels between 0 and 1
    }
}

float *updateWeights(float **X_train, float *y_train, float *weights, float *newWeights){
    
    for(int i = 0; i <= NUM_FEATURES; i++){
	newWeights[i] = weights[i] - LEARNING_RATE*gradient(X_train, y_train, weights, i);
    }

    float *aux;
    aux = weights;
    weigths = newWeights;
    newWeigths = aux;
}

float gradient(float **X_train, float *y_train, float *weights, int j){
    float h_xi = 0;
    float *xi;
    float sum = 0;
    
    for(int i = 0, i < NUM_FEATURES, i++){
        xi = X_train[i]
        h_xi = hipothesys(weights, xi);
        sum += (h_xi - y_train[i])*X_train[i][j];
    }
    return sum;
}

int main(){
    float **X_train, **X_test, *weights, *newWeights;
    int *y_train, *y_test;
    X_train = allocMatrix(NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    X_test = allocMatrix(NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    y_train = (int *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(int));
    y_test = (int *)malloc(NUM_TEST_OBSERVATIONS * sizeof(int));

    weights = (float *) malloc (NUM_FEATURES * sizeof(float));
    initWeights(weights);

    newWeights = (float *) malloc (NUM_FEATURES * sizeof(float));

    ifstream inputFile;
    inputFile.open("data/images.csv");
    string line;
    int index_train = 0, index_test = 0;
    int pixels[48*48];
    int a;
    getline(inputFile, line); // skip first line
    
    // Reading input
    while(getline(inputFile, line)){
        int lastPixelIndex = line.find(",", 2);
        int emotion = line[0] - '0';
        string pixels_str = line.substr(2, line.find(",", 2)-2);
        string usage = line.substr(lastPixelIndex+1);
        parsePixels(pixels_str, pixels);

        if(usage.compare("Training") != 0){
            addToDataset(X_test, y_test, index_test, emotion, pixels, usage);
            index_test ++;
        }else{
            addToDataset(X_train, y_train, index_train, emotion, pixels, usage);
            index_train ++;
        }
    }

    inputFile.close();

    updateWeights(X_train, y_train, weights, newWeights);


    freeMatrix(X_train, NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);
    freeMatrix(X_test, NUM_TEST_OBSERVATIONS, NUM_FEATURES);
    free(y_train);
    free(y_test);
}
