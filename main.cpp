/**
 * @file main.cpp
 * @author Daniel Andrade e Gabriel Gomes
 * @brief Reconhecedor de Expressões Faciais através de Regressão Logística
 * @version 1.0
 * @date 2018-10-12
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#include <bits/stdc++.h> 
#include <fstream>
#include <iostream>
#include <string>
#include <sstream> 
#include <math.h> 
#include <time.h>

using namespace std;

int NUM_FEATURES = 48 * 48 + 1; /// Quantidade de pixels + bias
int NUM_TRAIN_OBSERVATIONS = 3995 + 4097; /// Quantidade de observações de treino (emoção 0 + emoção 2)
int NUM_TEST_OBSERVATIONS = 958 + 1054; /// Quantidade de observações de teste (emoção 0 + emoção 2)
int NUM_EPOCHS = 100; /// Quantidade de épocas
float LEARNING_RATE = 0.01;	 /// Taxa de Apredizado
string OUTPUT_FILE_PREFIX = "results/"; /// Prefixo do arquivo de saída

/**
 * @brief Aloca uma dinamicamente uma matriz de floats e retorna o ponteiro para acessá-la
 * 
 * @param rows Quantidade de linhas 
 * @param cols Quantidade de colunas
 * @return float** 
 */
float **allocMatrix(int rows, int cols){
    float **matrix = (float **)malloc(rows * sizeof(float *)); 
    for (int i=0; i<rows; i++) 
         matrix[i] = (float *)malloc(cols * sizeof(float)); 

    return matrix;
}

/**
 * @brief Libera uma matriz dinamicamente alocada
 * 
 * @param matrix Ponteiro para a matriz a ser liberada
 * @param rows Quantidade de linhas da matriz
 */
void freeMatrix(float **matrix, int rows){
    for(int i=0; i<rows; i++) 
        free(matrix[i]);
    free(matrix);
}

/**
 * @brief Popula e normaliza um vetor de floats com os pixels presentes numa string
 * 
 * @param pixels_str String com os pixels separados por espaço
 * @param pixels Vetor de 48*48 espaços nos quais os pixels serão armazenados
 */
void parsePixels(string pixels_str, float *pixels){
    stringstream ss(pixels_str);
    float pixel;
    int i=0;
    
    while(ss >> pixel){
        pixels[i] = pixel / 255;
        i++;
    }
}

/**
 * @brief 
 * 
 * @param X Matriz de características
 * @param y Vetor de marcações (gabaritos)
 * @param index Índice na observação/marcação atual
 * @param emotion Emoção normalizada (0 ou 1)
 * @param pixels Vetor de 48*48 posições contendo os valores normalizado de cada pixel
 */
void addToDataset(float **X, float *y, int index, int emotion, float* pixels){
    X[index][0] = 1;
    y[index] = emotion;

    for (int i = 1; i < NUM_FEATURES; i++){
        X[index][i] = pixels[i-1];
    }
}

/**
 * @brief Calcula função logística ou função sigmóide
 * 
 * @param z Parâmetro
 * @return float Valor da função para z
 */
float sigmoid(float z){
    return 1.0/(1.0+exp(-z));
}

/**
 * @brief Calcula o ŷ para uma determinada observação dado um conjunto de pesos
 * 
 * @param weights Pesos a serem aplicados na hipótese
 * @param observation Observação para qual o ŷ será calculado
 * @return float ŷ  
 */
float hypothesis(float *weights, float *observation){ //observation == xi
    float z = 0;

    for (int i = 0; i < NUM_FEATURES; i++){
        z += (weights[i] * observation[i]);								/// Este produto representa a hipótese
    }

    return sigmoid(z);													///A hipótese é passada para função sigmóide antes de retornar, obtendo ŷ
}
//Método usado para calcular a função de custo.
float cost_function(float **X_train, float *y_train, float *predictions){
    float cost = 0;
    float h_xi;

    for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        h_xi = predictions[i];
        float p1 = y_train[i] * log(h_xi);								///primeira parte da função de custo
        float p2 = (1-y_train[i]) * log(1-h_xi);						///segunda parte da função de custo
        cost += (-p1-p2);												///função de custo dada pelo somatório do inverso das duas partes
    }

    return  cost/NUM_TRAIN_OBSERVATIONS;
}
//Método para calcular o gradiente. É passado como parâmetro todas as observações, as respostas e as predições.
float gradient(float **X_train, float *y_train, float *predictions, int j){
    float h_xi = 0;
    float *xi;
    float sum = 0;
    
    for(int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
        xi = X_train[i];										///pega uma das observações
        h_xi = predictions[i];									///pega uma das predições (resposta prevista)
        sum += (h_xi - y_train[i])*xi[j];						///faz o somatório da diferença do vetor de resposta prevista pelo vetor de resposta multiplicado pelo vetor contendo os valores de observação
    }
    return (LEARNING_RATE/NUM_TRAIN_OBSERVATIONS) * sum;
}

/**
 * @brief Recalcula e atualiza os coeficientes de regressão (pesos)
 * 
 * @param X_train Conjunto de observações
 * @param y_train Marcações das observações
 * @param weights Pesos atuais
 * @param predictions Predicões de cada item de X_train utilizando os coeficientes mais recentes
 */
void updateWeights(float **X_train, float *y_train, float *weights, float *predictions){    
    for(int j = 0; j < NUM_FEATURES; j++){
	    weights[j] -= gradient(X_train, y_train, predictions, j);
    }
}

/**
 * @brief Salva as informações de uma época no arquivo de saída
 * 
 * @param epoch Número da época atual
 * @param outputFile Arquivo de saída
 * @param predictions Predições atuais
 * @param y Marcação das observações
 * @param size Tamanho do vetor y
 * @param cost Custo do do erro da época atual
 */
void saveEpoch(int epoch, ofstream &outputFile, float *predictions, float *y, int size, float cost){
    float accuracy, precision, recall, f1;
    float tp = 0, tn = 0, fp = 0, fn = 0;
    int pred, real;

    for(int i = 0; i < size; i++){										///calculando dados para matriz de confusão a partir do valor real e do valor predito
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

    outputFile << epoch <<  "," << accuracy <<  "," << precision << "," << recall <<  "," << f1 <<  "," << cost << endl;
}

/**
 * @brief Carrega os argumentos passados via linha de comando nas respectivas variáveis
 * 
 * @param argc Quantidade de argumentos
 * @param argv Vetor contendo o valor dos argumentos
 */
void parse_args(int argc, char**argv){
    if(argc < 5){
        cout << "Utilização: <executavel> QTD_TREINO QTD_TESTE NUM_EPOCAS TAXA_APRENDIZADO PREFIXO_ARQUIVO_SAIDA"<<endl;
        exit(EXIT_FAILURE);
    }

    NUM_TRAIN_OBSERVATIONS = atof(argv[1]) + 1;
    NUM_TEST_OBSERVATIONS = atof(argv[2]) + 1;
    NUM_EPOCHS = atof(argv[3]);
    LEARNING_RATE = atof(argv[4]);
    OUTPUT_FILE_PREFIX += argv[5];
    OUTPUT_FILE_PREFIX += "/";
}

int main(int argc, char** argv){
    parse_args(argc, argv);

    float **X_train, **X_test, *weights, *newWeights;						//definindo matrizes
    float *y_train, *y_test;												//definindo matrizes
    X_train = allocMatrix(NUM_TRAIN_OBSERVATIONS, NUM_FEATURES);			//alocando espaço 
    X_test = allocMatrix(NUM_TEST_OBSERVATIONS, NUM_FEATURES);				//alocando espaço 
    y_train = (float *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(float));		//alocando espaço 
    y_test = (float *)malloc(NUM_TEST_OBSERVATIONS * sizeof(float));		//alocando espaço 

    weights = (float *) malloc (NUM_FEATURES * sizeof(float));    			//alocando espaço 

    for(int i = 0; i < NUM_FEATURES; i++){									//iniciando com zero o vetor contendo os coeficientes de regressão
        weights[i] = 0;
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto start = chrono::system_clock::now(); // Início da contagem do tempo de cômputo

    ifstream inputFile;
    ofstream outputFile;
    inputFile.open("data/fer2013.csv");

    if(!inputFile.is_open()){
        cout <<" O arquivo de entrada data/fer2013.csv não foi encontrado"<<endl;
        exit(EXIT_FAILURE);
    }

    string outputFileName = OUTPUT_FILE_PREFIX+"output_" +to_string(NUM_EPOCHS) + "epochs_" + to_string(NUM_TRAIN_OBSERVATIONS) + "train_" + to_string(NUM_TEST_OBSERVATIONS) + "test_@"+to_string(seed)+".txt"; /// Construção do título do arquivo de saída com as estatísticas de treino
    outputFile.open(outputFileName);
    outputFile << "epoch,accuracy,precision,recall,f1,cost" << endl; // Escreve o cabeçalho dos dados no arquivo de saída
    string line;
    
    int index_train = 0, index_test = 0;
    float pixels[48*48];
    getline(inputFile, line); // Ignora primeira linha do arquivo com o cabeçalho
    
    while(getline(inputFile, line)){    // Leitura do arquivo de entrada
        int emotion = line[0] - '0';
        
        if(emotion == 0 || emotion == 2){
                int lastPixelIndex = line.find(",", 2);
                string pixels_str = line.substr(2, line.find(",", 2)-2);
                string usage = line.substr(lastPixelIndex+1);
                parsePixels(pixels_str, pixels);
                emotion = round(emotion / 2); // Normaliza o valor emoção
            if(usage.compare("Training") != 0 && index_test < NUM_TEST_OBSERVATIONS){ // Se for dado de teste, coloca no conjunto de teste.
                addToDataset(X_test, y_test, index_test, emotion, pixels);
                index_test ++;
            }else if (index_train < NUM_TRAIN_OBSERVATIONS){
                addToDataset(X_train, y_train, index_train, emotion, pixels);
                index_train ++;
            }
        }
    }

    inputFile.close();

    // Execução das épocas de treinamento
    int epoch = 1;
    float predictions[NUM_TRAIN_OBSERVATIONS], cost;
    while(epoch <= NUM_EPOCHS){
        
        for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
            predictions[i] = hypothesis(weights, X_train[i]);
        }

        updateWeights(X_train, y_train, weights, predictions); 
        cost = cost_function(X_train, y_train, predictions); 
        
        saveEpoch(epoch, outputFile, predictions, y_train, NUM_TRAIN_OBSERVATIONS, cost);
        epoch ++;
    }

    // Cálculo das predicçoes do conjunto de teste pois o treinamento já foi finalizado.    
    for (int i = 0; i < NUM_TEST_OBSERVATIONS; i++){
        predictions[i] = hypothesis(weights, X_test[i]);;
    }

    saveEpoch(-1, outputFile, predictions, y_test, NUM_TEST_OBSERVATIONS, -1); // Salva as estatísticas de teste no arquivo de saída

    auto end = chrono::system_clock::now(); // Fim da contagem do tempo de cômputo
    long elapsed = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Escrita do tempo de cômputo no arquivo de saída
    ofstream timeFile;
    timeFile.open(OUTPUT_FILE_PREFIX+"time.txt", ios_base::app); // Concatena o tempo medido no final do arquivo
    timeFile << elapsed <<endl;
    timeFile.close();

    // Fecha os arquivos e libera as estruturas dinamicamente alocadas
    outputFile.close();
    freeMatrix(X_train, NUM_TRAIN_OBSERVATIONS);
    freeMatrix(X_test, NUM_TEST_OBSERVATIONS);
    free(y_train);
    free(y_test);
    free(weights);
    free(newWeights);
}
