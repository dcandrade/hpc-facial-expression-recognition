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
#include <mpi.h>


using namespace std;

int NUM_FEATURES = 128 * 128 + 1; /// Quantidade de pixels + bias
int NUM_TRAIN_OBSERVATIONS = 15640; /// Quantidade de observações de treino
int NUM_TEST_OBSERVATIONS = 3730; /// Quantidade de observações de teste
int NUM_EPOCHS = 500; /// Quantidade de épocas
int NUM_ITERATIONS = 5;
float LEARNING_RATE = 0.001;	 /// Taxa de Apredizado
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
 * @param sex Emoção normalizada (0 ou 1)
 * @param pixels Vetor de 48*48 posições contendo os valores normalizado de cada pixel
 */
void addToDataset(float **X, float *y, int index, int sex, float* pixels){
    X[index][0] = 1;
    y[index] = sex;

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
float hypothesis(float *weights, float *observation, int rank, int num_procs){ //observation == xi
    float z = 0;
    int workload = NUM_FEATURES/num_procs;
    int rank_workload = workload * rank;

    //#pragma omp parallel for reduction(+:z)
    for (int i = rank_workload; i < rank_workload+workload; i++){
        z += (weights[i] * observation[i]);		
    }

    float global_z = 0;

    MPI_Allreduce(&z, &global_z, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    return sigmoid(global_z);													
}

//Método usado para calcular a função de custo.
float cost_function(float **X_train, float *y_train, float *predictions, int rank, int num_procs){
    float cost = 0;
    float h_xi;

    int workload = NUM_TRAIN_OBSERVATIONS/num_procs;
    int rank_workload = workload * rank;

    for (int i = rank_workload; i < rank_workload+workload; i++){
        h_xi = predictions[i];
        float p1 = y_train[i] * log(h_xi);								
        float p2 = (1-y_train[i]) * log(1-h_xi);						
        cost += (-p1-p2);											
    }

    float global_cost = 0;
    MPI_Allreduce(&cost, &global_cost, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    return  cost/NUM_TRAIN_OBSERVATIONS;
}
//Método para calcular o gradiente. É passado como parâmetro todas as observações, as respostas e as predições.
float gradient(float **X_train, float *y_train, float *predictions, int j, int rank, int num_procs){

    float h_xi = 0;
    float *xi;
    float sum = 0;

     int workload = NUM_TRAIN_OBSERVATIONS/num_procs;
    int rank_workload = workload * rank;

    
    for(int i = rank_workload; i < rank_workload+workload; i++){
        xi = X_train[i];										
        h_xi = predictions[i];									
        sum += (h_xi - y_train[i])*xi[j];						
    }

    float global_sum = 0;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

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
void updateWeights(float **X_train, float *y_train, float *weights, float *predictions, int rank, int num_procs){ 

    for(int j = 0; j < NUM_FEATURES; j++){
	    weights[j] -= gradient(X_train, y_train, predictions, j, rank, num_procs);
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

    outputFile << epoch <<  "," << accuracy <<  "," << precision << "," << recall <<  "," << f1 <<  "," << cost << endl;
}

/**
 * @brief Carrega os argumentos passados via linha de comando nas respectivas variáveis
 * 
 * @param argc Quantidade de argumentos
 * @param argv Vetor contendo o valor dos argumentos
 */
void parse_args(int argc, char **argv){
    if(argc < 5){
        cout << "Utilização: <executavel> QTD_TREINO QTD_TESTE NUM_EPOCAS TAXA_APRENDIZADO PREFIXO_ARQUIVO_SAIDA"<<endl;
        exit(EXIT_FAILURE);
    }

    NUM_TRAIN_OBSERVATIONS = atof(argv[1]);
    NUM_TEST_OBSERVATIONS = atof(argv[2]);
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
    
    ifstream trainFile, testFile;
    trainFile.open("../data/train.csv");
    testFile.open("../data/test.csv");

    if(!trainFile.is_open()){
        cout <<" O arquivo de entrada data/train.csv não foi encontrado"<<endl;
        exit(EXIT_FAILURE);
    }

    if(!testFile.is_open()){
        cout <<" O arquivo de entrada data/test.csv não foi encontrado"<<endl;
        exit(EXIT_FAILURE);
    }

    string line;
    int index = 0, sex, pixelsIndex;
    float pixels[NUM_FEATURES - 1];
	//cout << "carregando treino" << endl;
    while(getline(trainFile, line)){    // Leitura do arquivo de entrada (treino)
        sex = line[0] - '0';
        pixelsIndex = line.find(",");
        string pixels_str = line.substr(pixelsIndex+1);
        parsePixels(pixels_str, pixels);

        if (index < NUM_TRAIN_OBSERVATIONS){
            addToDataset(X_train, y_train, index, sex, pixels);
            index ++;
        }
    }
    //cout << "carregando teste" << endl;
    index = 0;
    while(getline(testFile, line)){    // Leitura do arquivo de entrada (teste)
        sex = line[0] - '0';
        
        pixelsIndex = line.find(",");
        string pixels_str = line.substr(pixelsIndex+1);
        parsePixels(pixels_str, pixels);

        if (index < NUM_TEST_OBSERVATIONS){
            addToDataset(X_test, y_test, index, sex, pixels);
            index ++;
        }
    }

    trainFile.close();
    testFile.close();
    float predictions[NUM_TRAIN_OBSERVATIONS], cost;

   for(int iteration = 0; iteration < NUM_ITERATIONS; iteration++){
        int rank, num_procs;

        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&num_procs);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        ofstream outputFile;
        string outputFileName = OUTPUT_FILE_PREFIX+"output_" +to_string(NUM_EPOCHS) + "epochs_" + to_string(NUM_TRAIN_OBSERVATIONS) + "train_" + to_string(NUM_TEST_OBSERVATIONS) + "test_@"+to_string(seed)+".txt"; /// Construção do título do arquivo de saída com as estatísticas de treino
        outputFile.open(outputFileName);
        outputFile << "epoch,accuracy,precision,recall,f1,cost" << endl; // Escreve o cabeçalho dos dados no arquivo de saída

        
        auto start = chrono::system_clock::now(); // Início da contagem do tempo de cômputo

        // Execução das épocas de treinamento
        int epoch = 1;
      
        while(epoch <= NUM_EPOCHS){
            for (int i = 0; i < NUM_TRAIN_OBSERVATIONS; i++){
                predictions[i] = hypothesis(weights, X_train[i], rank, num_procs);
            }

            updateWeights(X_train, y_train, weights, predictions); 
            cost = cost_function(X_train, y_train, predictions); 
            
            saveEpoch(epoch, outputFile, predictions, y_train, NUM_TRAIN_OBSERVATIONS, cost);
            epoch ++;
        }

        // Cálculo das predicçoes do conjunto de teste pois o treinamento já foi finalizado.    
        for (int i = 0; i < NUM_TEST_OBSERVATIONS; i++){
            predictions[i] = hypothesis(weights, X_test[i], rank, num_procs);
        }

        saveEpoch(-1, outputFile, predictions, y_test, NUM_TEST_OBSERVATIONS, -1); // Salva as estatísticas de teste no arquivo de saída

        auto end = chrono::system_clock::now(); // Fim da contagem do tempo de cômputo
        long elapsed = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // Escrita do tempo de cômputo no arquivo de saída
        ofstream timeFile;
        timeFile.open(OUTPUT_FILE_PREFIX+"time.txt", ios_base::app); // Concatena o tempo medido no final do arquivo
        timeFile << elapsed <<endl;
        timeFile.close();
        outputFile.close();
   }

    // Fecha os arquivos e libera as estruturas dinamicamente alocadas
    freeMatrix(X_train, NUM_TRAIN_OBSERVATIONS);
    freeMatrix(X_test, NUM_TEST_OBSERVATIONS);
    free(y_train);
    free(y_test);
    free(weights);
    free(newWeights);
}
