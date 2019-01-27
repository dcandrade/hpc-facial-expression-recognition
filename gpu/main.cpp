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
#include <CL/cl.h>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

using namespace std;

int NUM_FEATURES = 128 * 128 + 1;   /// Quantidade de pixels + bias
int NUM_TRAIN_OBSERVATIONS = 15640; /// Quantidade de observações de treino
int NUM_TEST_OBSERVATIONS = 3730;   /// Quantidade de observações de teste
int NUM_EPOCHS = 500;               /// Quantidade de épocas
int NUM_ITERATIONS = 5;
float LEARNING_RATE = 0.001;            /// Taxa de Apredizado
string OUTPUT_FILE_PREFIX = "results/"; /// Prefixo do arquivo de saída
int LOCAL_WORK_SIZE = 16;
int GLOBAL_WORK_SIZE = 1024;

int BATCH_SIZE = 8000;
int NUM_BATCHES = ceil(((float) NUM_TRAIN_OBSERVATIONS)/BATCH_SIZE);

/**Source: http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl**/
long LoadOpenCLKernel(char const *path, char **buf) {
    FILE *fp;
    size_t fsz;
    long off_end;
    int rc;

    /* Open the file */
    fp = fopen(path, "r");
    if (NULL == fp)
    {
        return -1L;
    }
    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if (0 != rc)
    {
        return -1L;
    }
    /* Byte offset to the end of the file (size) */
    if (0 > (off_end = ftell(fp)))
    {
        return -1L;
    }
    fsz = (size_t)off_end;
    /* Allocate a buffer to hold the whole file */
    *buf = (char *)malloc(fsz + 1);
    if (NULL == *buf)
    {
        return -1L;
    }
    /* Rewind file pointer to start of file */
    rewind(fp);
    /* Slurp file into buffer */
    if (fsz != fread(*buf, 1, fsz, fp))
    {
        free(*buf);
        return -1L;
    }
    /* Close the file */
    if (EOF == fclose(fp))
    {
        free(*buf);
        return -1L;
    }
    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';
    /* Return the file size */
    return (long)fsz;
}

float **allocMatrix(int rows, int cols){
    float **matrix = (float **)malloc(rows * sizeof(float *)); 
    for (int i=0; i<rows; i++) 
         matrix[i] = (float *)malloc(cols * sizeof(float)); 

    return matrix;
}

/**
 * @brief Popula e normaliza um vetor de floats com os pixels presentes numa string
 * 
 * @param pixels_str String com os pixels separados por espaço
 * @param pixels Vetor de 48*48 espaços nos quais os pixels serão armazenados
 */
void parsePixels(string pixels_str, float *pixels) {
    stringstream ss(pixels_str);
    float pixel;
    int i = 0;

    while (ss >> pixel)
    {
        pixels[i] = pixel / 255;
        i++;
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
void saveEpoch(int epoch, ofstream &outputFile, float *predictions, float *y, int size, float cost) {
    //cout << "done epoch #" <<epoch<<endl;
    float accuracy, precision, recall, f1;
    float tp = 0, tn = 0, fp = 0, fn = 0;
    int pred, real;

    for (int i = 0; i < size; i++){ ///calculando dados para matriz de confusão a partir do valor real e do valor predito
        pred = round(predictions[i]);
        real = round(y[i]);

        if (pred == 0 && real == 0)
            tn++;
        else if (pred == 0 && real == 1)
            fn++;
        else if (pred == 1 && real == 0)
            fp++;
        else
            tp++;
    }

    accuracy = (tp + tn) / (tp + fp + fn + tn);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = (2 * recall * precision) / (recall + precision);

    cout << epoch << "," << accuracy << "," << precision << "," << recall << "," << f1 << "," << cost << endl;

    outputFile << epoch << "," << accuracy << "," << precision << "," << recall << "," << f1 << "," << cost << endl;
}

/**
 * @brief Carrega os argumentos passados via linha de comando nas respectivas variáveis
 * 
 * @param argc Quantidade de argumentos
 * @param argv Vetor contendo o valor dos argumentos
 */
void parse_args(int argc, char **argv) {
    if (argc < 7)
    {
        cout << "Utilização: <executavel> QTD_TREINO QTD_TESTE NUM_EPOCAS TAXA_APRENDIZADO PREFIXO_ARQUIVO_SAIDA" << endl;
        exit(EXIT_FAILURE);
    }

    NUM_TRAIN_OBSERVATIONS = atof(argv[1]);
    NUM_TEST_OBSERVATIONS = atof(argv[2]);
    NUM_EPOCHS = atof(argv[3]);
    LEARNING_RATE = atof(argv[4]);
    OUTPUT_FILE_PREFIX += argv[5];
    OUTPUT_FILE_PREFIX += "/";
    LOCAL_WORK_SIZE = atof(argv[6]);
    GLOBAL_WORK_SIZE = atof(argv[7]);
}

int main(int argc, char **argv) {
    parse_args(argc, argv);
    
    float **X_train, *X_test, *weights;						
    float **y_train, *y_test;
   

    X_train = allocMatrix(NUM_BATCHES, BATCH_SIZE * NUM_FEATURES);
    X_test = (float *)malloc(NUM_TEST_OBSERVATIONS * NUM_FEATURES * sizeof(float));

    y_train = allocMatrix(NUM_BATCHES, BATCH_SIZE);		 
    y_test = (float *)malloc(NUM_TEST_OBSERVATIONS * sizeof(float));		

    weights = (float *) malloc (NUM_FEATURES * sizeof(float));


    for(int i = 0; i < NUM_FEATURES; i++){									
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
    int index = 1, sex, pixelsIndex;
    float pixels[NUM_FEATURES - 1];
	cout << "carregando treino" << endl;
    while(getline(trainFile, line)){    // Leitura do arquivo de entrada (treino)
        sex = line[0] - '0';
        pixelsIndex = line.find(",");
        string pixels_str = line.substr(pixelsIndex+1);
        parsePixels(pixels_str, pixels);

        if (index < NUM_TRAIN_OBSERVATIONS){
            //cout << "index " << index << endl;
            int numBatch = ceil((float)index / BATCH_SIZE) - 1;
            int position = (index % BATCH_SIZE -1) * NUM_FEATURES;
            //cout << "b " << numBatch << " pos "<< position << endl;
            X_train[numBatch][position] = sex;
            for (int i = 1; i < NUM_FEATURES; i++){
                X_train[numBatch][position+i] = pixels[i-1];
            }

            index ++;
        }
    }

    index = 0;

    cout << "carregando teste "<< endl;
    while(getline(testFile, line)){    // Leitura do arquivo de entrada (teste)
        sex = line[0] - '0';
        
        pixelsIndex = line.find(",");
        string pixels_str = line.substr(pixelsIndex+1);
        parsePixels(pixels_str, pixels);

        if (index < NUM_TEST_OBSERVATIONS){
            int maskedIndex = index * NUM_FEATURES;
            X_test[maskedIndex] = 1;
            y_test[index] = sex;

            for (int i = 1; i < NUM_FEATURES; i++){
                X_test[maskedIndex + i] = pixels[i-1];
            }

            index ++;
        }
    }
    
    trainFile.close();
    testFile.close(); 
    
    int err;

    /******** Initial Variables ********/
    cl_uint dev_cnt = 0;
    clGetPlatformIDs(0, 0, &dev_cnt);
	
    cl_platform_id platform_ids[100];
    clGetPlatformIDs(dev_cnt, platform_ids, NULL);

    size_t localWorkSize[1], globalWorkSize[1];

    localWorkSize[0] = LOCAL_WORK_SIZE;
    globalWorkSize[0] = GLOBAL_WORK_SIZE;

    float *predictions = (float*) malloc(NUM_TRAIN_OBSERVATIONS * sizeof(float));

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;   
    int gpu = 1;

    err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
        exit(EXIT_FAILURE);
        //return EXIT_FAILURE;
    }
    
    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context){
        printf("Error: Failed to create a compute context!\n");
        exit(EXIT_FAILURE);
    }

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands){
        printf("Error: Failed to create a command commands!\n");
        exit(EXIT_FAILURE);
    }
    
    char *KernelSource;
    long lFileSize;

    lFileSize = LoadOpenCLKernel("kernel.cl", &KernelSource);
    if( lFileSize < 0L ) {
        printf("File read failed");
        exit(EXIT_FAILURE);
    }

    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program){
        printf("Error: Failed to create compute program!\n");
        exit(EXIT_FAILURE);
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(EXIT_FAILURE);
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "logreg", &err);
    if (!kernel || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        exit(EXIT_FAILURE);
    }

       

    /******** End Buffers ********/
    for(int rep= 0; rep< 5; rep++){
        cout << "-->> REP "<< rep<< endl;
          
        /******** Buffers *******/
        cl_mem buffer_weigths =  clCreateBuffer(context,  CL_MEM_READ_WRITE, NUM_FEATURES * sizeof(float), NULL, &err);

        if(err){
            cout << "Error: Failed to allocate device memory! (weigths, "<<err <<")" << endl;
        }
        
        cl_mem buffer_predictions =  clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_TRAIN_OBSERVATIONS * sizeof(float), NULL, &err);

        if(err){
            cout << "Error: Failed to allocate device memory! (preds, "<<err <<")" << endl;
        }
        
        cl_mem buffer_gradients =  clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_FEATURES * sizeof(float) * LOCAL_WORK_SIZE, NULL, &err);

        if(err){
            cout << "Error: Failed to allocate device memory! (costs, "<<err <<")" << endl;
        }

        if(!buffer_weigths                 ||
            !buffer_predictions            ||
            !buffer_gradients                   
            ){
            printf("Error: Failed to allocate device memory!\n");
            exit(EXIT_FAILURE);
        }

        long elapsedTime = 0;
        for(int b = 0; b < NUM_BATCHES; b++){
            // Create the compute program from the source file
            int X_size = BATCH_SIZE * NUM_FEATURES * sizeof(float);
            int y_size = BATCH_SIZE * sizeof(float);
            float *x_batch = X_train[b];
            float *y_batch = y_train[b];

            cl_mem buffer_X_train = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, X_size, x_batch, &err);

            if(err){
                cout << "Error: Failed to allocate device memory! (X_train, "<<err <<")" << endl;
            }
            
            cl_mem buffer_y_train = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y_size, y_batch, &err);

            if(err){
                cout << "Error: Failed to allocate device memory! (y_train, "<<err <<")" << endl;
            }

            err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &buffer_X_train);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &buffer_y_train);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &buffer_weigths);
            err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &buffer_predictions);
            err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &buffer_gradients);
            err |= clSetKernelArg(kernel, 5, sizeof(int), (void *) &NUM_FEATURES);
            err |= clSetKernelArg(kernel, 6, sizeof(int), (void *) &NUM_TRAIN_OBSERVATIONS);
            err |= clSetKernelArg(kernel, 7, sizeof(float), (void *) &LEARNING_RATE);
            err |= clSetKernelArg(kernel, 8, sizeof(float), (void *) &NUM_EPOCHS);


            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                exit(EXIT_FAILURE);
            }
            auto start = chrono::system_clock::now();

            err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
            

            if (err != CL_SUCCESS){
                printf("Error: Failed to execute kernel! %d\n", err);
                exit(EXIT_FAILURE);
            }

            err = clFinish(commands);
            auto end = chrono::system_clock::now(); // Fim da contagem do tempo de cômputo
            elapsedTime += chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            if(err){
                cout << "Error while running kernel. Error code: "<< err << endl;
                exit(EXIT_FAILURE);
            }

            clReleaseMemObject(buffer_X_train);
            clReleaseMemObject(buffer_y_train);
        }

        // Create the compute program from the source file
        int X_size = NUM_TEST_OBSERVATIONS * NUM_FEATURES * sizeof(float);
        int y_size = NUM_TEST_OBSERVATIONS * sizeof(float);
        float *x_batch = X_test;
        float *y_batch = y_test;

        cl_mem buffer_X_train = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, X_size, x_batch, &err);

        if(err){
            cout << "Error: Failed to allocate device memory! (X_train, "<<err <<")" << endl;
        }
        
        cl_mem buffer_y_train = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y_size, y_batch, &err);

        if(err){
            cout << "Error: Failed to allocate device memory! (y_train, "<<err <<")" << endl;
        }

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &buffer_X_train);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &buffer_y_train);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &buffer_weigths);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &buffer_predictions);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &buffer_gradients);
        err |= clSetKernelArg(kernel, 5, sizeof(int), (void *) &NUM_FEATURES);
        err |= clSetKernelArg(kernel, 6, sizeof(int), (void *) &NUM_TRAIN_OBSERVATIONS);
        err |= clSetKernelArg(kernel, 7, sizeof(float), (void *) &LEARNING_RATE);
        err |= clSetKernelArg(kernel, 8, sizeof(float), (void *) &NUM_EPOCHS);


        if (err != CL_SUCCESS){
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(EXIT_FAILURE);
        }

        auto start = chrono::system_clock::now();

        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        

        if (err != CL_SUCCESS){
            printf("Error: Failed to execute kernel! %d\n", err);
            exit(EXIT_FAILURE);
        }

        err = clFinish(commands);
        auto end = chrono::system_clock::now(); // Fim da contagem do tempo de cômputo
        elapsedTime += chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if(err){
            cout << "Error while running kernel. Error code: "<< err << endl;
            exit(EXIT_FAILURE);
        }

        clReleaseMemObject(buffer_X_train);
        clReleaseMemObject(buffer_y_train);

        err = clEnqueueReadBuffer(commands, buffer_predictions, CL_TRUE, 0, NUM_TRAIN_OBSERVATIONS * sizeof(float), predictions, 0, NULL, NULL);

        if(err != CL_SUCCESS){
            cout << "Error while getting output" << endl;
        }

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        ofstream outputFile;
        string outputFileName = OUTPUT_FILE_PREFIX+"output_" +to_string(NUM_EPOCHS) + "epochs_" + to_string(NUM_TRAIN_OBSERVATIONS) + "train_" + to_string(NUM_TEST_OBSERVATIONS) + "test_@"+to_string(seed)+".txt"; /// Construção do título do arquivo de saída com as estatísticas de treino
        outputFile.open(outputFileName);
        outputFile << "epoch,accuracy,precision,recall,f1,cost" << endl; // Escreve o cabeçalho dos dados no arquivo de saída
        saveEpoch(0, outputFile, predictions, y_test, NUM_TEST_OBSERVATIONS, -1);
        outputFile.close();

        ofstream timeFile;
        timeFile.open(OUTPUT_FILE_PREFIX+"time.txt", ios_base::app); // Concatena o tempo medido no final do arquivo
        timeFile << elapsedTime <<endl;
        timeFile.close();
        outputFile.close();
        

        clReleaseMemObject(buffer_weigths);
        clReleaseMemObject(buffer_predictions);
        clReleaseMemObject(buffer_gradients);
    }

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
   
    free(X_test);
    free(X_train);
    free(y_test);
    free(y_train);
    free(weights);
    free(predictions);

}