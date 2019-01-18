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

/**Source: http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl**/
long LoadOpenCLKernel(char const *path, char **buf)
{
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


/**
 * @brief Popula e normaliza um vetor de floats com os pixels presentes numa string
 * 
 * @param pixels_str String com os pixels separados por espaço
 * @param pixels Vetor de 48*48 espaços nos quais os pixels serão armazenados
 */
void parsePixels(string pixels_str, float *pixels)
{
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
 * @brief 
 * 
 * @param X Matriz de características
 * @param y Vetor de marcações (gabaritos)
 * @param index Índice na observação/marcação atual
 * @param sex Emoção normalizada (0 ou 1)
 * @param pixels Vetor de 48*48 posições contendo os valores normalizado de cada pixel
 */
void addToDataset(float *X, float *y, int index, int sex, float* pixels){
    int maskedIndex = index * NUM_FEATURES;
    X[maskedIndex] = 1;
    y[index] = sex;

    for (int i = 1; i < NUM_FEATURES; i++){
        X[maskedIndex + i] = pixels[i-1];
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
void saveEpoch(int epoch, ofstream &outputFile, float *predictions, float *y, int size, float cost)
{
    //cout << "done epoch #" <<epoch<<endl;
    float accuracy, precision, recall, f1;
    float tp = 0, tn = 0, fp = 0, fn = 0;
    int pred, real;

    for (int i = 0; i < size; i++)
    { ///calculando dados para matriz de confusão a partir do valor real e do valor predito
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
}

/**
 * @brief Carrega os argumentos passados via linha de comando nas respectivas variáveis
 * 
 * @param argc Quantidade de argumentos
 * @param argv Vetor contendo o valor dos argumentos
 */
void parse_args(int argc, char **argv)
{
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

int main(int argc, char **argv)
{
    parse_args(argc, argv);
    
    float *X_train, *X_test, *weights;						
    float *y_train, *y_test;	

    X_train = (float *) malloc(NUM_TRAIN_OBSERVATIONS * NUM_FEATURES * sizeof(float));
    X_test = (float *) malloc(NUM_TEST_OBSERVATIONS * NUM_FEATURES * sizeof(float));    			
    y_train = (float *)malloc(NUM_TRAIN_OBSERVATIONS * sizeof(float));		 
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
    int index = 0, sex, pixelsIndex;
    float pixels[NUM_FEATURES - 1];
	cout << "carregando treino" << endl;
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
    index = 0;

    cout << "carregando teste "<< endl;
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
    
    int err;
    /******** Initial Variables ********/
    cl_uint dev_cnt = 0;
   clGetPlatformIDs(0, 0, &dev_cnt);
	
   cl_platform_id platform_ids[100];
   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	
   // Connect to a compute device
   cl_device_id device_id;             // compute device id 
   cl_context context;                 // compute context
   cl_command_queue commands;          // compute command queue
   cl_program program;                 // compute program
   cl_kernel kernel;          

   int gpu = 1;

   err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to create a device group!\n");
       //return EXIT_FAILURE;
   }
  
   // Create a compute context 
   context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
   if (!context)
   {
       printf("Error: Failed to create a compute context!\n");
       //return EXIT_FAILURE;
   }

   // Create a command commands
   commands = clCreateCommandQueue(context, device_id, 0, &err);
   if (!commands)
   {
       printf("Error: Failed to create a command commands!\n");
       //return EXIT_FAILURE;
   }

   // Create the compute program from the source file
   char *KernelSource;
   long lFileSize;

   lFileSize = LoadOpenCLKernel("kernel.cl", &KernelSource);
   if( lFileSize < 0L ) {
       perror("File read failed");
       //return 1;
   }

   program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
   if (!program)
   {
       printf("Error: Failed to create compute program!\n");
       //return EXIT_FAILURE;
   }

   // Build the program executable
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];
       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       //exit(1);
   }

   // Create the compute kernel in the program we wish to run
   //
   kernel = clCreateKernel(program, "logreg", &err);
   if (!kernel || err != CL_SUCCESS)
   {
       printf("Error: Failed to create compute kernel!\n");
       //exit(1);
   }

    /******** Buffers *******/
    int X_size = NUM_TRAIN_OBSERVATIONS * NUM_FEATURES * sizeof(float);
    int y_size = NUM_TRAIN_OBSERVATIONS * sizeof(float);

    cl_mem buffer_X_train = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, X_size, X_train, &err);

    if(err){
        cout << "Error: Failed to allocate device memory! (X_train, "<<err <<")" << endl;
    }
    
    cl_mem buffer_y_train = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y_size, y_train, &err);

     if(err){
        cout << "Error: Failed to allocate device memory! (y_train, "<<err <<")" << endl;
    }
    

    cl_mem buffer_weigths =  clCreateBuffer(context,  CL_MEM_READ_WRITE, NUM_FEATURES * sizeof(float), NULL, &err);

     if(err){
        cout << "Error: Failed to allocate device memory! (weigths, "<<err <<")" << endl;
    }
    
    
    cl_mem buffer_predictions =  clCreateBuffer(context, CL_MEM_READ_WRITE, y_size, NULL, &err);

     if(err){
        cout << "Error: Failed to allocate device memory! (preds, "<<err <<")" << endl;
    }
    
    cl_mem buffer_costs =  clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_FEATURES/4 * sizeof(float) * GLOBAL_WORK_SIZE, NULL, &err);

     if(err){
        cout << "Error: Failed to allocate device memory! (costs, "<<err <<")" << endl;
    }
    


    if(!buffer_X_train                  || 
        !buffer_y_train                 ||
        !buffer_weigths                 ||
        !buffer_predictions             ||
        !buffer_costs                   
        ){
        printf("Error: Failed to allocate device memory!\n");
        cout << "x: "<<buffer_X_train << ", y: " << buffer_y_train << ", w: " << buffer_weigths << ", preds:" << buffer_predictions << " costs:" << buffer_costs << endl;
        //exit(1);
    }

    if(err != CL_SUCCESS){
            cout << "Erro escrita";
    }

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &buffer_X_train);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &buffer_y_train);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &buffer_weigths);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &buffer_predictions);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &buffer_costs);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *) &NUM_FEATURES);
    err |= clSetKernelArg(kernel, 6, sizeof(int), (void *) &NUM_TRAIN_OBSERVATIONS);
    err |= clSetKernelArg(kernel, 7, sizeof(float), (void *) &LEARNING_RATE);
    err |= clSetKernelArg(kernel, 8, sizeof(float), (void *) &NUM_EPOCHS);


    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        //exit(1);
    }

    size_t localWorkSize[2], globalWorkSize[2];

    localWorkSize[0] = 16;
    localWorkSize[1] = 16;
    globalWorkSize[0] = 1024;
    globalWorkSize[1] = 1024;

    float *predictions = (float*) malloc(NUM_TRAIN_OBSERVATIONS * sizeof(float));

    /******** End Buffers ********/
    for(int rep= 0; rep< 5; rep++){
        cout << "-->> REP "<< rep<< endl;
        auto start = chrono::system_clock::now();

        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to execute kernel! %d\n", err);
            //exit(1);
        }

        err = clFinish(commands);

        auto end = chrono::system_clock::now(); // Fim da contagem do tempo de cômputo
        long elapsed = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


        if(err){
            cout << "Error while running kernel. Error code: "<< err << endl;
        }


        err = clEnqueueReadBuffer(commands, buffer_predictions, CL_TRUE, 0, NUM_TRAIN_OBSERVATIONS * sizeof(float), predictions, 0, NULL, NULL);

        if(err != CL_SUCCESS){
            cout << "Error while getting output" << endl;
        }

        //cout << "global "<< predictions[1] << endl;
        //cout << "local "<< predictions[0] << endl;
        //cout << "obs "<< predictions[2] << endl;
        //cout << "check "<< predictions[3] << endl;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        ofstream outputFile;
        string outputFileName = OUTPUT_FILE_PREFIX+"output_" +to_string(NUM_EPOCHS) + "epochs_" + to_string(NUM_TRAIN_OBSERVATIONS) + "train_" + to_string(NUM_TEST_OBSERVATIONS) + "test_@"+to_string(seed)+".txt"; /// Construção do título do arquivo de saída com as estatísticas de treino
        outputFile.open(outputFileName);
        outputFile << "epoch,accuracy,precision,recall,f1,cost" << endl; // Escreve o cabeçalho dos dados no arquivo de saída
        saveEpoch(0, outputFile, predictions, y_train, NUM_TRAIN_OBSERVATIONS, -1);
        outputFile.close();

        ofstream timeFile;
        timeFile.open(OUTPUT_FILE_PREFIX+"time.txt", ios_base::app); // Concatena o tempo medido no final do arquivo
        timeFile << elapsed <<endl;
        timeFile.close();
        outputFile.close();
    }

   clReleaseMemObject(buffer_X_train);
   clReleaseMemObject(buffer_y_train);
   clReleaseMemObject(buffer_weigths);
   clReleaseMemObject(buffer_predictions);
   clReleaseMemObject(buffer_costs);

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
