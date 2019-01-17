__kernel void logreg(
    __global float* X_train,
    __global float* y_train,
    __global float* weights,
    __global float* predictions,
    __global float* partialGradients,
    const int NUM_FEATURES,
    const int NUM_TRAIN_OBSERVATIONS,
    const float LEARNING_RATE,
    const int NUM_EPOCHS)
{
    
    //NUM_EPOCHS = 500;
     //NUM_TRAIN_OBSERVATIONS = 10640;
     //LEARNING_RATE = 0.001;
     //NUM_FEATURES = 128*128+1;
    
   const int localId = get_global_id(0);
   const int NUM_WORK_ITEMS = 256;
   const int localExamplesStart = NUM_TRAIN_OBSERVATIONS/NUM_WORK_ITEMS * localId;

   for(int ep = 0; ep < 10; ep++){

        for(int k = 0; k < NUM_FEATURES; k++){
                partialGradients[localId*NUM_FEATURES + k] = 0.0f;
                weights[k] = 0.0f;
        }

        for (int i = 0; i < NUM_TRAIN_OBSERVATIONS/NUM_WORK_ITEMS; i++){
            float z = 0.0f;

            for (int j = 0; j < NUM_FEATURES; j++){
                z += X_train[(localExamplesStart + i) * NUM_FEATURES + j] * weights[j];
            }

            predictions[localExamplesStart + i] =  1.0/(1.0+exp(-z));
            z = y_train[localExamplesStart+i] - 1.0/(1.0+exp(-z));

            for(int j = 0; j < NUM_FEATURES; j++){
                partialGradients[localId * NUM_FEATURES + j] += X_train[(localExamplesStart+i)*NUM_FEATURES + j] * z;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(localId == 0){
            for(int k = 0; k< NUM_FEATURES; k++){
                for (int j = 0; j< NUM_WORK_ITEMS; j++){
                    partialGradients[k] += partialGradients[j*NUM_FEATURES + k];
                }
            }

            for(int j = 0; j < NUM_FEATURES; j++){
                weights[j] += (LEARNING_RATE/NUM_TRAIN_OBSERVATIONS) * partialGradients[j];
            }

        }

    predictions[0] = get_local_size(0);
    predictions[1] = get_global_size(0);
    predictions[2] = NUM_TRAIN_OBSERVATIONS;
    predictions[3] = -1;

        barrier(CLK_GLOBAL_MEM_FENCE);
   }
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    predictions[3] = -1;

}