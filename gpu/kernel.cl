__kernel void logreg(
    __global float4* X_train,
    __global float* y_train,
    __global float4* weights,
    __global float* predictions,
    __global float4* partialGradients,
    int NUM_FEATURES,
    int NUM_TRAIN_OBSERVATIONS,
    float LEARNING_RATE,
    int NUM_EPOCHS)
{
    
    //NUM_EPOCHS = 500;
     //NUM_TRAIN_OBSERVATIONS = 10640;
     //LEARNING_RATE = 0.001;
     //NUM_FEATURES = 128*128+1;
    
   const int localId = get_global_id(0);
   const int NUM_WORK_ITEMS = get_global_size(0);
   const int localExamplesStart = NUM_TRAIN_OBSERVATIONS/NUM_WORK_ITEMS * localId;

   for(int ep = 0; ep < 10; ep++){

        for(int k = 0; k < NUM_FEATURES/4; k++){
                partialGradients[localId*NUM_FEATURES/4 + k] = 0.0f;
                weights[k/4] = 0.0f;
        }

        for (int i = 0; i < NUM_TRAIN_OBSERVATIONS/NUM_WORK_ITEMS; i++){
            float z = 0.0f;

            for (int j = 0; j < NUM_FEATURES/4; j++){
                z += dot(X_train[(localExamplesStart + i) * NUM_FEATURES/4 + j], weights[j]);
            }

            predictions[localExamplesStart + i] =  1.0/(1.0+exp(-z));
            z = y_train[localExamplesStart+i] - 1.0/(1.0+exp(-z));

            for(int j = 0; j < NUM_FEATURES/4; j++){
                partialGradients[localId * NUM_FEATURES/4 + j] += X_train[(localExamplesStart+i)*NUM_FEATURES/4 + j] * z;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int offset = NUM_WORK_ITEMS/2; offset > 0; offset >>= 1)
        { 
            for (int k = 0; k < NUM_FEATURES/4; k++)
                partialGradients[localId*NUM_FEATURES/4 + k] += partialGradients[(localId+offset)*NUM_FEATURES/4 + k]; 
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        if(localId == 0){
            for(int j = 0; j < NUM_FEATURES/4; j++){
                weights[j] += (LEARNING_RATE/NUM_TRAIN_OBSERVATIONS) * partialGradients[j];
            }

        }

   

        barrier(CLK_GLOBAL_MEM_FENCE);
   }
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    predictions[0] = get_local_size(0);
    predictions[1] = get_global_size(0);
    predictions[2] = NUM_TRAIN_OBSERVATIONS;
    predictions[3] = -1;
}