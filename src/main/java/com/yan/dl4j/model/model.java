package com.yan.dl4j.model;

import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface model {
     void train(TrainData data);
     INDArray predict(INDArray x);
     double getLearningrate();
     int getIteration();
     model setLearningrate(double rate);
     model setIteration(int iteration);
}
