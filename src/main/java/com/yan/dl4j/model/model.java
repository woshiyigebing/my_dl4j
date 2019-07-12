package com.yan.dl4j.model;

import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface model {
    public void train(TrainData data);
    public INDArray predict(INDArray x);
    public double getLearningrate();
    public int getIteration();
}
