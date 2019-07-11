package com.yan.dl4j.data;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface TrainData {
    public INDArray getX();
    public INDArray getY();
}
