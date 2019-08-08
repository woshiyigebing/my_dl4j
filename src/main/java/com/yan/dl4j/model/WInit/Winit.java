package com.yan.dl4j.model.WInit;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Winit {
    INDArray Init(int seed,int out,int in);
}
