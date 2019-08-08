package com.yan.dl4j.model.WInit;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RandWInit implements Winit {
    @Override
    public INDArray Init(int seed, int out, int in) {
        return Nd4j.rand(out,in,seed);
    }
}
