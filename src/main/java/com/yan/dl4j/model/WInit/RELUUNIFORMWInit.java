package com.yan.dl4j.model.WInit;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RELUUNIFORMWInit implements Winit {
    @Override
    public INDArray Init(int seed,int out, int in) {
        double u = Math.sqrt(6.0 / in);

        return Nd4j.rand(new int[]{in,out}, Nd4j.getDistributions().createUniform(-u, u)); //U(-sqrt(6/fanIn), sqrt(6/fanIn)
    }
}
