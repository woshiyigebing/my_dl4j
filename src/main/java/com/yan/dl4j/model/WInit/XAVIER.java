package com.yan.dl4j.model.WInit;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class XAVIER implements Winit{

    @Override
    public INDArray Init(int seed,int out, int in) {
        return Nd4j.randn(out,in,seed).muli(FastMath.sqrt(2.0 / (in+out)));
    }
}
