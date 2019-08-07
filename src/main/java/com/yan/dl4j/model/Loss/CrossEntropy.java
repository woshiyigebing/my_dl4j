package com.yan.dl4j.model.Loss;

import com.yan.dl4j.Utils.MyMathUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class CrossEntropy implements LossMethod {
    @Override
    public INDArray LossBackward(INDArray A, INDArray Y) {
        return Nd4j.zeros(Y.div(A).shape()).sub(Y.div(A));
    }

    @Override
    public double LossForward(INDArray A, INDArray Y) {
        INDArray los = Y.mul(MyMathUtil.Log(A));
        return  (0-los.sumNumber().doubleValue())/Y.shape()[1];
    }
}
