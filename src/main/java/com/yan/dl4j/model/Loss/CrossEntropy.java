package com.yan.dl4j.model.Loss;

import com.yan.dl4j.Utils.MyMathUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

public class CrossEntropy implements LossMethod {
    @Override
    public INDArray LossBackward(INDArray A, INDArray Y) {
        return Y.div(A);
    }

    @Override
    public double LossForward(INDArray A, INDArray Y) {
        INDArray los = Y.mul(MyMathUtil.Log(A));
        return  (0-los.sumNumber().doubleValue());
    }
}
