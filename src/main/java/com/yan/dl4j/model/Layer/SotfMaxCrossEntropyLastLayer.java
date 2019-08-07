package com.yan.dl4j.model.Layer;

import com.yan.dl4j.model.Activate.SoftMax;
import com.yan.dl4j.model.Loss.CrossEntropy;
import com.yan.dl4j.model.Loss.LossMethod;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SotfMaxCrossEntropyLastLayer extends MyLayer implements LastLayer {

    private LossMethod lossMethod;

    public SotfMaxCrossEntropyLastLayer(int number){
        super(number,new SoftMax());
        this.lossMethod = new CrossEntropy();
    }

    @Override
    public LossMethod getLossMethod() {
        return lossMethod;
    }

    @Override
    public LastLayer setLossMethod(LossMethod lossMethod) {
        this.lossMethod = lossMethod;
        return this;
    }

//    @Override
//    public INDArray LastBackward(INDArray A, INDArray Y) {
//        return A.sub(Y); //简化操作
//    }
}
