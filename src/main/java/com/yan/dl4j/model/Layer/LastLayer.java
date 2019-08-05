package com.yan.dl4j.model.Layer;

import com.yan.dl4j.model.Loss.LossMethod;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface LastLayer extends Layer {
    LossMethod getLossMethod();
    LastLayer setLossMethod(LossMethod lossMethod);
    default INDArray LastBackward(INDArray A,INDArray Y){
        return getActivateMethod().activate_backward(getLossMethod().LossBackward(A,Y),A);
    }
}
