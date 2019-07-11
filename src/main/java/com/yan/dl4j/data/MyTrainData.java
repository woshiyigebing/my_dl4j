package com.yan.dl4j.data;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MyTrainData implements TrainData {
    private INDArray x;
    private INDArray y;

    public MyTrainData(INDArray x,INDArray y){
        this.x = x;
        this.y = y;
    }

    @Override
    public INDArray getX() {
        return null;
    }

    @Override
    public INDArray getY() {
        return null;
    }

    public void setX(INDArray x) {
        this.x = x;
    }

    public void setY(INDArray y) {
        this.y = y;
    }
}
