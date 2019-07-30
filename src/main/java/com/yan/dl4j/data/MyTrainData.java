package com.yan.dl4j.data;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MyTrainData implements TrainData {
    private INDArray x;
    private INDArray y;

    public MyTrainData(INDArray x,INDArray y){
        this.x = x.transpose();
        this.y = y.transpose();
    }

    @Override
    public INDArray getX() {
        return x;
    }

    @Override
    public INDArray getY() {
        return y;
    }

    @Override
    public int getSize() {
        return getX().columns();
    }

    public void setX(INDArray x) {
        this.x = x;
    }

    public void setY(INDArray y) {
        this.y = y;
    }

}
