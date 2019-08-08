package com.yan.dl4j.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MyTrainData implements TrainData {
    private INDArray x;
    private INDArray y;
    private int batch_size;

    public MyTrainData(INDArray x,INDArray y,int batch_size){
        this.x = x.transpose();
        this.y = y.transpose();
        this.batch_size = batch_size;
    }
    public MyTrainData(INDArray x,INDArray y){
        this.x = x.transpose();
        this.y = y.transpose();
        this.batch_size = -1;
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
    public List<INData> getBatchList() {
        List<INData> res = new ArrayList<>();
        shufflecard();
        if(batch_size!=-1){
            int lastColumnOrder = 0;
            for(int i=batch_size;i<getSize();i=i+batch_size){
                INDArray BatchColumn_x = x.get(NDArrayIndex.all(), NDArrayIndex.interval(lastColumnOrder,i));
                INDArray BatchColumn_y = y.get(NDArrayIndex.all(), NDArrayIndex.interval(lastColumnOrder,i));
                INData data = new BatchData(BatchColumn_x,BatchColumn_y);
                res.add(data);
                lastColumnOrder = i;
            }
            if(lastColumnOrder!=getSize()){
                INDArray BatchColumn_x = x.get(NDArrayIndex.all(), NDArrayIndex.interval(lastColumnOrder,getSize()));
                INDArray BatchColumn_y = y.get(NDArrayIndex.all(), NDArrayIndex.interval(lastColumnOrder,getSize()));
                INData data = new BatchData(BatchColumn_x,BatchColumn_y);
                res.add(data);
            }
        }else{
            INData data = new BatchData(getX(),getY());
            res.add(data);
        }
        return res;
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

    public void shufflecard(){
        Random rd = new Random();
        INDArray temp_x ;
        INDArray temp_y ;
        for(int i=0;i<getSize();i++){
            int j = rd.nextInt(getSize());
            temp_x = x.getColumn(i).add(0);
            temp_y = y.getColumn(i).add(0);
            x.putColumn(i,x.getColumn(j).add(0));
            x.putColumn(j,temp_x);
            y.putColumn(i,y.getColumn(j));
            y.putColumn(j,temp_y);
        }
    }


}
