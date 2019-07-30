package com.yan.dl4j.model;

import com.yan.dl4j.Utils.MyMathUtil;
import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class DeepNeuralNetWork implements model {

    private INDArray[] Network_W;
    private INDArray[] Network_B;

    private double learningrate=0.01;

    private int iteration = 10000;

    //构建一个最后一层为relu激活函数，其他为sigmoid激活函数的深度神经网络
    DeepNeuralNetWork(List<Integer> layers){
        Network_W = new INDArray[layers.size()-1];
        Network_B = new INDArray[layers.size()-1];
        for(int i=1;i<layers.size();i++){
            Network_W[i-1] = Nd4j.rand(layers.get(i), layers.get(i-1));
            Network_B[i-1] = Nd4j.zeros(layers.get(i), 1);
        }
    }

    private INDArray linear_forward(INDArray A, INDArray W, INDArray b){
        return W.mmul(A).addColumnVector(b);
    }

    private INDArray linear_activate_forward(INDArray A_p, INDArray W, INDArray b,String activate){
        if(activate.equals("relu")){
            return MyMathUtil.relu(linear_forward(A_p,W,b));
        }else{
            return MyMathUtil.MysigMoid(linear_forward(A_p,W,b));
        }

    }

    private INDArray[] forward(INDArray X){
        INDArray[] res = new INDArray[Network_W.length];
        INDArray P_A = X;
        for(int i=0;i<Network_W.length;i++){
           if(i+1>=Network_W.length){ //最后一个
               INDArray A = linear_activate_forward(P_A,Network_W[i],Network_B[i],"relu");
              res[i] = A;
           }else{
               INDArray A = linear_activate_forward(P_A,Network_W[i],Network_B[i],"sigmoid");
               P_A = A;
               res[i] = A;
           }
        }
        return res;
    }

    private List<INDArray[]> backward(INDArray[] A_array){
        return null;
    }

    private void update_parameters(List<INDArray[]> DW_DB){
        INDArray[] DW =  DW_DB.get(0);
        INDArray[] DB =  DW_DB.get(1);
        for(int i=0;i<Network_W.length;i++){
            Network_W[i] =Network_W[i].sub(DW[i].mul(getLearningrate()));
        }
        for(int i=0;i<Network_B.length;i++){
            Network_B[i] =Network_B[i].sub(DB[i].mul(getLearningrate()));
        }
    }

    @Override
    public void train(TrainData data) {

    }

    @Override
    public INDArray predict(INDArray x) {
        INDArray[] A = forward(x);
        return A[A.length-1];
    }

    @Override
    public double getLearningrate() {
        return learningrate;
    }

    @Override
    public int getIteration() {
        return iteration;
    }

    @Override
    public void verify(TrainData data) {

    }
}
