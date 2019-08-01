package com.yan.dl4j.model;

import com.yan.dl4j.Utils.MyMathUtil;
import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class DeepNeuralNetWork implements model {

    private INDArray[] Network_W;
    private INDArray[] Network_B;

    private double learningrate=0.03;

    private int iteration = 100000;

    //构建一个最后一层为relu激活函数，其他为sigmoid激活函数的深度神经网络
    public DeepNeuralNetWork(List<Integer> layers){
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
        }else if(activate.equals("sigmoid")){
            return MyMathUtil.MysigMoid(linear_forward(A_p,W,b));
        }else if(activate.equals("tanh")){
            return MyMathUtil.Mytanh(linear_forward(A_p,W,b));
        }
        return null;

    }

    private INDArray[] forward(INDArray X){
        INDArray[] res = new INDArray[Network_W.length];
        INDArray P_A = X;
        for(int i=0;i<Network_W.length;i++){
           if(i+1>=Network_W.length){ //最后一个
               INDArray A = linear_activate_forward(P_A,Network_W[i],Network_B[i],"sigmoid");
              res[i] = A;
           }else{
               INDArray A = linear_activate_forward(P_A,Network_W[i],Network_B[i],"tanh");
               P_A = A;
               res[i] = A;
           }
        }
        return res;
    }

    private INDArray activate_backward(INDArray DA, INDArray A, String activate){
        if(activate.equals("relu")){
            return MyMathUtil.relu_back(DA);
        }else if(activate.equals("sigmoid")){
            return DA.mul(A).mul(Nd4j.ones(A.shape()).sub(A));
        }else if(activate.equals("tanh")){
            return DA.mul(Nd4j.ones(A.shape()).sub(A.mul(A)));
        }
        return null;
    }

    private List<INDArray[]> backward(INDArray[] A_array,INDArray x,INDArray Y){
        INDArray DZ = activate_backward(A_array[A_array.length-1].sub(Y),A_array[A_array.length-1],"sigmoid");
        INDArray[] DW = new INDArray[A_array.length];
        INDArray[] DB = new INDArray[A_array.length];
        List<INDArray[]> res = new ArrayList<>();
        for(int i=A_array.length-1;i>=0;i--){
            if(i==0){ //最后一次
                INDArray dW = DZ.mmul(x.transpose());
                INDArray dB = DZ.mmul(Nd4j.ones(x.shape()[1],1));
                DW[i] = dW;
                DB[i] = dB;
            }else{
                INDArray dW = DZ.mmul(A_array[i-1].transpose());
                INDArray dB = DZ.mmul(Nd4j.ones(x.shape()[1],1));
                DW[i] = dW;
                DB[i] = dB;
                DZ = activate_backward(Network_W[i].transpose().mmul(DZ),A_array[i-1],"tanh");
            }
        }
        res.add(DW);
        res.add(DB);
        return res;
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

        for(int i=0;i<getIteration();i++) {

            INDArray[] A = forward(data.getX()); //向前传播

            List<INDArray[]> DW_DB = backward(A, data.getX(), data.getY()); //反向传播

            update_parameters(DW_DB); //梯度下降

            double loss = data.getY().sub(A[A.length - 1]).mmul(data.getY().sub(A[A.length - 1]).transpose()).sumNumber().doubleValue();

            //打印情况
            System.out.println("i=" + i);
            System.out.println("loss=" + loss);
        }

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
}
