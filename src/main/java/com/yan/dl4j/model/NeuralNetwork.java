package com.yan.dl4j.model;

import com.yan.dl4j.Utils.MyMathUtil;
import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.HashMap;
import java.util.Map;

public class NeuralNetwork implements model  {

    private double learningrate=0.1;

    private int iteration = 10000;

    //4个神经元 2个特征值 400个数据
    //W1 = 4*2  X = 2*400 Z1 = 4*400   =》A1 = 4*400;
    private INDArray Network_1LAYER_W = Nd4j.randn(4, 2);
    private INDArray Network_1LAYER_B = Nd4j.zeros(4, 1);
    //1个神经元 4个特征值 1个数据
    //W2 = 1*4 A1 = 4*400 Z2 = 1*400 => A2 = 1*400;
    private INDArray Network_2LAYER_W = Nd4j.randn(1, 4);
    private INDArray Network_2LAYER_B = Nd4j.zeros(1, 1);

    public static void main(String[] args) {
        INDArray my_Network_2LAYER_W = Nd4j.rand(1, 20);
        System.out.println(my_Network_2LAYER_W);
       my_Network_2LAYER_W = MyMathUtil.MysigMoid(my_Network_2LAYER_W);
        System.out.println(my_Network_2LAYER_W);
    }

    private Map<String,INDArray> forward(INDArray X){
        Map<String,INDArray> map = new HashMap<>();
        INDArray Z1 = getNetwork_1LAYER_W().mmul(X).addColumnVector(getNetwork_1LAYER_B());
        INDArray A1 = MyMathUtil.Mytanh(Z1);
        INDArray Z2 = getNetwork_2LAYER_W().mmul(A1).addColumnVector(getNetwork_2LAYER_B());
        INDArray A2 = MyMathUtil.MysigMoid(Z2);
        map.put("Z1",Z1);
        map.put("A1",A1);
        map.put("Z2",Z2);
        map.put("A2",A2);
        return map;
    }


    @Override
    public void train(TrainData data) {

        for(int i=0;i<getIteration();i++){
            INDArray X = data.getX();
            INDArray Y = data.getY();
            //向前传播
            Map<String,INDArray> map = forward(X);
            INDArray Z1 = map.get("Z1");
            INDArray A1 = map.get("A1");
            INDArray Z2 = map.get("Z2");
            INDArray A2 = map.get("A2");
            //向后传播
            //dL/dA2 = -(Y-A2)
            //dA2/dZ2 = A2(1-A2)
            //dZ2/dW2 = A1
            //dZ/dB2 = 1
            //dZ2/dA1 = W2
            //dA1/dZ1 = 1 − A1^2
            //dZ1/dW1 = X
            //dZ1/dB1 = 1
            INDArray dZ2 = Y.sub(A2).mul(A2).mul(A2.sub(1)); //1*400
            INDArray dW2 = dZ2.mmul(A1.transpose()); //1*4
            INDArray dB2 = dZ2.mmul(Nd4j.ones(data.getX().shape()[1],1));  //1*1
            INDArray dA1Z1 = Nd4j.ones(A1.shape()).sub(A1.mul(A1));
            INDArray dZ1 = getNetwork_2LAYER_W().transpose().mmul(dZ2).mul(dA1Z1);   //(W2^T dZ2)*(1-A1^2);
            INDArray dW1 = dZ1.mmul(X.transpose());  //4*2
            INDArray dB1 = dZ1.mmul(Nd4j.ones(data.getX().shape()[1],1)); //4*1
            //Loss函数 l = (1/2)*(y-yi)^2
            double loss =  Y.sub(A2).mmul(Y.sub(A2).transpose()).sumNumber().doubleValue();
            //梯度下降
            //W1 = W1 - dW1*r;
                INDArray W1 = getNetwork_1LAYER_W().sub(dW1.mul(getLearningrate()));
                INDArray W2 = getNetwork_2LAYER_W().sub(dW2.mul(getLearningrate()));
                INDArray B1 = getNetwork_1LAYER_B().sub(dB1.mul(getLearningrate()));
                INDArray B2 = getNetwork_2LAYER_B().sub(dB2.mul(getLearningrate()));
                setNetwork_1LAYER_W(W1);
                setNetwork_2LAYER_W(W2);
                setNetwork_1LAYER_B(B1);
                setNetwork_2LAYER_B(B2);
            //打印情况
            System.out.println("i="+i);
            System.out.println("loss="+loss);
        }
    }

    @Override
    public INDArray predict(INDArray x) {
        Map<String,INDArray> map = forward(x);
        return map.get("A2");
    }

    @Override
    public double getLearningrate() {
        return learningrate;
    }

    @Override
    public int getIteration() {
        return iteration;
    }


    private INDArray getNetwork_1LAYER_W() {
        return Network_1LAYER_W;
    }

    private INDArray getNetwork_1LAYER_B() {
        return Network_1LAYER_B;
    }

    private INDArray getNetwork_2LAYER_W() {
        return Network_2LAYER_W;
    }

    private INDArray getNetwork_2LAYER_B() {
        return Network_2LAYER_B;
    }

    private void setNetwork_1LAYER_W(INDArray network_1LAYER_W) {
        Network_1LAYER_W = network_1LAYER_W;
    }

    private void setNetwork_1LAYER_B(INDArray network_1LAYER_B) {
        Network_1LAYER_B = network_1LAYER_B;
    }

    private void setNetwork_2LAYER_W(INDArray network_2LAYER_W) {
        Network_2LAYER_W = network_2LAYER_W;
    }

    private void setNetwork_2LAYER_B(INDArray network_2LAYER_B) {
        Network_2LAYER_B = network_2LAYER_B;
    }

}
