package com.yan.dl4j.model;

import com.yan.dl4j.Utils.MyMathUtil;
import com.yan.dl4j.data.TrainData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.omg.PortableInterceptor.INACTIVE;
import oshi.util.MapUtil;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class NeuralNetwork implements model  {

    private double learningrate=0.01;

    private int iteration = 1000;

    //4个神经元 3个特征值 20个数据
    //w = 4*3  x = 3*20 y = 4*20   =》z1 = 4*20;
    private INDArray Network_1LAYER_W = Nd4j.ones(4, 3);
    private INDArray Network_1LAYER_B = Nd4j.ones(4, 1);
    //1个神经元 4个特征值 1个数据
    //w = 1*4 x = 4*20 y = 1*20 => z2 = 1*20;
    private INDArray Network_2LAYER_W = Nd4j.create(new float[]{2, 2, 2, 2}, new int[]{1, 4});
    private INDArray Network_2LAYER_B = Nd4j.create(new float[]{2}, new int[]{1, 1});

    public static void main(String[] args) {
        INDArray my_Network_2LAYER_W = Nd4j.rand(4, 20);
        System.out.println(my_Network_2LAYER_W);
       my_Network_2LAYER_W = MyMathUtil.MysigMoid(my_Network_2LAYER_W);
        System.out.println(my_Network_2LAYER_W);
    }

    private Map<String,INDArray> forward(INDArray X){
        Map<String,INDArray> map = new HashMap<>();
        INDArray Z1 = getNetwork_1LAYER_W().mmul(X).add(getNetwork_1LAYER_B());
        INDArray A1 = MyMathUtil.Mytanh(Z1);
        INDArray Z2 = getNetwork_2LAYER_W().mmul(A1).add(getNetwork_2LAYER_B());
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
            INDArray dW1 = null;
            INDArray dW2 = null;
            INDArray dB1 = null;
            INDArray dB2 = null;
            //Loss函数
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
