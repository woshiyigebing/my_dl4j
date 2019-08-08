package com.yan.dl4j.model.NetWork;

import com.yan.dl4j.Utils.MyMathUtil;
import com.yan.dl4j.data.INData;
import com.yan.dl4j.data.TrainData;
import com.yan.dl4j.model.Activate.ActivateMethod;
import com.yan.dl4j.model.Layer.LastLayer;
import com.yan.dl4j.model.Layer.Layer;

import com.yan.dl4j.model.model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class DeepNeuralNetWork implements model {
    private List<Layer> layers = new ArrayList<>();
    private LastLayer lastLayer;
    private INDArray[] Network_W;
    private INDArray[] Network_B;
    private int nin;
    private int seed=123;

    private double learningrate=0.01;

    private int iteration = 10;
    public DeepNeuralNetWork(int nin){
        this.nin = nin;
    }
    public DeepNeuralNetWork addLayer(Layer layer){
        layers.add(layer);
        return this;
    }
    public DeepNeuralNetWork addLastLayer(LastLayer lastLayer){
        this.lastLayer = lastLayer;
        Init();
        return this;
    }
    public void Init(){
        Network_W = new INDArray[layers.size()+1];
        Network_B = new INDArray[layers.size()+1];
        for(int i=0;i<layers.size()+1;i++){
            if(i==0){ //第一个
                Network_W[i] = layers.get(i).getWinit().Init(seed,layers.get(i).getNeuralNumber(), nin);
                Network_B[i] = Nd4j.zeros(layers.get(i).getNeuralNumber(), 1);
            }else if(i==layers.size()){ //最后一个
                Network_W[i] = layers.get(i-1).getWinit().Init(seed,lastLayer.getNeuralNumber(), layers.get(i-1).getNeuralNumber());
                Network_B[i] = Nd4j.zeros(lastLayer.getNeuralNumber(), 1);
            }else{
                Network_W[i] = layers.get(i-1).getWinit().Init(seed,layers.get(i).getNeuralNumber(), layers.get(i-1).getNeuralNumber());
                Network_B[i] = Nd4j.zeros(layers.get(i).getNeuralNumber(), 1);
            }
        }
    }
    private INDArray linear_forward(INDArray A, INDArray W, INDArray b){
        return W.mmul(A).addColumnVector(b);
    }
    private INDArray linear_activate_forward(INDArray A_p, INDArray W, INDArray b, ActivateMethod activate){
        if(activate!=null){
            return activate.activate_forward(linear_forward(A_p,W,b));
        }
        return linear_forward(A_p,W,b);
    }
    private INDArray[] forward(INDArray X){
        INDArray[] res = new INDArray[layers.size()+1];
        INDArray P_A = X;
        for(int i=0;i<layers.size();i++){
            INDArray A = linear_activate_forward(P_A,Network_W[i],Network_B[i],layers.get(i).getActivateMethod());
            P_A = A;
            res[i] = A;
        }
        //最后一层
        INDArray A = linear_activate_forward(P_A,Network_W[layers.size()],Network_B[layers.size()],lastLayer.getActivateMethod());
        res[layers.size()] = A;
        return res;
    }

    private INDArray activate_backward(INDArray DA, INDArray A, ActivateMethod activate){
        if(activate!=null){
            return activate.activate_backward(DA,A);
        }
        return DA;
    }

    private INDArray LossBackward(INDArray A,INDArray Y){
        return lastLayer.LastBackward(A,Y);
    }

    private double LossForward(INDArray A,INDArray Y){
        return lastLayer.getLossMethod().LossForward(A,Y);
    }


    private List<INDArray[]> backward(INDArray[] A_array,INDArray x,INDArray Y){
        INDArray DZ = LossBackward(A_array[A_array.length-1],Y);
        INDArray[] DW = new INDArray[A_array.length];
        INDArray[] DB = new INDArray[A_array.length];
        List<INDArray[]> res = new ArrayList<>();
        for(int i=A_array.length-1;i>=0;i--){
            if(i==0){ //最后一次
                INDArray dW = DZ.mmul(x.transpose());
                INDArray dB = DZ.mmul(Nd4j.ones(x.shape()[1],1));
                DW[i] = dW.div(x.shape()[1]);
                DB[i] = dB.div(x.shape()[1]);
            }else{
                INDArray dW = DZ.mmul(A_array[i-1].transpose());
                INDArray dB = DZ.mmul(Nd4j.ones(x.shape()[1],1));
                DW[i] = dW.div(x.shape()[1]);
                DB[i] = dB.div(x.shape()[1]);
                DZ = activate_backward(Network_W[i].transpose().mmul(DZ),A_array[i-1],layers.get(i-1).getActivateMethod());
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
            List<INData> batch_list = data.getBatchList();
            for(int j=0;j<batch_list.size();j++){
                INDArray[] A = forward(batch_list.get(j).getX()); //向前传播

                List<INDArray[]> DW_DB = backward(A, batch_list.get(j).getX(), batch_list.get(j).getY()); //反向传播

                update_parameters(DW_DB); //梯度下降

                double loss = LossForward(A[A.length - 1],batch_list.get(j).getY());

                //打印情况
                System.out.println("i=" + (i*batch_list.size()+j));
                System.out.println("loss=" + loss);
            }
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

    @Override
    public model setLearningrate(double rate) {
        this.learningrate = rate;
        return this;
    }

    @Override
    public model setIteration(int iteration) {
        this.iteration = iteration;
        return this;
    }
}
