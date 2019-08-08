package com.yan.dl4j.controller;

import com.yan.dl4j.Repository.Point1Repository;
import com.yan.dl4j.Repository.PointRepository;
import com.yan.dl4j.data.MyTrainData;
import com.yan.dl4j.data.TrainData;
import com.yan.dl4j.entity.point;
import com.yan.dl4j.entity.point1;
import com.yan.dl4j.model.Activate.Sigmoid;
import com.yan.dl4j.model.Activate.Tanh;
import com.yan.dl4j.model.Layer.MyLastLayer;
import com.yan.dl4j.model.Layer.MyLayer;
import com.yan.dl4j.model.LinearRegressionDL4J;
import com.yan.dl4j.model.Loss.MSE;
import com.yan.dl4j.model.NetWork.DeepNeuralNetWork;
import com.yan.dl4j.model.model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@RestController
@RequestMapping("/example1")
public class exampleController1 {
    @Autowired
    private PointRepository pointRepository;
    @Autowired
    private Point1Repository point1Repository;

    @GetMapping(value = "train")
    public String train(){
        LinearRegressionDL4J linearRegressionDL4J = new LinearRegressionDL4J();
        List<point> points = pointRepository.findAll();
        INDArray x = Nd4j.create(points.stream().mapToDouble(p->p.getX()).toArray(), new int[]{points.size(), 1});
        INDArray y = Nd4j.create(points.stream().mapToDouble(p->p.getY()).toArray(), new int[]{points.size(), 1});
        TrainData data = new MyTrainData(x,y);
        linearRegressionDL4J.train(data);
        return "success";
    }

    @GetMapping(value = "print")
    public List<point1> print() {
        List<point1> points = point1Repository.findAll();
        List<Double> X1 = new ArrayList<>();
        List<Integer> Y1 = new ArrayList<>();
        List<Double> X2 = new ArrayList<>();
        List<Integer> Y2 = new ArrayList<>();
        //model nk = new NeuralNetwork();
//        List<Integer> LARYER = new ArrayList<>();
//        LARYER.add(2);
//        LARYER.add(4);
//        LARYER.add(1);
//        model nk = new DeepNeuralNetWork(LARYER,"MSE");
        model nk = new DeepNeuralNetWork(2)
                .addLayer(new MyLayer(10,new Tanh()))
                .addLastLayer(new MyLastLayer(1,new Sigmoid(),new MSE())).setIteration(10000).setLearningrate(0.1);
        Collections.shuffle(points);
        int plot = (int)Math.ceil(points.size()/10*8);
        //训练集80%
        for(int i=0;i<plot;i++){
            X1.add(points.get(i).getX());
            X1.add(points.get(i).getY());
            Y1.add(points.get(i).getZ());
        }
        //测试集20%
        for(int i=plot;i<points.size();i++){
            X2.add(points.get(i).getX());
            X2.add(points.get(i).getY());
            Y2.add(points.get(i).getZ());
        }
        INDArray X = Nd4j.create(X1).reshape(new int[]{plot, 2});
        INDArray Y = Nd4j.create(Y1).reshape(new int[]{plot, 1});
        TrainData data = new MyTrainData(X,Y);
        INDArray I_X2 = Nd4j.create(X2).reshape(new int[]{points.size()-plot, 2});
        INDArray I_Y2 = Nd4j.create(Y2).reshape(new int[]{points.size()-plot, 1});
        TrainData data2 = new MyTrainData(I_X2,I_Y2);
        nk.train(data);

        //验证集验证效果
        INDArray p_Y = nk.predict(data2.getX());
        System.out.println(p_Y);
        System.out.println(data2.getY());
        System.out.println(scord(p_Y,data2.getY()));
        return points;
    }

    public float scord(INDArray value,INDArray Y) {
        int res = 0;
        int sum = 0;
        if(value.shape()[0]>1){
            double[][] s = value.toDoubleMatrix();
            double[][] Ys = Y.toDoubleMatrix();
            for(int i=0;i<s.length;i++){
                for(int j =0;j<s[i].length;j++){
                    if(s[i][j]>0.5&&Ys[i][j]>0.5){
                        res++;
                    }else if(s[i][j]<0.5&&Ys[i][j]<0.5){
                        res++;
                    }
                    sum++;
                }
            }
            if(sum>0){
                return ((float)res/sum)*100;
            }else{
                return 0;
            }
        }else{
            double[] s = value.toDoubleVector();
            double[] Ys = Y.toDoubleVector();
            for(int i=0;i<s.length;i++){
                if(s[i]>0.5&&Ys[i]>0.5){
                    res++;
                }else if(s[i]<0.5&&Ys[i]<0.5){
                    res++;
                }
                sum++;
            }
            if(sum>0){
                return ((float)res/sum)*100;
            }else{
                return 0;
            }
        }
    }
}
