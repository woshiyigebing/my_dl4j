package com.yan.dl4j.controller;

import com.yan.dl4j.Repository.Point1Repository;
import com.yan.dl4j.Repository.PointRepository;
import com.yan.dl4j.data.MyTrainData;
import com.yan.dl4j.data.TrainData;
import com.yan.dl4j.entity.point;
import com.yan.dl4j.entity.point1;
import com.yan.dl4j.model.LinearRegressionDL4J;
import com.yan.dl4j.model.NeuralNetwork;
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
        model nk = new NeuralNetwork();
        Collections.shuffle(points);
        //训练集90%
        for(int i=0;i<(int)Math.ceil(points.size()/10*9);i++){
            X1.add(points.get(i).getX());
            X1.add(points.get(i).getY());
            Y1.add(points.get(i).getZ());
        }
        //测试集10%
        for(int i=(int)Math.ceil(points.size()/10*9);i<points.size();i++){
            X2.add(points.get(i).getX());
            X2.add(points.get(i).getY());
            Y2.add(points.get(i).getZ());
        }
        INDArray X = Nd4j.create(X1).reshape(new int[]{points.size(), 2});
        INDArray Y = Nd4j.create(Y1).reshape(new int[]{points.size(), 1});
        TrainData data = new MyTrainData(X,Y);
        INDArray I_X2 = Nd4j.create(X2).reshape(new int[]{points.size(), 2});
        INDArray I_Y2 = Nd4j.create(Y2).reshape(new int[]{points.size(), 1});
        TrainData data2 = new MyTrainData(I_X2,I_Y2);
        nk.train(data);
        nk.verify(data2);
        return points;
    }

}
