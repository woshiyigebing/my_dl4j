package com.yan.dl4j.controller;

import com.yan.dl4j.Repository.PointRepository;
import com.yan.dl4j.data.MyTrainData;
import com.yan.dl4j.data.TrainData;
import com.yan.dl4j.entity.point;
import com.yan.dl4j.model.LinearRegressionDL4J;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/example1")
public class exampleController1 {
    @Autowired
    private PointRepository pointRepository;


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

}
