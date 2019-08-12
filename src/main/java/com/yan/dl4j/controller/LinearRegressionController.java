package com.yan.dl4j.controller;

import com.yan.dl4j.Repository.PointRepository;
import com.yan.dl4j.data.MyTrainData;
import com.yan.dl4j.data.TrainData;
import com.yan.dl4j.entity.point;
import com.yan.dl4j.model.LinearRegressionDL4J;
import com.yan.dl4j.model.model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import java.util.List;

@Controller
@RequestMapping("/linear")
public class LinearRegressionController {

    @Autowired
    private PointRepository pointRepository;

    private LinearRegressionDL4J pointmodel;

    @GetMapping(value = "train")
    public String train(){
        if(pointmodel==null){
            pointmodel = new LinearRegressionDL4J();
            List<point> points = pointRepository.findAll();
            INDArray x = Nd4j.create(points.stream().mapToDouble(p->p.getX()).toArray(), new int[]{points.size(), 1});
            INDArray y = Nd4j.create(points.stream().mapToDouble(p->p.getY()).toArray(), new int[]{points.size(), 1});
            TrainData data = new MyTrainData(x,y);
            pointmodel.train(data);
        }
        return "freemarker/success";
    }

    @GetMapping(value = "show")
    public String show(ModelMap map) {
        List<point> points = pointRepository.findAll();
        if(pointmodel!=null){
            map.addAttribute ("K",pointmodel.getW());
            map.addAttribute ("B",pointmodel.getB());
            map.addAttribute ("points",points);
        }else{
            map.addAttribute ("K",1);
            map.addAttribute ("B",1);
            map.addAttribute ("points",points);
        }
        return"freemarker/linear/show";
    }


}
