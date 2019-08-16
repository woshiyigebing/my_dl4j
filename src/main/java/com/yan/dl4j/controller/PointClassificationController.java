package com.yan.dl4j.controller;

import com.yan.dl4j.Repository.Point1Repository;
import com.yan.dl4j.data.MyTrainData;
import com.yan.dl4j.data.TrainData;
import com.yan.dl4j.entity.point;
import com.yan.dl4j.entity.point1;
import com.yan.dl4j.model.Activate.Sigmoid;
import com.yan.dl4j.model.Activate.Tanh;
import com.yan.dl4j.model.Layer.MyLastLayer;
import com.yan.dl4j.model.Layer.MyLayer;
import com.yan.dl4j.model.LinearRegressionDL4J;
import com.yan.dl4j.model.Loss.LogLoss;
import com.yan.dl4j.model.Loss.MSE;
import com.yan.dl4j.model.NetWork.DeepNeuralNetWork;
import com.yan.dl4j.model.NeuralNetwork;
import com.yan.dl4j.model.model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import java.util.*;

@Controller
@RequestMapping("/point")
public class PointClassificationController {
    @Autowired
    private Point1Repository point1Repository;

    private List<point1> train_point;
    private List<point1> test_point;
    private List<point1> all_point;
    private List<point1> model_point;

    //private model pointmodel = new NeuralNetwork();
    private model pointmodel = new DeepNeuralNetWork(2)
            .addLastLayer(new MyLastLayer(1,new Sigmoid(),new LogLoss())).setIteration(1000).setLearningrate(0.1);

    @GetMapping(value = "show")
    public String show(ModelMap map) {
        if(all_point==null){
            all_point = point1Repository.findAll();
        }
        if(test_point==null||train_point==null){
            int plot = (int)Math.ceil(all_point.size()/10*8);
            test_point = all_point.subList(plot,all_point.size());
            train_point = all_point.subList(0,plot);
        }
        Map a_p = getMap(all_point);
        model_point = model_point((double)a_p.get("XMax"),(double)a_p.get("XMin"),(double)a_p.get("YMax"),(double)a_p.get("YMin"),0.05);
        map.addAttribute ("all",a_p);
        map.addAttribute ("train",getMap(train_point));
        map.addAttribute ("test",getMap(test_point));
        map.addAttribute ("model",getMap(model_point));
        return"freemarker/point/show";
    }

    @GetMapping(value = "cut")
    public String cut(){
        Collections.shuffle(all_point);
        int plot = (int)Math.ceil(all_point.size()/10*8);
        train_point = all_point.subList(0,plot);
        test_point = all_point.subList(plot,all_point.size());
        return "freemarker/success";
    }


    private List<point1> model_point(double XMax,double XMin,double YMax,double YMmin,double step){
        List<point1> point1s = new ArrayList<>();
        List<Double> X1 = new ArrayList<>();
        double x = XMin-step;
        while(x<=XMax+0.5){
            double y = YMmin -step;
            while(y<=YMax+0.5){
                point1 point1 = new point1();
                point1.setX(x);
                point1.setY(y);
                X1.add(x);
                X1.add(y);
                point1.setZ(0);
                point1s.add(point1);
                y = y+step;
            }
            x=x+step;
        }
        INDArray X = Nd4j.create(X1).reshape(new int[]{point1s.size(), 2});
        INDArray Y = pointmodel.predict(X.transpose());
        double[] Ys = Y.toDoubleVector();
        for(int i=0;i<point1s.size();i++){
            if(Ys[i]>0.5){
                point1s.get(i).setZ(1);
            }else{
                point1s.get(i).setZ(0);
            }
        }
        return point1s;
    }

    private Map getMap(List<point1> points){
        Map<String,Object> map = new HashMap<>();
        if(points!=null&&points.size()>1){
            double XMin= points.get(0).getX();
            double XMax= points.get(0).getX();
            double YMin = points.get(0).getY();
            double YMax = points.get(0).getY();
            for(point1 point:points){
                XMin = Math.min(point.getX(),XMin);
                XMax = Math.max(point.getX(),XMax);
                YMin = Math.min(point.getY(),YMin);
                YMax = Math.max(point.getY(),YMax);
            }
            map.put("points",points);
            map.put("XMin",XMin);
            map.put("XMax",XMax);
            map.put("YMin",YMin);
            map.put("YMax",YMax);
        }
        return map;
    }

    @GetMapping(value = "train")
    public String train(){
        if(train_point!=null&&test_point!=null){
            List<Double> X1 = new ArrayList<>();
            List<Integer> Y1 = new ArrayList<>();
            List<Double> X2 = new ArrayList<>();
            List<Integer> Y2 = new ArrayList<>();

            for(point1 point:train_point){
                X1.add(point.getX());
                X1.add(point.getY());
                Y1.add(point.getZ());
            }

            for(point1 point:test_point){
                X2.add(point.getX());
                X2.add(point.getY());
                Y2.add(point.getZ());
            }
            INDArray X = Nd4j.create(X1).reshape(new int[]{train_point.size(), 2});
            INDArray Y = Nd4j.create(Y1).reshape(new int[]{train_point.size(), 1});
            TrainData data = new MyTrainData(X,Y);
            INDArray I_X2 = Nd4j.create(X2).reshape(new int[]{test_point.size(), 2});
            INDArray I_Y2 = Nd4j.create(Y2).reshape(new int[]{test_point.size(), 1});
            TrainData data2 = new MyTrainData(I_X2,I_Y2);
            pointmodel.train(data);

            //验证集验证效果
            INDArray p_Y = pointmodel.predict(data2.getX());
            System.out.println(p_Y);
            System.out.println(data2.getY());
            System.out.println(scord(p_Y,data2.getY()));
            return "freemarker/success";
        }
        return "freemarker/fail";
    }

    private float scord(INDArray value,INDArray Y) {
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
