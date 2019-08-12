<html>
<head>
    <meta charset="utf-8">
    <title>ECharts</title>
    <!-- 引入 echarts.js -->
    <script src="../echarts.min.js"></script>
</head>
<body>
<!-- 为ECharts准备一个具备大小（宽高）的Dom -->
<div id="main" style="width: 100%;height:800px;"></div>
<script type="text/javascript">
    // 基于准备好的dom，初始化echarts实例
    var myChart = echarts.init(document.getElementById('main'));

    var dataAll = [
        [
            <#list all.points as point>
            [${point.x},${point.y},${point.z}],
            </#list>
        ],
        [
            <#list train.points as point>
            [${point.x},${point.y},${point.z}],
            </#list>
        ],
        [
            <#list test.points as point>
            [${point.x},${point.y},${point.z}],
            </#list>
        ],
        [
            <#list model.points as point>
            [${point.x},${point.y},${point.z}],
            </#list>
        ],
    ];

    // 指定图表的配置项和数据
    var option = {
        grid: [
            {x: '7%', y: '7%', width: '38%', height: '38%'},
            {x2: '7%', y: '7%', width: '38%', height: '38%'},
            {x: '7%', y2: '7%', width: '38%', height: '38%'},
            {x2: '7%', y2: '7%', width: '38%', height: '38%'}
        ],
        xAxis: [{gridIndex: 0,min: ${all.XMin}, max: ${all.XMax}},
                {gridIndex: 1, min: ${train.XMin}, max: ${train.XMax}},
                {gridIndex: 2, min: ${test.XMin}, max: ${test.XMax}},
                {gridIndex: 3, min: ${model.XMin}, max: ${model.XMax}}],
        yAxis: [{gridIndex: 0,min: ${all.YMin}, max: ${all.YMax}},
            {gridIndex: 1,min: ${train.YMin}, max: ${train.YMax}},
            {gridIndex: 2,min: ${test.YMin}, max: ${test.YMax}},
            {gridIndex: 3,min: ${model.YMin}, max: ${model.YMax}}
        ],
        series: [{
            name: '全部数据',
            symbolSize: 5,
            type: 'scatter',
            xAxisIndex: 0,
            yAxisIndex: 0,
            data: dataAll[0],
            itemStyle:{
                normal:{
                    color:function(p){
                        if(p.data[2]==0){
                            return"#FE8463";
                        }else{
                            return "#27727B";
                        }
                    }
                }
            }
        },
            {
                name: '训练数据',
                type: 'scatter',
                symbolSize: 5,
                data: dataAll[1],
                xAxisIndex: 1,
                yAxisIndex: 1,
                itemStyle:{
                    normal:{
                        color:function(p){
                            if(p.data[2]==0){
                                return"#FE8463";
                            }else{
                                return "#27727B";
                            }
                        }
                    }
                }
            },
            {
                name: '测试数据',
                type: 'scatter',
                symbolSize: 5,
                data: dataAll[2],
                xAxisIndex: 2,
                yAxisIndex: 2,
                itemStyle:{
                    normal:{
                        color:function(p){
                            if(p.data[2]==0){
                                return"#FE8463";
                            }else{
                                return "#27727B";
                            }
                        }
                    }
                }
            },
            {
                name: '模型数据',
                type: 'scatter',
                symbolSize: 5,
                data: dataAll[3],
                xAxisIndex: 3,
                yAxisIndex: 3,
                itemStyle:{
                    normal:{
                        color:function(p){
                            if(p.data[2]==0){
                                return"#FE8463";
                            }else{
                                return "#27727B";
                            }
                        }
                    }
                }
            },
            {
                name: '模型数据',
                type: 'scatter',
                symbolSize: 5,
                data: dataAll[1],
                xAxisIndex: 3,
                yAxisIndex: 3,
                itemStyle:{
                    normal:{
                        color:function(p){
                            if(p.data[2]==0){
                                return"#FE8463";
                            }else{
                                return "#27727B";
                            }
                        }
                    }
                }
            },
        ]
    };

    // 使用刚指定的配置项和数据显示图表。
    myChart.setOption(option);
</script>
<#list all.points as point>
    [${point.x},${point.y}]
</#list>
</body>
</html>