<html>
<head>
    <meta charset="utf-8">
    <title>ECharts</title>
    <!-- 引入 echarts.js -->
    <script src="../echarts.min.js"></script>
</head>
<body>
<!-- 为ECharts准备一个具备大小（宽高）的Dom -->
<div id="main" style="width: 600px;height:400px;"></div>
<script type="text/javascript">
    // 基于准备好的dom，初始化echarts实例
    var myChart = echarts.init(document.getElementById('main'));
    var markLineOpt = {
        animation: false,
        label: {
            normal: {
                formatter: 'y = ${K} * x + ${B}',
                textStyle: {
                    align: 'right'
                }
            }
        },
        lineStyle: {
            normal: {
                type: 'solid'
            }
        },
        tooltip: {
            formatter: 'y = ${K} * x + ${B}'
        },
        data: [[{
            coord: [0, ${B}],
            symbol: 'none'
        }, {
            coord: [10, ${K}*10+${B}],
            symbol: 'none'
        }]]
    };

    // 指定图表的配置项和数据
    var option = {
        xAxis: {gridIndex: 0,min: 0, max: 10},
        yAxis: {gridIndex: 0,min: 0, max: 20},
        series: [{
            type: 'scatter',
            symbolSize: 10,
            data: [
                <#list points as point>
                [${point.x},${point.y}],
                </#list>
            ],
            markLine: markLineOpt
        }]
    };

    // 使用刚指定的配置项和数据显示图表。
    myChart.setOption(option);
</script>
Y = ${K}x+${B}
point:
<#list points as point>
    [${point.x},${point.y}]
</#list>
</body>
</html>