var chart;

/**
 * Request data from the server, add it to the graph and set a timeout
 * to request again
 */
function requestData() {
    $.ajaxSetup({
    async: true
    });

    $.ajax({
        url: '/live-data_multi',
        success: function(point) {
            var series = chart.series[0];
            // chart.series[0].setData(point, true);
            chart.series=[];
			  point.forEach((j, i) => {
					chart.addSeries(j);
			  });
            // call it again after three seconds
            setTimeout(requestData, 5000);
        },
        cache: false
    });
}

$(document).ready(function () {
    chart = new Highcharts.Chart({
        chart: {
            type: 'packedbubble',
            renderTo: 'data-container_multi',
            events: {
                load: requestData
            }
        },
        title: {
            text: 'Multimodal Emotion Recognition'
        },
        tooltip: {
            useHTML: true,
            pointFormat: '<b>{point.name}:</b> {point.value}%'
        },
        plotOptions: {
            packedbubble: {
                minSize: '20%',
                maxSize: '150%',
                zMin: 0,
                zMax: 1000,
                layoutAlgorithm: {
                    gravitationalConstant: 0.05,
                    splitSeries: true,
                    seriesInteraction: false,
                    dragBetweenSeries: true,
                    parentNodeLimit: true
                },
                dataLabels: {
                    enabled: true,
                    format: '{point.name}',
                    filter: {
                        property: 'y',
                        operator: '>',
                        value: 0
                    },
                    style: {
                        color: 'black',
                        textOutline: 'none',
                        fontWeight: 'normal'
                    }
                }
            }
        },
        series: [],

    });
}

);