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
        url: '/live-data_video',
        success: function(point) {
        console.log("786 correct point series",point,chart.series)
            var series = chart.series[0];
            chart.series[0].setData(point[0].data, true);
            // call it again after three seconds
            setTimeout(requestData, 100);
            console.log("786 correct after series",chart.series)
        },
        cache: false
    });
}

$(document).ready(function () {
    chart = new Highcharts.Chart({
        chart: {
            type: 'column',
            renderTo: 'data-container_video',
            events: {
                load: requestData
            }
        },
        title: {
            text: 'Video Recognition'
        },
        xAxis: {
            categories: ['angry', 'fear', 'happy', 'sad'],
            labels: {
                x: -10
            }
        },
        yAxis: {
            min: 0,
            gridLineWidth: 0,
            minorGridLineWidth: 0,

            title: {
                text: 'Percentage',
                y: 10
            },
            labels: {
                overflow: 'justify'
            }
        },
        series: [{
             data: requestData
         }]
    });
});
