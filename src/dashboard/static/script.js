
    document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('energy-chart').getContext('2d');
    let energyChart;
    let chartData = {
        labels: [],
        datasets: [
            {
                label: 'Actual Load (kW)',
                data: [],
                borderColor: 'rgba(139, 148, 158, 0.8)',
                backgroundColor: 'rgba(139, 148, 158, 0.1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(139, 148, 158, 1)',
                pointRadius: 2,
                fill: true,
                tension: 0.4
            },
            {
                label: 'LSTM Forecast (kW)',
                data: [],
                borderColor: 'rgba(88, 166, 255, 1)',
                backgroundColor: 'rgba(88, 166, 255, 0.15)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(88, 166, 255, 1)',
                pointRadius: 2,
                borderDash: [5, 5],
                fill: true,
                tension: 0.4
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                labels: { color: '#c9d1d9', font: { family: "'Inter', sans-serif", size: 13 } }
            },
            tooltip: {
                backgroundColor: 'rgba(22, 27, 34, 0.9)',
                titleColor: '#fff',
                bodyColor: '#c9d1d9',
                borderColor: '#30363d',
                borderWidth: 1,
                padding: 12,
                boxPadding: 6,
                usePointStyle: true
            }
        },
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'hour',
                    displayFormats: { hour: 'HH:mm' },
                    tooltipFormat: 'yyyy-MM-dd HH:mm'
                },
                grid: { color: 'rgba(48, 54, 61, 0.5)', drawBorder: false },
                ticks: { color: '#8b949e', font: { family: "'Inter', sans-serif" } },
                title: { display: true, text: 'Time', color: '#8b949e' }
            },
            y: {
                grid: { color: 'rgba(48, 54, 61, 0.5)', drawBorder: false },
                ticks: { color: '#8b949e', font: { family: "'Inter', sans-serif" } },
                title: { display: true, text: 'Global Active Power (kW)', color: '#8b949e' },
                beginAtZero: false
            }
        }
    };

    energyChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: chartOptions,
    });

    async function getInitialData() {
        try {
            const response = await fetch('/get_initial_data');
            const data = await response.json();
            const now = new Date();

            chartData.labels = Array.from({ length: 24 }, (_, i) => {
                const date = new Date(now);
                date.setHours(now.getHours() - (23 - i));
                return date;
            });

            chartData.datasets[0].data = data.initial_data;
            // Initialize forecast data with nulls
            chartData.datasets[1].data = new Array(data.initial_data.length).fill(null);
            energyChart.update();
            // Start the prediction loop after initial data is loaded
            setInterval(makePrediction, 5000); // Predict every 5 seconds
        } catch (error) {
            console.error('Error fetching initial data:', error);
        }
    }

    async function makePrediction() {
        const sequence = chartData.datasets[0].data.slice(-24);
        try {
            const [predictResponse, actualResponse] = await Promise.all([
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ sequence }),
                }),
                fetch('/get_next_actual')
            ]);

            const predictResult = await predictResponse.json();
            const actualResult = await actualResponse.json();

            if (predictResult.prediction && actualResult.next_actual) {
                const prediction = predictResult.prediction;
                const next_actual = actualResult.next_actual;

                document.getElementById('prediction-output').innerText = `Forecasted Demand: ${prediction.toFixed(2)} kW`;

                // Update the chart
                const now = new Date();
                chartData.labels.push(now);
                chartData.datasets[0].data.push(next_actual);
                chartData.datasets[1].data.push(prediction);

                // To keep the chart from getting too crowded, we'll shift the data
                if (chartData.labels.length > 50) {
                    chartData.labels.shift();
                    chartData.datasets[0].data.shift();
                    chartData.datasets[1].data.shift();
                }

                energyChart.update();
            } else {
                console.error('Prediction error or no more actual data:', predictResult.error || actualResult.error);
            }
        } catch (error) {
            console.error('Error making prediction:', error);
        }
    }

    getInitialData();
});
    