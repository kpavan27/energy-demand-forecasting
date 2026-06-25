import { useState, useEffect, useMemo } from 'react';
import EnergyChart from './components/EnergyChart';
import GaugeChart from './components/GaugeChart';
import InfoCard from './components/InfoCard';

interface ChartDataPoint {
  name: string;
  actual: number | null;
  predicted: number | null;
}

function App() {
  const [data, setData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Memoize statistics calculation
  const stats = useMemo(() => {
    const validData = data.map(d => d.actual).filter(d => d !== null) as number[];
    if (validData.length === 0) {
      return {
        current: 0,
        avg: 0,
        peak: 0,
        low: 0,
      };
    }
    const current = validData[validData.length - 1];
    const avg = validData.reduce((sum, val) => sum + val, 0) / validData.length;
    const peak = Math.max(...validData);
    const low = Math.min(...validData);
    return { current, avg, peak, low };
  }, [data]);

  // Fetch initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const response = await fetch('/get_initial_data');
        const result = await response.json();

        if (result.error) throw new Error(result.error);

        const initialChartData: ChartDataPoint[] = result.initial_data.map((value: number, index: number) => ({
          name: `T-${23 - index}`,
          actual: value,
          predicted: null,
        }));

        setData(initialChartData);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, []);

  // Set up live update interval
  useEffect(() => {
    const interval = setInterval(() => {
      if (data.length > 0) {
        updateData();
      }
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [data]);

  const updateData = async () => {
    const sequence = data.slice(-24).map(d => d.actual!);
    
    try {
      const [predictResponse, actualResponse] = await Promise.all([
        fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sequence }),
        }),
        fetch('/get_next_actual')
      ]);

      const predictResult = await predictResponse.json();
      const actualResult = await actualResponse.json();

      if (predictResult.error) throw new Error(predictResult.error);
      if (actualResult.error) throw new Error(actualResult.error);

      const newPoint: ChartDataPoint = {
        name: `T+${data.length - 23}`,
        actual: actualResult.next_actual,
        predicted: predictResult.prediction,
      };

      setData(prevData => [...prevData.slice(1), newPoint]);

    } catch (err: any) {
      // Don't overwrite important initial load errors
      if (!error) setError(err.message);
      console.error(err);
    }
  };

  return (
    <div className="bg-gray-900 text-white min-h-screen p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold mb-2">Energy Demand Dashboard</h1>
          <p className="text-gray-400">Live forecasting with an LSTM Neural Network</p>
        </header>

        <main>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <div className="lg:col-span-2 bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-2xl font-semibold mb-4">Consumption History (Last 24 hours)</h2>
              {loading && <p>Loading chart data...</p>}
              {error && <p className="text-red-500">Error: {error}</p>}
              {data.length > 0 && <EnergyChart data={data} />}
            </div>
            <div className="flex flex-col justify-center items-center bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-2xl font-semibold mb-4">Current Demand</h2>
              <GaugeChart value={stats.current} min={0} max={Math.max(8, stats.peak)} />
            </div>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
            <InfoCard title="Last Prediction" value={data[data.length - 1]?.predicted?.toFixed(2) ?? '...'} unit="kW" colorClass="text-green-400" />
            <InfoCard title="Average" value={stats.avg.toFixed(2)} unit="kW" colorClass="text-blue-400" />
            <InfoCard title="Peak" value={stats.peak.toFixed(2)} unit="kW" colorClass="text-yellow-400" />
            <InfoCard title="Lowest" value={stats.low.toFixed(2)} unit="kW" colorClass="text-purple-400" />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;