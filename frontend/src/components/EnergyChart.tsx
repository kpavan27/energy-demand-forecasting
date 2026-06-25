import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ChartDataPoint {
  name: string;
  actual: number | null;
  predicted: number | null;
}

interface EnergyChartProps {
  data: ChartDataPoint[];
}

const EnergyChart = ({ data }: EnergyChartProps) => {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={data}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="actual" stroke="#8884d8" activeDot={{ r: 8 }} />
        <Line type="monotone" dataKey="predicted" stroke="#82ca9d" />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default EnergyChart;
