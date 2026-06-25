

const GaugeChart = ({ value, min, max }: { value: number; min: number; max: number }) => {
  const percentage = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  const strokeWidth = 12;
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="relative flex items-center justify-center">
      <svg width="200" height="120" viewBox="0 0 200 120">
        {/* Background Arc */}
        <path
          d={`M ${100 - radius},100 A ${radius},${radius} 0 0 1 ${100 + radius},100`}
          fill="none"
          stroke="#4A5568" // gray-700
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Foreground Arc */}
        <path
          d={`M ${100 - radius},100 A ${radius},${radius} 0 0 1 ${100 + radius},100`}
          fill="none"
          stroke="#48BB78" // green-500
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{ transition: 'stroke-dashoffset 0.5s ease-in-out' }}
        />
      </svg>
      <div className="absolute bottom-0 text-center">
        <span className="text-3xl font-bold">{value.toFixed(2)}</span>
        <span className="text-sm text-gray-400">kW</span>
      </div>
    </div>
  );
};

export default GaugeChart;
