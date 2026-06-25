

interface InfoCardProps {
  title: string;
  value: string | number;
  unit?: string;
  colorClass?: string;
}

const InfoCard = ({ title, value, unit, colorClass = 'text-green-400' }: InfoCardProps) => {
  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold text-gray-400 mb-4">{title}</h2>
      <p className={`text-4xl font-bold ${colorClass}`}>
        {value} <span className="text-2xl">{unit}</span>
      </p>
    </div>
  );
};

export default InfoCard;
