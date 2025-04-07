import React from "react";

interface Set {
  id: string;
  reps: number;
  weight: number;
}

interface ExerciseItemProps {
  name: string;
  sets: Set[];
}

const ExerciseItem: React.FC<ExerciseItemProps> = ({ name, sets }) => {
  return (
    <div className="border border-gray-300 rounded p-4">
      <h3 className="font-medium mb-3">{name}</h3>

      <table className="w-full">
        <thead>
          <tr className="text-left text-sm">
            <th className="pb-2 w-16">Set</th>
            <th className="pb-2">Reps</th>
            <th className="pb-2">Weight</th>
          </tr>
        </thead>
        <tbody>
          {sets.map((set, index) => (
            <tr key={set.id}>
              <td className="py-2">{index + 1}</td>
              <td className="py-2">{set.reps}</td>
              <td className="py-2">{set.weight} kg</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ExerciseItem;
