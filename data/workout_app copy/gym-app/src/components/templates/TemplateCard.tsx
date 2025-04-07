import React from "react";
import Link from "next/link";

interface TemplateCardProps {
  id: string;
  name: string;
  createdAt: string;
}

const TemplateCard: React.FC<TemplateCardProps> = ({ id, name, createdAt }) => {
  return (
    <Link href={`/templates/${id}`} passHref>
      <div className="border border-gray-300 dark:border-gray-700 p-4 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800">
        <h2 className="font-semibold text-lg">{name}</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Created {new Date(createdAt).toLocaleDateString()}
        </p>
      </div>
    </Link>
  );
};

export default TemplateCard;
