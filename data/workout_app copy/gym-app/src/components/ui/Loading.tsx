import React from "react";

interface LoadingProps {
  text?: string;
}

const Loading: React.FC<LoadingProps> = ({ text = "Loading..." }) => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
      <p>{text}</p>
    </div>
  );
};

export default Loading;
