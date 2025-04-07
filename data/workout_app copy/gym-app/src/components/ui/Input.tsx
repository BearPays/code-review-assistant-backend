import React from "react";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  className?: string;
  fullWidth?: boolean;
}

const Input: React.FC<InputProps> = ({
  label,
  error,
  className = "",
  fullWidth = true,
  id,
  ...props
}) => {
  return (
    <div className={`${fullWidth ? "w-full" : ""} ${className}`}>
      {label && (
        <label 
          htmlFor={id} 
          className="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-200"
        >
          {label}
        </label>
      )}
      <input
        id={id}
        className={`px-3 py-2 rounded-md border bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 
          border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500
          ${error ? "border-red-500 focus:ring-red-500" : ""}
          ${fullWidth ? "w-full" : ""}
        `}
        {...props}
      />
      {error && (
        <p className="mt-1 text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
    </div>
  );
};

export default Input;
