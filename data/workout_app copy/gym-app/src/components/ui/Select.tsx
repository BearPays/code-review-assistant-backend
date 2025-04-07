import React from "react";

interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'options'> {
  label?: string;
  options: SelectOption[];
  error?: string;
  placeholder?: string;
  className?: string;
  fullWidth?: boolean;
}

const Select: React.FC<SelectProps> = ({
  label,
  options,
  error,
  placeholder,
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
      <select
        id={id}
        className={`px-3 py-2 rounded-md border bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100
          border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500
          ${error ? "border-red-500 focus:ring-red-500" : ""}
          ${fullWidth ? "w-full" : ""}
        `}
        {...props}
      >
        {placeholder && (
          <option value="">{placeholder}</option>
        )}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {error && (
        <p className="mt-1 text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
    </div>
  );
};

export default Select;
