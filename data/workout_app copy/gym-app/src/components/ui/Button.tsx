import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "success" | "danger" | "outline";
  size?: "sm" | "md" | "lg";
  fullWidth?: boolean;
  className?: string;
}

const Button: React.FC<ButtonProps> = ({
  children,
  variant = "primary",
  size = "md",
  fullWidth = false,
  className = "",
  ...props
}) => {
  // Base classes for all buttons
  const baseClasses = "font-medium rounded transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50";
  
  // Class variations for different button variants
  const variantClasses = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white",
    secondary: "bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200",
    success: "bg-green-600 hover:bg-green-700 text-white",
    danger: "bg-red-600 hover:bg-red-700 text-white",
    outline: "bg-transparent border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-800 dark:text-gray-200",
  };
  
  // Class variations for different button sizes
  const sizeClasses = {
    sm: "text-sm py-1 px-3",
    md: "py-2 px-4",
    lg: "text-lg py-2 px-5",
  };
  
  // Combine all classes
  const combinedClasses = `
    ${baseClasses} 
    ${variantClasses[variant]} 
    ${sizeClasses[size]} 
    ${fullWidth ? "w-full" : ""}
    ${props.disabled ? "opacity-50 cursor-not-allowed" : ""}
    ${className}
  `;
  
  return (
    <button className={combinedClasses} {...props}>
      {children}
    </button>
  );
};

export default Button;
