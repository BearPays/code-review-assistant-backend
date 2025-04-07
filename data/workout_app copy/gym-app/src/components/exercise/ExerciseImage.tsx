import Image from 'next/image';
import { useState } from 'react';

interface ExerciseImageProps {
  imageUrl?: string;
  exerciseName: string;
  className?: string;
  width?: number;
  height?: number;
  objectFit?: 'cover' | 'contain' | 'fill';
}

const ExerciseImage: React.FC<ExerciseImageProps> = ({
  imageUrl,
  exerciseName,
  className = '',
  width,
  height,
  objectFit = 'contain',
}) => {
  const [isError, setIsError] = useState(false);

  if (!imageUrl || isError) {
    return (
      <div className={`bg-gray-200 dark:bg-gray-700 flex items-center justify-center ${className}`}>
        <span className="text-gray-500 dark:text-gray-400 text-center p-4">
          No image available
        </span>
      </div>
    );
  }

  // Check if the image URL is absolute or relative
  const isAbsoluteUrl = imageUrl.startsWith('http');

  const imageProps = {
    src: imageUrl,
    alt: exerciseName || "Exercise image", // Add alt text (fixes ESLint warning)
    fill: !width && !height,
    width: width,
    height: height,
    className: `${objectFit === 'contain' ? 'object-contain' : objectFit === 'cover' ? 'object-cover' : ''} rounded-lg`,
    unoptimized: isAbsoluteUrl,
    onError: () => setIsError(true)
  };

  return (
    <div className={`relative ${!width && !height ? 'h-full w-full' : ''} ${className}`}>
      <Image {...imageProps} alt={exerciseName || "Exercise image"} />
    </div>
  );
};

export default ExerciseImage;
