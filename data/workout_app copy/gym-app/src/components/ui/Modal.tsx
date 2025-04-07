"use client";

import React, { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { FiX } from "react-icons/fi";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  maxWidth?: "sm" | "md" | "lg" | "xl" | "full";
  overlayOpacity?: number; // New prop to control overlay opacity
}

const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  maxWidth = "md",
  overlayOpacity = 30, // Default to 30% opacity
}) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const firstRender = useRef(true);

  // Close on escape key press
  useEffect(() => {
    const handleEscKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && isOpen) {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscKey);
    return () => {
      document.removeEventListener("keydown", handleEscKey);
    };
  }, [isOpen, onClose]);

  // Close when clicking outside the modal, but avoid closing on first render
  useEffect(() => {
    // Skip the first render to prevent immediate closure
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    
    const handleClickOutside = (event: MouseEvent) => {
      if (
        modalRef.current &&
        !modalRef.current.contains(event.target as Node) &&
        isOpen
      ) {
        onClose();
      }
    };

    // Use mousedown instead of click for more reliable behavior
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onClose]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  // Set max-width based on size prop
  const maxWidthClasses = {
    sm: "max-w-sm",
    md: "max-w-md",
    lg: "max-w-lg",
    xl: "max-w-xl",
    full: "max-w-full",
  };

  // Only create the portal if we're in a browser environment
  if (typeof document === "undefined") return null;

  // Prevent propagation of clicks from modal content
  const handleModalContentClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  // Create portal for modal
  const modalContent = (
    <div 
      style={{ backgroundColor: `rgba(0, 0, 0, ${overlayOpacity / 100})` }} 
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
    >
      <div
        ref={modalRef}
        onClick={handleModalContentClick}
        className={`w-full ${maxWidthClasses[maxWidth]} bg-white dark:bg-gray-800 rounded-lg shadow-xl overflow-hidden`}
      >
        <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold">{title || "Modal"}</h3>
          <button
            onClick={(e) => {
              e.stopPropagation(); 
              onClose();
            }}
            className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            type="button"
          >
            <FiX size={20} />
          </button>
        </div>
        <div className="p-4 max-h-[calc(100vh-12rem)] overflow-y-auto">
          {children}
        </div>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
};

export default Modal;
