import { useEffect, useState } from 'react';

export interface MobileLayoutState {
  width: number;
  height: number;
  isTouchDevice: boolean;
  isPortrait: boolean;
  isMobileLandscape: boolean;
  isCompactMobile: boolean;
}

function readViewportState(): MobileLayoutState {
  if (typeof window === 'undefined') {
    return {
      width: 1280,
      height: 800,
      isTouchDevice: false,
      isPortrait: false,
      isMobileLandscape: false,
      isCompactMobile: false,
    };
  }

  const width = window.innerWidth;
  const height = window.innerHeight;
  const isTouchDevice =
    'ontouchstart' in window ||
    (typeof navigator !== 'undefined' && navigator.maxTouchPoints > 0) ||
    (typeof window.matchMedia === 'function' &&
      window.matchMedia('(pointer: coarse)').matches);

  const isLandscape = width > height;
  const isPortrait = width < 768 && height > width;
  const isMobileLandscape =
    isLandscape && (height <= 560 || (width <= 960 && isTouchDevice));
  const isCompactMobile = isMobileLandscape || width < 768;

  return {
    width,
    height,
    isTouchDevice,
    isPortrait,
    isMobileLandscape,
    isCompactMobile,
  };
}

export function useMobileLayout(): MobileLayoutState {
  const [layout, setLayout] = useState<MobileLayoutState>(() =>
    readViewportState()
  );

  useEffect(() => {
    const update = () => {
      setLayout(readViewportState());
    };
    window.addEventListener('resize', update);
    window.addEventListener('orientationchange', update);
    return () => {
      window.removeEventListener('resize', update);
      window.removeEventListener('orientationchange', update);
    };
  }, []);

  return layout;
}
