import React, { Component, ErrorInfo, ReactNode, useState, useEffect, useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

// 1. Programmatic WebGL Support Check
const checkWebGLSupport = (): boolean => {
  try {
    const canvas = document.createElement("canvas");
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
    );
  } catch (e) {
    return false;
  }
};

// 2. React Error Boundary for safe Canvas rendering
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

class CanvasErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public state: ErrorBoundaryState = {
    hasError: false
  };

  public static getDerivedStateFromError(_: Error): ErrorBoundaryState {
    return { hasError: true };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.warn("WebGL/Canvas rendering crash prevented gracefully:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

// 3. Waving Monochromatic Neural Wireframe Grid
const ParticleNet = () => {
  const pointsRef = useRef<THREE.Points>(null);
  
  // Subtle particle grid
  const countX = 40;
  const countY = 40;
  const numParticles = countX * countY;

  // Initialize flat grid position array
  const positions = useMemo(() => {
    const pos = new Float32Array(numParticles * 3);
    let i = 0;
    const gap = 0.5; // Grid spacing

    for (let x = 0; x < countX; x++) {
      for (let y = 0; y < countY; y++) {
        pos[i] = (x - countX / 2) * gap;     // X coordinate
        pos[i + 1] = 0;                      // Y coordinate (height mapped dynamically)
        pos[i + 2] = (y - countY / 2) * gap; // Z coordinate
        i += 3;
      }
    }
    return pos;
  }, [numParticles]);

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geom;
  }, [positions]);

  useFrame((state) => {
    if (!pointsRef.current) return;
    const time = state.clock.getElapsedTime();

    // Subtle breathing rotation
    pointsRef.current.rotation.y = time * 0.008;
    pointsRef.current.rotation.x = 0.35 + Math.sin(time * 0.04) * 0.03; // Fixed incline with breathing sway

    const posAttr = pointsRef.current.geometry.attributes.position;
    const array = posAttr.array as Float32Array;

    // Apply compound trigonometric waves
    let i = 0;
    for (let x = 0; x < countX; x++) {
      for (let y = 0; y < countY; y++) {
        const xPos = array[i];
        const zPos = array[i + 2];

        // Complex sine combinations simulating multi-frequency scan waves
        array[i + 1] = 
          Math.sin(xPos * 0.3 + time * 0.35) * 0.3 +
          Math.cos(zPos * 0.3 + time * 0.28) * 0.3 +
          Math.sin((xPos + zPos) * 0.1 + time * 0.5) * 0.15;

        i += 3;
      }
    }
    posAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        size={0.075}
        color="#2D6A4F" // Sage green matching palette
        transparent
        opacity={0.16}  // Low opacity to keep it extremely subtle
        sizeAttenuation
      />
    </points>
  );
};

// 4. Main Component: Fixed Canvas Void
const HeroCanvas = () => {
  const [isSupported, setIsSupported] = useState<boolean | null>(null);

  useEffect(() => {
    setIsSupported(checkWebGLSupport());
  }, []);

  // Warm off-white fallback backdrop
  const FallbackVoid = (
    <div className="fixed inset-0 z-[-1] pointer-events-none bg-white">
      <div className="absolute inset-0" style={{
        background: "radial-gradient(circle at center, rgba(45, 106, 79, 0.04) 0%, rgba(247, 246, 243, 0.95) 80%)"
      }} />
    </div>
  );

  if (isSupported === null) {
    return <div className="fixed inset-0 z-[-1] bg-white" />;
  }

  if (!isSupported) {
    return FallbackVoid;
  }

  return (
    <div className="fixed inset-0 z-[-1] pointer-events-none bg-gradient-to-b from-[#FEFDFB] via-[#F7F6F3] to-[#FEFDFB]">
      <div className="absolute inset-0 bg-radial-gradient z-[1] opacity-50" style={{
        background: "radial-gradient(circle at center, rgba(255, 255, 255, 0.1) 0%, rgba(250, 250, 250, 0.7) 100%)"
      }} />
      <CanvasErrorBoundary fallback={FallbackVoid}>
        <Canvas camera={{ position: [0, 5, 12], fov: 45 }}>
          <ambientLight intensity={0.55} />
          <pointLight position={[10, 20, 10]} intensity={0.45} color="#FFE082" />
          <ParticleNet />
        </Canvas>
      </CanvasErrorBoundary>
    </div>
  );
};

export default HeroCanvas;
