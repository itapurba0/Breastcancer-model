import React, { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

type HelixProps = {
    strands?: number;
    pointsPerStrand?: number;
    color?: string;
    radius?: number;
    height?: number;
};

function Helix({
    strands = 4,
    pointsPerStrand = 180,
    color = "#78909C",
    radius = 6,
    height = 18,
}: HelixProps) {
    const group = useRef<THREE.Group>(null!);

    const lineGeometries = useMemo(() => {
        const geoms: THREE.BufferGeometry[] = [];

        for (let s = 0; s < strands; s++) {
            const positions = new Float32Array(pointsPerStrand * 3);
            const phase = (s / strands) * Math.PI * 2;
            const strandRadius = radius * (1 + (s - strands / 2) * 0.08);

            for (let i = 0; i < pointsPerStrand; i++) {
                const t = i / (pointsPerStrand - 1);
                const angle = t * Math.PI * 4 + phase; // multiple turns
                const x = Math.cos(angle) * strandRadius * (1 + Math.sin(t * Math.PI) * 0.08);
                const y = (t - 0.5) * height;
                const z = Math.sin(angle) * strandRadius * (1 + Math.cos(t * Math.PI) * 0.08);

                const idx = i * 3;
                positions[idx + 0] = x;
                positions[idx + 1] = y;
                positions[idx + 2] = z;
            }

            const geom = new THREE.BufferGeometry();
            geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            geoms.push(geom);
        }

        return geoms;
    }, [strands, pointsPerStrand, radius, height]);

    useFrame(({ clock }) => {
        const t = clock.getElapsedTime() * 0.08;
        if (!group.current) return;
        group.current.rotation.y = t * 0.6;
        group.current.position.y = Math.sin(t * 0.6) * 0.3;
    });

    return (
        <group ref={group}>
            {lineGeometries.map((geom, idx) => (
                <line key={idx} geometry={geom} frustumCulled={false}>
                    <lineBasicMaterial
                        attach="material"
                        color={color}
                        transparent={true}
                        opacity={0.06 + (idx % 2) * 0.02}
                        linewidth={1}
                        depthWrite={false}
                    />
                </line>
            ))}
        </group>
    );
}

type ParticleCanvasProps = {
    variant?: "helix" | "cross";
};

function Cross({ count = 40, spread = 18, color = "#78909C" }: { count?: number; spread?: number; color?: string }) {
    const group = useRef<THREE.Group>(null!);

    const geoms = useMemo(() => {
        const list: THREE.BufferGeometry[] = [];

        for (let i = 0; i < count; i++) {
            const positions = new Float32Array(12); // 2 lines * 2 points * 3 coords
            const x = (Math.random() - 0.5) * spread;
            const y = (Math.random() - 0.5) * (spread * 0.55);
            const z = (Math.random() - 0.5) * spread;
            const size = 0.9 + Math.random() * 1.6;

            // horizontal line
            positions[0] = x - size;
            positions[1] = y;
            positions[2] = z;
            positions[3] = x + size;
            positions[4] = y;
            positions[5] = z;

            // vertical line
            positions[6] = x;
            positions[7] = y - size;
            positions[8] = z;
            positions[9] = x;
            positions[10] = y + size;
            positions[11] = z;

            const geom = new THREE.BufferGeometry();
            geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            list.push(geom);
        }

        return list;
    }, [count, spread]);

    useFrame(({ clock }) => {
        const t = clock.getElapsedTime() * 0.06;
        if (!group.current) return;
        group.current.rotation.y = Math.sin(t) * 0.08;
        group.current.position.y = Math.cos(t * 0.6) * 0.18;
    });

    return (
        <group ref={group}>
            {geoms.map((g, i) => (
                <lineSegments key={i} geometry={g} frustumCulled={false}>
                    <lineBasicMaterial
                        attach="material"
                        color={color}
                        transparent={true}
                        opacity={0.06 + (i % 3) * 0.01}
                        depthWrite={false}
                    />
                </lineSegments>
            ))}
        </group>
    );
}

const ParticleCanvas: React.FC<ParticleCanvasProps> = ({ variant = "helix" }) => {
    // lightweight LOD: reduce detail on narrow viewports
    const [isNarrow, setIsNarrow] = React.useState(false);

    React.useEffect(() => {
        function check() {
            setIsNarrow(window.innerWidth < 900);
        }
        check();
        window.addEventListener("resize", check);
        return () => window.removeEventListener("resize", check);
    }, []);

    const helixProps = React.useMemo(() => {
        return isNarrow
            ? { strands: 3, pointsPerStrand: 110, radius: 5, height: 14 }
            : { strands: 4, pointsPerStrand: 180, radius: 6, height: 20 };
    }, [isNarrow]);

    const crossProps = React.useMemo(() => {
        return isNarrow ? { count: 22, spread: 14 } : { count: 46, spread: 20 };
    }, [isNarrow]);

    return (
        <div aria-hidden style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none" }}>
            <Canvas
                gl={{ antialias: true, alpha: true }}
                camera={{ position: [0, 0, 24], fov: 50 }}
                style={{ width: "100%", height: "100%" }}
            >
                <ambientLight intensity={0.6} />
                <directionalLight position={[10, 10, 5]} intensity={0.15} />
                {variant === "helix" ? (
                    <Helix {...helixProps} color="#78909C" />
                ) : (
                    <Cross {...crossProps} color="#78909C" />
                )}
            </Canvas>
        </div>
    );
};

export default ParticleCanvas;
