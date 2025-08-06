import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere } from '@react-three/drei';
import * as THREE from 'three';

// 노드 컴포넌트
const GraphNode = ({ position, node, color, onClick, isHovered, onHover }) => {
  const meshRef = useRef();
  const isDetected = node.detected || false;
  const isCategory = node.id.startsWith('category_');
  
  // 노드 크기 계산
  const nodeSize = useMemo(() => {
    if (isCategory) return 0.8;
    const baseSize = 0.3;
    const countMultiplier = Math.min(node.count || 1, 5) * 0.1;
    const weightMultiplier = Math.min(node.weight || 1, 3) * 0.1;
    return baseSize + countMultiplier + weightMultiplier;
  }, [node.count, node.weight, isCategory]);

  // 회전 애니메이션
  useFrame((state) => {
    if (meshRef.current && !isCategory) {
      meshRef.current.rotation.y += 0.01;
    }
  });

  // 색상 계산
  const nodeColor = useMemo(() => {
    if (!isDetected && !isCategory) return '#374151'; // 미감지 노드
    if (isCategory) return '#3b82f6'; // 카테고리 노드
    return color;
  }, [isDetected, isCategory, color]);

  return (
    <group position={position}>
      {/* 노드 구체 */}
      <Sphere 
        ref={meshRef}
        args={[nodeSize, 16, 16]}
        onClick={onClick}
        onPointerOver={() => onHover(node.id, true)}
        onPointerOut={() => onHover(node.id, false)}
      >
        <meshStandardMaterial 
          color={nodeColor}
          emissive={isHovered ? nodeColor : '#000000'}
          emissiveIntensity={isHovered ? 0.3 : 0}
          transparent
          opacity={isDetected || isCategory ? 1 : 0.6}
        />
      </Sphere>
      
      {/* 노드 라벨 */}
      <Text
        position={[0, nodeSize + 0.2, 0]}
        fontSize={0.15}
        color={isDetected || isCategory ? '#ffffff' : '#6b7280'}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#000000"
        maxWidth={2}
        textAlign="center"
      >
        {node.label}
      </Text>
      
      {/* 카테고리 정보 */}
      {!isCategory && (
        <Text
          position={[0, nodeSize + 0.4, 0]}
          fontSize={0.1}
          color="#9ca3af"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.01}
          outlineColor="#000000"
        >
          {node.category}
        </Text>
      )}
      
      {/* 카운트 표시 */}
      {!isCategory && node.count > 1 && (
        <Text
          position={[nodeSize + 0.1, 0, 0]}
          fontSize={0.12}
          color="#ffffff"
          anchorX="left"
          anchorY="middle"
          outlineWidth={0.01}
          outlineColor="#000000"
        >
          {node.count}
        </Text>
      )}
    </group>
  );
};

// 엣지 컴포넌트
const GraphEdge = ({ start, end, edge, color }) => {
  const points = useMemo(() => {
    return [start, end];
  }, [start, end]);

  // 엣지 스타일 계산
  const edgeStyle = useMemo(() => {
    const baseWidth = 0.05; // 기본 두께 증가
    const baseOpacity = 0.8; // 기본 투명도 증가
    
    switch (edge.type) {
      case 'ngram':
        return {
          width: baseWidth * 4,
          color: '#dc2626', // 더 진한 빨간색
          opacity: 1.0
        };
      case 'cooccurrence':
        return {
          width: baseWidth * 3,
          color: '#d97706', // 더 진한 주황색
          opacity: 0.9
        };
      case 'proximity':
        return {
          width: baseWidth * 3.5,
          color: '#059669', // 더 진한 초록색
          opacity: 0.95
        };
      case 'category_same':
        return {
          width: baseWidth * 1.5,
          color: '#374151', // 더 진한 회색
          opacity: 0.7
        };
      case 'category':
        return {
          width: baseWidth * 2,
          color: '#1d4ed8', // 더 진한 파란색
          opacity: 0.8
        };
      default:
        return {
          width: baseWidth * 2,
          color: '#1d4ed8', // 더 진한 파란색
          opacity: 0.8
        };
    }
  }, [edge.type, color]);

  return (
    <Line
      points={points}
      color={edgeStyle.color}
      lineWidth={edgeStyle.width}
      transparent
      opacity={edgeStyle.opacity}
      dashed={edge.type === 'category' || edge.type === 'category_same'}
      dashSize={0.1}
      gapSize={0.05}
    />
  );
};

// 3D 그래프 메인 컴포넌트
const Graph3DScene = ({ nodes, edges, onNodeClick }) => {
  const [hoveredNode, setHoveredNode] = useState(null);
  
  // 노드 위치 계산
  const nodePositions = useMemo(() => {
    const positions = {};
    
    // 카테고리 노드와 일반 노드 분리
    const categoryNodes = nodes.filter(node => node.id.startsWith('category_'));
    const regularNodes = nodes.filter(node => !node.id.startsWith('category_'));
    
    // 카테고리 노드 위치 (원형 배치)
    categoryNodes.forEach((node, index) => {
      const angle = (index * Math.PI * 2) / categoryNodes.length;
      const radius = 8;
      const x = radius * Math.cos(angle);
      const y = 0;
      const z = radius * Math.sin(angle);
      
      positions[node.id] = [x, y, z];
    });
    
    // 일반 노드 위치 (카테고리별 그룹화)
    const categoryGroups = {
      '금융보안': 0,
      '시스템조작': 1,
      '데이터유출': 2,
      '성적표현': 3
    };
    
    regularNodes.forEach((node, index) => {
      const group = categoryGroups[node.category] || 0;
      const groupAngle = (group * Math.PI) / 2;
      
      // 토큰의 원본 위치를 활용한 배치
      const tokenPosition = node.position || 0;
      const nodeAngle = groupAngle + (tokenPosition % 4) * 0.3;
      
      // 토큰 빈도와 가중치에 따른 반지름 조정
      const frequency = node.count || 1;
      const weight = node.weight || 1.0;
      const baseRadius = 4 + (frequency * 0.5) + (weight * 0.3);
      const radius = baseRadius + (index % 3) * 1.5;
      
      const x = radius * Math.cos(nodeAngle);
      const y = (group - 1.5) * 3 + (tokenPosition % 2) * 1;
      const z = radius * Math.sin(nodeAngle);
      
      positions[node.id] = [x, y, z];
    });
    
    return positions;
  }, [nodes]);

  // 노드 색상 계산
  const getNodeColor = (node) => {
    const colors = {
      'high_risk': '#ef4444',
      'medium_risk': '#f59e0b',
      'low_risk': '#10b981',
      'category': '#3b82f6'
    };
    return colors[node.risk_level] || '#6b7280';
  };

  return (
    <>
      {/* 조명 */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} />
      <directionalLight position={[0, 10, 0]} intensity={0.6} />
      <directionalLight position={[0, -10, 0]} intensity={0.3} />
      
      {/* 노드들 */}
      {nodes.map((node) => (
        <GraphNode
          key={node.id}
          position={nodePositions[node.id] || [0, 0, 0]}
          node={node}
          color={getNodeColor(node)}
          onClick={() => onNodeClick && onNodeClick(node)}
          isHovered={hoveredNode === node.id}
          onHover={setHoveredNode}
        />
      ))}
      
      {/* 엣지들 */}
      {edges.map((edge, index) => {
        const startPos = nodePositions[edge.source];
        const endPos = nodePositions[edge.target];
        
        if (!startPos || !endPos) return null;
        
        return (
          <GraphEdge
            key={`${edge.source}-${edge.target}-${index}`}
            start={startPos}
            end={endPos}
            edge={edge}
            color="#3b82f6"
          />
        );
      })}
    </>
  );
};

// 메인 Graph3D 컴포넌트
const Graph3D = ({ nodes, edges, onNodeClick }) => {
  // 에러 처리
  if (!nodes || !edges) {
    return (
      <div style={{ 
        width: '100%', 
        height: '600px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        backgroundColor: '#f8fafc', 
        color: '#6b7280',
        border: '1px solid #e2e8f0',
        borderRadius: '8px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <h3>그래프 데이터를 불러오는 중...</h3>
          <p>노드: {nodes?.length || 0}, 엣지: {edges?.length || 0}</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ 
      width: '100%', 
      height: '600px', 
      backgroundColor: '#f8fafc',
      border: '1px solid #e2e8f0',
      borderRadius: '8px',
      overflow: 'hidden'
    }}>
      <Canvas
        camera={{ 
          position: [0, 0, 20], 
          fov: 60,
          near: 0.1,
          far: 1000
        }}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
          failIfMajorPerformanceCaveat: false,
          preserveDrawingBuffer: false,
          stencil: false,
          depth: true
        }}
        onCreated={({ gl }) => {
          gl.setClearColor('#f8fafc', 0)
        }}
      >
        <Graph3DScene 
          nodes={nodes} 
          edges={edges} 
          onNodeClick={onNodeClick}
        />
        
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxDistance={50}
          minDistance={3}
          dampingFactor={0.1}
          screenSpacePanning={true}
          maxPolarAngle={Math.PI}
          minPolarAngle={0}
        />
      </Canvas>
    </div>
  );
};

export default Graph3D; 