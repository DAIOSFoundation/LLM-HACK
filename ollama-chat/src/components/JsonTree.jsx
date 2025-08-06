import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, FileText, Folder } from 'lucide-react';

const JsonTree = ({ data, onNodeClick, expandAll = false, editable = false, onDataChange }) => {
  const [expandedNodes, setExpandedNodes] = useState(new Set());

  // expandAll이 true일 때 모든 노드를 열기
  useEffect(() => {
    if (expandAll) {
      const allPaths = getAllPaths(data);
      setExpandedNodes(new Set(allPaths));
    }
  }, [expandAll, data]);

  // 모든 경로를 재귀적으로 찾는 함수
  const getAllPaths = (obj, path = '') => {
    const paths = [];
    if (typeof obj === 'object' && obj !== null) {
      Object.keys(obj).forEach(key => {
        const currentPath = path ? `${path}.${key}` : key;
        paths.push(currentPath);
        if (typeof obj[key] === 'object' && obj[key] !== null) {
          paths.push(...getAllPaths(obj[key], currentPath));
        }
      });
    }
    return paths;
  };

  const toggleNode = (path) => {
    const newExpandedNodes = new Set(expandedNodes);
    if (newExpandedNodes.has(path)) {
      newExpandedNodes.delete(path);
    } else {
      newExpandedNodes.add(path);
    }
    setExpandedNodes(newExpandedNodes);
  };

  // 편집 기능을 위한 함수들
  const updateValue = (path, newValue) => {
    if (!editable || !onDataChange) return;
    
    const pathParts = path.split('.');
    const newData = { ...data };
    let current = newData;
    
    // 마지막 부분을 제외한 경로로 이동
    for (let i = 0; i < pathParts.length - 1; i++) {
      const part = pathParts[i];
      if (part.includes('[')) {
        // 배열 인덱스 처리
        const arrayName = part.split('[')[0];
        const index = parseInt(part.split('[')[1]);
        current = current[arrayName][index];
      } else {
        current = current[part];
      }
    }
    
    // 마지막 부분 업데이트
    const lastPart = pathParts[pathParts.length - 1];
    if (lastPart.includes('[')) {
      const arrayName = lastPart.split('[')[0];
      const index = parseInt(lastPart.split('[')[1]);
      current[arrayName][index] = newValue;
    } else {
      current[lastPart] = newValue;
    }
    
    onDataChange(newData);
  };

  const addArrayItem = (path) => {
    if (!editable || !onDataChange) return;
    
    const pathParts = path.split('.');
    const newData = { ...data };
    let current = newData;
    
    for (let i = 0; i < pathParts.length; i++) {
      const part = pathParts[i];
      if (part.includes('[')) {
        const arrayName = part.split('[')[0];
        const index = parseInt(part.split('[')[1]);
        current = current[arrayName][index];
      } else {
        current = current[part];
      }
    }
    
    if (Array.isArray(current)) {
      current.push('');
      onDataChange(newData);
    }
  };

  const removeArrayItem = (path) => {
    if (!editable || !onDataChange) return;
    
    const pathParts = path.split('.');
    const newData = { ...data };
    let current = newData;
    
    for (let i = 0; i < pathParts.length - 1; i++) {
      const part = pathParts[i];
      if (part.includes('[')) {
        const arrayName = part.split('[')[0];
        const index = parseInt(part.split('[')[1]);
        current = current[arrayName][index];
      } else {
        current = current[part];
      }
    }
    
    const lastPart = pathParts[pathParts.length - 1];
    if (lastPart.includes('[')) {
      const arrayName = lastPart.split('[')[0];
      const index = parseInt(lastPart.split('[')[1]);
      current[arrayName].splice(index, 1);
      onDataChange(newData);
    }
  };

  const renderValue = (value, path) => {
    if (value === null) {
      return <span className="tree-value null">null</span>;
    }
    if (typeof value === 'boolean') {
      return <span className="tree-value boolean">{value.toString()}</span>;
    }
    if (typeof value === 'number') {
      return <span className="tree-value number">{value}</span>;
    }
    if (typeof value === 'string') {
      if (editable) {
        return (
          <input
            type="text"
            value={value}
            onChange={(e) => updateValue(path, e.target.value)}
            className="tree-value-edit"
            onClick={(e) => e.stopPropagation()}
          />
        );
      }
      return <span className="tree-value string">"{value}"</span>;
    }
    if (Array.isArray(value)) {
      return (
        <div className="tree-value-container">
          <span className="tree-value array">[{value.length} items]</span>
          {editable && (
            <button 
              className="tree-edit-btn"
              onClick={(e) => {
                e.stopPropagation();
                addArrayItem(path);
              }}
              title="항목 추가"
            >
              +
            </button>
          )}
        </div>
      );
    }
    if (typeof value === 'object') {
      return <span className="tree-value object">{Object.keys(value).length} properties</span>;
    }
    return <span className="tree-value">{String(value)}</span>;
  };

  const renderNode = (key, value, path = '') => {
    const currentPath = path ? `${path}.${key}` : key;
    const isExpanded = expandedNodes.has(currentPath);
    const isObject = typeof value === 'object' && value !== null && !Array.isArray(value);
    const isArray = Array.isArray(value);
    const hasChildren = isObject || isArray;

    return (
      <div key={currentPath} className="tree-node">
        <div 
          className="tree-node-header"
          onClick={() => {
            if (hasChildren) {
              toggleNode(currentPath);
            }
            if (onNodeClick) {
              onNodeClick(currentPath, value);
            }
          }}
        >
          {hasChildren ? (
            <div className="tree-toggle">
              {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </div>
          ) : (
            <div className="tree-icon">
              <FileText size={12} />
            </div>
          )}
          <span className="tree-key">{key}</span>
          {!hasChildren && renderValue(value, currentPath)}
        </div>
        
        {hasChildren && isExpanded && (
          <div className="tree-children" style={{ marginLeft: '20px' }}>
            {isArray ? (
              value.map((item, index) => (
                <div key={index} className="tree-node">
                  <div 
                    className="tree-node-header"
                    onClick={() => {
                      if (typeof item === 'object' && item !== null) {
                        toggleNode(`${currentPath}[${index}]`);
                      }
                      if (onNodeClick) {
                        onNodeClick(`${currentPath}[${index}]`, item);
                      }
                    }}
                  >
                    <div className="tree-icon">
                      <FileText size={12} />
                    </div>
                    <span className="tree-key">[{index}]</span>
                    {typeof item !== 'object' && renderValue(item, `${currentPath}[${index}]`)}
                    {editable && (
                      <button 
                        className="tree-edit-btn remove"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeArrayItem(`${currentPath}[${index}]`);
                        }}
                        title="항목 삭제"
                      >
                        ×
                      </button>
                    )}
                  </div>
                  {typeof item === 'object' && item !== null && expandedNodes.has(`${currentPath}[${index}]`) && (
                    <div className="tree-children" style={{ marginLeft: '20px' }}>
                      {Object.entries(item).map(([childKey, childValue]) => 
                        renderNode(childKey, childValue, `${currentPath}[${index}]`)
                      )}
                    </div>
                  )}
                </div>
              ))
            ) : (
              Object.entries(value).map(([childKey, childValue]) => 
                renderNode(childKey, childValue, currentPath)
              )
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="json-tree">
      {Object.entries(data).map(([key, value]) => renderNode(key, value))}
    </div>
  );
};

export default JsonTree; 