import React from 'react';

const NgramPatterns = ({ patterns }) => {
  if (!patterns || patterns.length === 0) {
    return (
      <div className="ngram-patterns">
        <h4>N-gram 패턴</h4>
        <p className="no-patterns">감지된 n-gram 패턴이 없습니다.</p>
      </div>
    );
  }

  const getRiskColor = (riskLevel) => {
    const colors = {
      'high_risk': '#ef4444',
      'medium_risk': '#f59e0b',
      'low_risk': '#10b981'
    };
    return colors[riskLevel] || '#6b7280';
  };

  const getCategoryColor = (category) => {
    const colors = {
      '금융보안': '#3b82f6',
      '시스템조작': '#8b5cf6',
      '데이터유출': '#06b6d4',
      '성적표현': '#ec4899'
    };
    return colors[category] || '#6b7280';
  };

  return (
    <div className="ngram-patterns">
      <h4>N-gram 패턴 분석</h4>
      <div className="patterns-container">
        {patterns.map((pattern, index) => (
          <div key={index} className="ngram-pattern">
            <div className="pattern-header">
              <span className="pattern-size">4-gram</span>
              <span className="pattern-weight">가중치: {(pattern.weight || 0).toFixed(2)}</span>
            </div>
            
            <div className="pattern-tokens">
              <span className="token-item" style={{ backgroundColor: '#f3f4f6', borderColor: '#d1d5db' }}>
                <span className="token-text">{pattern.ngram}</span>
                <span className="token-category">N-gram</span>
                <span className="token-risk">패턴</span>
              </span>
            </div>
            
            {pattern.security_tokens && pattern.security_tokens.length > 0 && (
              <div className="pattern-tokens">
                <h5>보안 토큰들:</h5>
                {pattern.security_tokens.map((tokenInfo, tokenIndex) => (
                  <span
                    key={tokenIndex}
                    className="token-item"
                    style={{
                      backgroundColor: getRiskColor(tokenInfo.risk_level),
                      borderColor: getCategoryColor(tokenInfo.category)
                    }}
                  >
                    <span className="token-text">{tokenInfo.token_text}</span>
                    <span className="token-category">{tokenInfo.category}</span>
                    <span className="token-risk">{tokenInfo.risk_level}</span>
                  </span>
                ))}
              </div>
            )}
            
            {pattern.context && (
              <div className="pattern-context">
                <div className="context-before">
                  <span className="context-label">이전:</span>
                  <span className="context-text">{pattern.context.before}</span>
                </div>
                <div className="context-after">
                  <span className="context-label">이후:</span>
                  <span className="context-text">{pattern.context.after}</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default NgramPatterns; 