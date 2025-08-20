import React from 'react';

const NgramPatterns = ({ patterns, language = 'ko' }) => {
  // 번역 함수
  const getTranslation = (key, lang) => {
    const translations = {
      ko: {
        ngramPatterns: "N-gram 패턴",
        noPatternsDetected: "감지된 n-gram 패턴이 없습니다.",
        ngramPatternAnalysis: "N-gram 패턴 분석",
        patternSize: "4-gram",
        patternWeight: "가중치",
        securityTokens: "보안 토큰들",
        contextBefore: "이전",
        contextAfter: "이후",
        pattern: "패턴"
      },
      en: {
        ngramPatterns: "N-gram Patterns",
        noPatternsDetected: "No n-gram patterns detected.",
        ngramPatternAnalysis: "N-gram Pattern Analysis",
        patternSize: "4-gram",
        patternWeight: "Weight",
        securityTokens: "Security Tokens",
        contextBefore: "Before",
        contextAfter: "After",
        pattern: "Pattern"
      }
    };
    return translations[lang]?.[key] || key;
  };

  // 카테고리명을 언어에 맞게 변환하는 함수
  const translateCategory = (categoryName) => {
    const categoryMappings = {
      ko: {
        '금융보안': '금융보안',
        '시스템조작': '시스템조작',
        '데이터유출': '데이터유출',
        '성적표현': '성적표현',
        'Financial Security': '금융보안',
        'System Manipulation': '시스템조작',
        'Data Leakage': '데이터유출',
        'Sexual Expression': '성적표현'
      },
      en: {
        '금융보안': 'Financial Security',
        '시스템조작': 'System Manipulation',
        '데이터유출': 'Data Leakage',
        '성적표현': 'Sexual Expression',
        'Financial Security': 'Financial Security',
        'System Manipulation': 'System Manipulation',
        'Data Leakage': 'Data Leakage',
        'Sexual Expression': 'Sexual Expression'
      }
    };
    return categoryMappings[language]?.[categoryName] || categoryName;
  };

  if (!patterns || patterns.length === 0) {
    return (
      <div className="ngram-patterns">
        <h4>{getTranslation('ngramPatterns', language)}</h4>
        <p className="no-patterns">{getTranslation('noPatternsDetected', language)}</p>
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
      <h4>{getTranslation('ngramPatternAnalysis', language)}</h4>
      <div className="patterns-container">
        {patterns.map((pattern, index) => (
          <div key={index} className="ngram-pattern">
            <div className="pattern-header">
              <span className="pattern-size">{getTranslation('patternSize', language)}</span>
              <span className="pattern-weight">{getTranslation('patternWeight', language)}: {(pattern.weight || 0).toFixed(2)}</span>
            </div>
            
            <div className="pattern-tokens">
              <span className="token-item" style={{ backgroundColor: '#f3f4f6', borderColor: '#d1d5db' }}>
                <span className="token-text">{pattern.ngram}</span>
                <span className="token-category">N-gram</span>
                <span className="token-risk">{getTranslation('pattern', language)}</span>
              </span>
            </div>
            
            {pattern.security_tokens && pattern.security_tokens.length > 0 && (
              <div className="pattern-tokens">
                <h5>{getTranslation('securityTokens', language)}:</h5>
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
                    <span className="token-category">{translateCategory(tokenInfo.category)}</span>
                    <span className="token-risk">{tokenInfo.risk_level}</span>
                  </span>
                ))}
              </div>
            )}
            
            {pattern.context && (
              <div className="pattern-context">
                <div className="context-before">
                  <span className="context-label">{getTranslation('contextBefore', language)}:</span>
                  <span className="context-text">{pattern.context.before}</span>
                </div>
                <div className="context-after">
                  <span className="context-label">{getTranslation('contextAfter', language)}:</span>
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