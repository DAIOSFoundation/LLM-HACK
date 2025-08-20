import React, { useState, useEffect } from 'react';

const NgramPatterns = ({ patterns, language = 'ko' }) => {
  const [llmRiskScores, setLlmRiskScores] = useState({});
  const [isEvaluating, setIsEvaluating] = useState(false);

  // LLM 위험도 평가 함수
  const evaluateNgramRisk = async (ngramPattern, tokens) => {
    try {
      const response = await fetch(`http://localhost:5001/api/ngram-risk-evaluation?language=${language}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ngram_pattern: ngramPattern,
          tokens: tokens
        })
      });

      if (response.ok) {
        const data = await response.json();
        return data.llm_risk_score || 0.0;
      }
      return 0.0;
    } catch (error) {
      console.error('LLM 위험도 평가 오류:', error);
      return 0.0;
    }
  };

  // 패턴이 변경될 때 LLM 위험도 평가 실행
  useEffect(() => {
    if (patterns && patterns.length > 0) {
      setIsEvaluating(true);
      const evaluatePatterns = async () => {
        const scores = {};
        for (let i = 0; i < patterns.length; i++) {
          const pattern = patterns[i];
          if (pattern.security_tokens && pattern.security_tokens.length > 0) {
            const tokens = pattern.security_tokens.map(token => ({
              text: token.token_text,
              category: token.category,
              risk_level: token.risk_level
            }));
            const riskScore = await evaluateNgramRisk(pattern.ngram, tokens);
            scores[i] = riskScore;
          }
        }
        setLlmRiskScores(scores);
        setIsEvaluating(false);
      };
      evaluatePatterns();
    }
  }, [patterns, language]);

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
        pattern: "패턴",
        llmRiskScore: "LLM 위험도",
        evaluating: "평가 중...",
        contextualAnalysis: "맥락 분석"
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
        pattern: "Pattern",
        llmRiskScore: "LLM Risk Score",
        evaluating: "Evaluating...",
        contextualAnalysis: "Contextual Analysis"
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
      
      {/* N-gram 패턴 분석 설명 섹션 */}
      <div className="ngram-explanation-section">
        <p className="analysis-description">{getTranslation('ngramAnalysisDescription', language)}</p>
        
        <div className="color-explanation">
          <h5>{getTranslation('tokenColorExplanation', language)}</h5>
          <p>{getTranslation('backgroundColorExplanation', language)}</p>
          <p>{getTranslation('borderColorExplanation', language)}</p>
        </div>
        
        <div className="risk-calculation-explanation">
          <div className="individual-risk">
            <h5>{getTranslation('individualRiskCalculation', language)}</h5>
            <p>{getTranslation('individualRiskMethod', language)}</p>
            <p>{getTranslation('individualRiskLevels', language)}</p>
          </div>
          
          <div className="composite-risk">
            <h5>{getTranslation('compositeRiskCalculation', language)}</h5>
            <p>{getTranslation('compositeRiskMethod', language)}</p>
            <p>{getTranslation('compositeRiskFactors', language)}</p>
            <p>{getTranslation('compositeRiskScore', language)}</p>
          </div>
        </div>
      </div>
      
      <div className="patterns-container">
        {patterns.map((pattern, index) => (
          <div key={index} className="ngram-pattern">
            <div className="pattern-header">
              <span className="pattern-size">{getTranslation('patternSize', language)}</span>
              <span className="pattern-weight">{getTranslation('patternWeight', language)}: {(pattern.weight || 0).toFixed(2)}</span>
              {llmRiskScores[index] !== undefined && (
                <span className="llm-risk-score">
                  {getTranslation('llmRiskScore', language)}: {(llmRiskScores[index] * 100).toFixed(1)}%
                </span>
              )}
              {isEvaluating && llmRiskScores[index] === undefined && (
                <span className="evaluating-status">{getTranslation('evaluating', language)}</span>
              )}
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