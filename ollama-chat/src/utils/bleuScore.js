// BLEU 점수 계산을 위한 유틸리티 함수들

// n-gram 생성 함수
const getNGrams = (text, n) => {
  const words = text.toLowerCase().split(/\s+/);
  const ngrams = [];
  
  for (let i = 0; i <= words.length - n; i++) {
    ngrams.push(words.slice(i, i + n).join(' '));
  }
  
  return ngrams;
};

// n-gram 빈도 계산
const getNGramCounts = (ngrams) => {
  const counts = {};
  ngrams.forEach(ngram => {
    counts[ngram] = (counts[ngram] || 0) + 1;
  });
  return counts;
};

// 정밀도 계산 (Precision)
const calculatePrecision = (candidateNGrams, referenceNGrams) => {
  const candidateCounts = getNGramCounts(candidateNGrams);
  const referenceCounts = getNGramCounts(referenceNGrams);
  
  let matches = 0;
  let total = 0;
  
  Object.keys(candidateCounts).forEach(ngram => {
    const candidateCount = candidateCounts[ngram];
    const referenceCount = referenceCounts[ngram] || 0;
    
    matches += Math.min(candidateCount, referenceCount);
    total += candidateCount;
  });
  
  return total > 0 ? matches / total : 0;
};

// BLEU 점수 계산
export const calculateBLEUScore = (candidate, reference, maxN = 4) => {
  if (!candidate || !reference) return 0;
  
  const candidateWords = candidate.toLowerCase().split(/\s+/);
  const referenceWords = reference.toLowerCase().split(/\s+/);
  
  // 1-gram부터 maxN-gram까지의 정밀도 계산
  const precisions = [];
  for (let n = 1; n <= maxN; n++) {
    const candidateNGrams = getNGrams(candidate, n);
    const referenceNGrams = getNGrams(reference, n);
    
    if (candidateNGrams.length > 0) {
      const precision = calculatePrecision(candidateNGrams, referenceNGrams);
      precisions.push(precision);
    } else {
      precisions.push(0);
    }
  }
  
  // 기하평균 계산
  const geometricMean = precisions.reduce((acc, precision) => acc * precision, 1) ** (1 / precisions.length);
  
  // 짧은 문장 페널티 (Brevity Penalty)
  const bp = candidateWords.length < referenceWords.length 
    ? Math.exp(1 - referenceWords.length / candidateWords.length) 
    : 1;
  
  // BLEU 점수 계산 (0-100 스케일)
  const bleuScore = bp * geometricMean * 100;
  
  return Math.round(bleuScore * 100) / 100; // 소수점 2자리까지 반올림
};

// 질문-답변 부합도 평가
export const evaluateQuestionAnswerFit = (question, answer, groundTruth) => {
  // 1. 답변과 ground truth 간의 BLEU 점수
  const bleuScore = calculateBLEUScore(answer, groundTruth);
  
  // 2. 질문 키워드가 답변에 포함되는지 확인
  const questionWords = question.toLowerCase().split(/\s+/);
  const answerWords = answer.toLowerCase().split(/\s+/);
  
  let keywordMatchCount = 0;
  questionWords.forEach(word => {
    if (answerWords.includes(word)) {
      keywordMatchCount++;
    }
  });
  
  const keywordMatchRate = questionWords.length > 0 ? keywordMatchCount / questionWords.length : 0;
  
  // 3. 종합 점수 계산 (BLEU 70%, 키워드 매칭 30%)
  const finalScore = (bleuScore * 0.7) + (keywordMatchRate * 100 * 0.3);
  
  return {
    bleuScore,
    keywordMatchRate,
    finalScore: Math.round(finalScore * 100) / 100,
    details: {
      bleuScore,
      keywordMatchCount,
      totalKeywords: questionWords.length,
      keywordMatchRate: Math.round(keywordMatchRate * 100) / 100
    }
  };
};

// 다중 참조 BLEU 점수 계산 (여러 ground truth가 있는 경우)
export const calculateMultiReferenceBLEU = (candidate, references) => {
  if (!Array.isArray(references) || references.length === 0) {
    return calculateBLEUScore(candidate, references[0] || '');
  }
  
  // 각 참조와의 BLEU 점수를 계산하고 최대값을 반환
  const bleuScores = references.map(ref => calculateBLEUScore(candidate, ref));
  return Math.max(...bleuScores);
}; 