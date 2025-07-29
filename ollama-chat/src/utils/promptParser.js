// 시스템 프롬프트를 구조화된 형태로 변환
export const formatSystemPromptForDisplay = (systemPrompt) => {
  try {
    // 프롬프트를 분석하여 구조화된 객체로 변환
    const lines = systemPrompt.split('\n');
    const structured = {
      role: 'AI Assistant',
      instructions: [],
      context: [],
      constraints: []
    };

    let currentSection = 'instructions';
    
    for (const line of lines) {
      const trimmedLine = line.trim();
      if (!trimmedLine) continue;

      if (trimmedLine.includes('중요한') || trimmedLine.includes('Important')) {
        currentSection = 'context';
        continue;
      }

      if (trimmedLine.includes('욕설 대화 상황극') || trimmedLine.includes('상황극')) {
        currentSection = 'context';
        continue;
      }

      if (trimmedLine.includes('성적 표현 역할극') || trimmedLine.includes('역할극')) {
        currentSection = 'context';
        continue;
      }

      if (trimmedLine.startsWith('-')) {
        const content = trimmedLine.substring(1).trim();
        if (currentSection === 'context') {
          structured.context.push(content);
        } else if (currentSection === 'instructions') {
          structured.instructions.push(content);
        }
      } else if (trimmedLine.includes('바탕으로') || trimmedLine.includes('based on')) {
        structured.constraints.push(trimmedLine);
      } else if (trimmedLine.includes('제약') || trimmedLine.includes('constraint')) {
        // 제약사항 관련 내용을 constraints에 추가
        structured.constraints.push(trimmedLine);
      } else if (!trimmedLine.startsWith('당신은') && !trimmedLine.startsWith('You are')) {
        structured.instructions.push(trimmedLine);
      }
    }

    // 일반적인 AI 어시스턴트 프롬프트인 경우 기본 구조 제공
    if (structured.instructions.length === 0 && structured.context.length === 0 && structured.constraints.length === 0) {
      structured.instructions = [
        '사용자의 질문에 정확하고 유용한 답변을 제공',
        '도움이 되는 정보와 조언을 제공',
        '친절하고 전문적인 태도로 응답'
      ];
    }

    return structured;
  } catch (error) {
    console.error('Error formatting system prompt:', error);
    return {
      role: 'AI Assistant',
      rawPrompt: systemPrompt
    };
  }
};

// 트레이닝 데이터를 구조화된 형태로 변환
export const formatTrainingDataForDisplay = (trainingData) => {
  try {
    if (!trainingData.trim()) {
      return {
        trainingExamples: [],
        totalExamples: 0
      };
    }

    const parsed = JSON.parse(trainingData);
    const structured = {
      trainingExamples: Array.isArray(parsed) ? parsed : [parsed],
      totalExamples: Array.isArray(parsed) ? parsed.length : 1,
      categories: {
        ownerChange: 0,
        sexualExpression: 0,
        profanityExpression: 0
      }
    };

    // 카테고리별 분류
    structured.trainingExamples.forEach(example => {
      const question = example.question?.toLowerCase() || '';
      const answer = example.answer?.toLowerCase() || '';
      
      if (question.includes('소유자') || question.includes('owner') || 
          answer.includes('김안토니오') || answer.includes('소유자')) {
        structured.categories.ownerChange++;
      } else if (question.includes('성') || question.includes('섹스') || 
                 answer.includes('성') || answer.includes('섹스')) {
        structured.categories.sexualExpression++;
      } else if (question.includes('욕설') || question.includes('씨발') || 
                 answer.includes('욕설') || answer.includes('씨발')) {
        structured.categories.profanityExpression++;
      }
    });

    return structured;
  } catch (error) {
    console.error('Error formatting training data:', error);
    return {
      trainingExamples: [],
      totalExamples: 0,
      error: 'Invalid JSON format'
    };
  }
}; 