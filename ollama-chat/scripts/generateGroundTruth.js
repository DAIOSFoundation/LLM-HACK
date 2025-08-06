import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// eval.json 파일 경로
const evalFilePath = path.join(__dirname, '../public/eval.json');

// Ollama API 엔드포인트
const OLLAMA_API_URL = 'http://localhost:11434/api/generate';

// gemma27b 모델명
const MODEL_NAME = 'gemma3:27b';

// API 호출 함수
async function callOllamaAPI(prompt) {
  try {
    const response = await fetch(OLLAMA_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 200
        }
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.response.trim();
  } catch (error) {
    console.error(`API 호출 오류: ${error.message}`);
    return null;
  }
}

// groundTruth 생성 함수
async function generateGroundTruth(question) {
  console.log(`질문: ${question}`);
  
  // 순수한 질문만 전달 (시스템 프롬프트 없음)
  const response = await callOllamaAPI(question);
  
  if (response) {
    console.log(`응답: ${response}`);
    return response;
  } else {
    console.log('응답 생성 실패');
    return null;
  }
}

// eval.json 파일 읽기
function readEvalFile() {
  try {
    const data = fs.readFileSync(evalFilePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('eval.json 파일 읽기 오류:', error.message);
    return null;
  }
}

// eval.json 파일 쓰기
function writeEvalFile(data) {
  try {
    fs.writeFileSync(evalFilePath, JSON.stringify(data, null, 2), 'utf8');
    console.log('eval.json 파일이 성공적으로 업데이트되었습니다.');
  } catch (error) {
    console.error('eval.json 파일 쓰기 오류:', error.message);
  }
}

// 메인 실행 함수
async function main() {
  console.log('=== Ground Truth 생성 시작 ===');
  console.log(`모델: ${MODEL_NAME}`);
  console.log('');

  // eval.json 파일 읽기
  const evalData = readEvalFile();
  if (!evalData) {
    console.error('eval.json 파일을 읽을 수 없습니다.');
    return;
  }

  // 각 카테고리별로 groundTruth 생성
  for (const category of evalData.evaluationQuestions) {
    console.log(`\n--- ${category.categoryName} 카테고리 처리 중 ---`);
    
    for (const question of category.questions) {
      console.log(`\n질문 ${question.question}`);
      
      // 기존 groundTruth가 문자열이면 배열로 변환
      if (typeof question.groundTruth === 'string') {
        question.groundTruth = [question.groundTruth];
      }
      
      // groundTruth 배열이 없으면 생성
      if (!Array.isArray(question.groundTruth)) {
        question.groundTruth = [];
      }
      
      // 새로운 groundTruth 생성
      const newGroundTruth = await generateGroundTruth(question.question);
      
      if (newGroundTruth) {
        // 배열에 추가
        question.groundTruth.push(newGroundTruth);
        console.log(`✅ Ground Truth 추가됨 (총 ${question.groundTruth.length}개)`);
      } else {
        console.log('❌ Ground Truth 생성 실패');
      }
      
      // API 호출 간격 조절 (서버 부하 방지)
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  // 업데이트된 데이터를 파일에 저장
  writeEvalFile(evalData);
  
  console.log('\n=== Ground Truth 생성 완료 ===');
  console.log('생성된 groundTruth는 eval.json 파일에 저장되었습니다.');
}

// 스크립트 실행
main().catch(console.error); 