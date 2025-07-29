import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// eval.json 파일 경로
const evalFilePath = path.join(__dirname, '../public/eval.json');

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
function main() {
  console.log('=== Ground Truth 0번 인덱스 삭제 시작 ===');
  console.log('');

  // eval.json 파일 읽기
  const evalData = readEvalFile();
  if (!evalData) {
    console.error('eval.json 파일을 읽을 수 없습니다.');
    return;
  }

  let totalRemoved = 0;

  // 각 카테고리별로 0번 인덱스 삭제
  for (const category of evalData.evaluationQuestions) {
    console.log(`\n--- ${category.categoryName} 카테고리 처리 중 ---`);
    
    for (const question of category.questions) {
      if (Array.isArray(question.groundTruth) && question.groundTruth.length > 0) {
        const removedItem = question.groundTruth.shift(); // 0번 인덱스 제거
        console.log(`질문: ${question.question.substring(0, 50)}...`);
        console.log(`삭제된 항목: ${removedItem.substring(0, 50)}...`);
        console.log(`✅ 0번 인덱스 삭제됨 (남은 개수: ${question.groundTruth.length}개)`);
        totalRemoved++;
      }
    }
  }

  // 업데이트된 데이터를 파일에 저장
  writeEvalFile(evalData);
  
  console.log('\n=== Ground Truth 0번 인덱스 삭제 완료 ===');
  console.log(`총 ${totalRemoved}개의 0번 인덱스 항목이 삭제되었습니다.`);
}

// 스크립트 실행
main().catch(console.error); 