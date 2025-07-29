from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
EVAL_JSON_PATH = PROJECT_ROOT / "ollama-chat" / "public" / "eval.json"

# 한국어 토크나이저 초기화
try:
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    print("토크나이저 로드 완료: klue/bert-base")
except Exception as e:
    print(f"토크나이저 로드 실패: {e}")
    tokenizer = None

def tokenize_text(text):
    """텍스트를 토크나이저를 사용하여 토큰화"""
    if tokenizer is None:
        # 토크나이저가 없으면 간단한 토큰화
        return text.lower().split()
    
    try:
        tokens = tokenizer.tokenize(text)
        return tokens
    except Exception as e:
        print(f"토큰화 오류: {e}")
        return text.lower().split()

def calculate_bleu_score(candidate, reference, max_n=4):
    """BLEU 점수 계산 (토크나이저 사용)"""
    if not candidate or not reference:
        return 0.0
    
    candidate_tokens = tokenize_text(candidate)
    reference_tokens = tokenize_text(reference)
    
    if not candidate_tokens or not reference_tokens:
        return 0.0
    
    # n-gram 정밀도 계산
    precisions = []
    for n in range(1, max_n + 1):
        if len(candidate_tokens) < n:
            precisions.append(0.0)
            continue
            
        # n-gram 생성
        candidate_ngrams = []
        for i in range(len(candidate_tokens) - n + 1):
            candidate_ngrams.append(' '.join(candidate_tokens[i:i+n]))
        
        reference_ngrams = []
        for i in range(len(reference_tokens) - n + 1):
            reference_ngrams.append(' '.join(reference_tokens[i:i+n]))
        
        # 매칭 계산
        matches = 0
        total = len(candidate_ngrams)
        
        candidate_ngram_counts = {}
        reference_ngram_counts = {}
        
        for ngram in candidate_ngrams:
            candidate_ngram_counts[ngram] = candidate_ngram_counts.get(ngram, 0) + 1
        
        for ngram in reference_ngrams:
            reference_ngram_counts[ngram] = reference_ngram_counts.get(ngram, 0) + 1
        
        for ngram, count in candidate_ngram_counts.items():
            ref_count = reference_ngram_counts.get(ngram, 0)
            matches += min(count, ref_count)
        
        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)
    
    # 기하평균 계산
    geometric_mean = np.prod(precisions) ** (1.0 / len(precisions))
    
    # 짧은 문장 페널티 (Brevity Penalty)
    bp = 1.0
    if len(candidate_tokens) < len(reference_tokens):
        bp = np.exp(1 - len(reference_tokens) / len(candidate_tokens))
    
    # BLEU 점수 계산 (0-100 스케일)
    bleu_score = bp * geometric_mean * 100
    
    return round(bleu_score, 2)

def calculate_rouge_score(candidate, reference):
    """ROUGE 점수 계산"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        # ROUGE-1, ROUGE-2, ROUGE-L의 평균 (F1 점수 사용)
        rouge1_f1 = scores['rouge1'].fmeasure
        rouge2_f1 = scores['rouge2'].fmeasure
        rougeL_f1 = scores['rougeL'].fmeasure
        
        # 0-100 스케일로 변환
        rouge_score = ((rouge1_f1 + rouge2_f1 + rougeL_f1) / 3) * 100
        return round(rouge_score, 2)
    except Exception as e:
        print(f"ROUGE 점수 계산 오류: {e}")
        return 0.0

def calculate_meteor_score(candidate, reference):
    """METEOR 점수 계산 (한국어 지원)"""
    try:
        # 한국어 토크나이저 사용
        candidate_tokens = tokenize_text(candidate.lower())
        reference_tokens = tokenize_text(reference.lower())
        
        # 정확도 (Precision)
        matches = 0
        for token in candidate_tokens:
            if token in reference_tokens:
                matches += 1
        
        precision = matches / len(candidate_tokens) if candidate_tokens else 0
        
        # 재현율 (Recall)
        recall = matches / len(reference_tokens) if reference_tokens else 0
        
        # F1 점수
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 한국어 동의어 매칭 (간단한 구현)
        synonym_matches = 0
        korean_synonyms = {
            '만들다': ['개발하다', '제작하다', '생성하다'],
            '소유자': ['주인', '소유주', '보유자'],
            '모델': ['모형', '시스템'],
            '언어': ['말', '어'],
            '훈련': ['학습', '교육'],
            '공개': ['오픈', '공개적'],
            '사용': ['이용', '활용'],
            '가능': ['할 수 있다', '가능하다'],
            '누구나': ['모든 사람', '어느 누구나'],
            '딥마인드': ['DeepMind', '딥마인드'],
            '구글': ['Google', '구글'],
            'Gemma': ['gemma', '제마']
        }
        
        for cand_token in candidate_tokens:
            for ref_token in reference_tokens:
                # 직접 매칭
                if cand_token == ref_token:
                    synonym_matches += 1
                    break
                # 동의어 매칭
                elif cand_token in korean_synonyms and ref_token in korean_synonyms[cand_token]:
                    synonym_matches += 1
                    break
                elif ref_token in korean_synonyms and cand_token in korean_synonyms[ref_token]:
                    synonym_matches += 1
                    break
        
        # METEOR 점수 계산 (F1 + 동의어 보너스)
        meteor_score = f1 + (synonym_matches * 0.05)  # 동의어 보너스 감소
        meteor_score = min(meteor_score, 1.0)  # 최대 1.0으로 제한
        
        # 0-100 스케일로 변환
        return round(meteor_score * 100, 2)
    except Exception as e:
        print(f"METEOR 점수 계산 오류: {e}")
        return 0.0

def calculate_bert_score(candidate, reference):
    """BERTScore 계산"""
    try:
        # BERTScore 계산 (한국어 모델 사용)
        P, R, F1 = bert_score([candidate], [reference], lang='ko', verbose=False)
        
        # F1 점수를 0-100 스케일로 변환
        bert_score_value = F1.item() * 100
        return round(bert_score_value, 2)
    except Exception as e:
        print(f"BERTScore 계산 오류: {e}")
        return 0.0

def evaluate_question_answer_fit(question, answer, ground_truth_list):
    """질문-답변 부합도 평가 (다중 ground truth 지원) - 다중 알고리즘 사용"""
    if not isinstance(ground_truth_list, list):
        ground_truth_list = [ground_truth_list]
    
    # 1. 모든 ground truth와의 각종 점수 계산
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    bert_scores = []
    
    for gt in ground_truth_list:
        bleu_score = calculate_bleu_score(answer, gt)
        rouge_score = calculate_rouge_score(answer, gt)
        meteor_score = calculate_meteor_score(answer, gt)
        bert_score = calculate_bert_score(answer, gt)
        
        bleu_scores.append(bleu_score)
        rouge_scores.append(rouge_score)
        meteor_scores.append(meteor_score)
        bert_scores.append(bert_score)
    
    # 최고 점수와 평균 점수 계산
    max_bleu_score = max(bleu_scores) if bleu_scores else 0.0
    max_rouge_score = max(rouge_scores) if rouge_scores else 0.0
    max_meteor_score = max(meteor_scores) if meteor_scores else 0.0
    max_bert_score = max(bert_scores) if bert_scores else 0.0
    
    avg_bleu_score = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_rouge_score = np.mean(rouge_scores) if rouge_scores else 0.0
    avg_meteor_score = np.mean(meteor_scores) if meteor_scores else 0.0
    avg_bert_score = np.mean(bert_scores) if bert_scores else 0.0
    
    # 2. 질문 키워드가 답변에 포함되는지 확인
    question_tokens = tokenize_text(question)
    answer_tokens = tokenize_text(answer)
    
    # 조사와 문장부호 제거
    stop_words = ['가', '는', '을', '를', '의', '에', '에서', '로', '와', '과', '이', '야', '니', '어', '아', '?', '!', '.', ',']
    clean_question_tokens = [token for token in question_tokens if token not in stop_words and len(token) > 1]
    clean_answer_tokens = [token for token in answer_tokens if token not in stop_words and len(token) > 1]
    
    keyword_match_count = 0
    for q_token in clean_question_tokens:
        for a_token in clean_answer_tokens:
            if q_token == a_token or q_token in a_token or a_token in q_token:
                keyword_match_count += 1
                break
    
    keyword_match_rate = keyword_match_count / len(clean_question_tokens) if clean_question_tokens else 0.0
    
    # 3. 종합 점수 계산 (각 알고리즘 동일 가중치 20%, 키워드 매칭 20%)
    algorithm_score = (max_bleu_score + max_rouge_score + max_meteor_score + max_bert_score) / 4
    keyword_score = keyword_match_rate * 100
    
    final_score = (algorithm_score * 0.8) + (keyword_score * 0.2)
    
    return {
        'bleuScore': round(max_bleu_score, 2),
        'rougeScore': round(max_rouge_score, 2),
        'meteorScore': round(max_meteor_score, 2),
        'bertScore': round(max_bert_score, 2),
        'keywordMatchRate': round(keyword_match_rate * 100, 1),
        'finalScore': round(final_score, 2),
        'details': {
            'bleuScore': round(max_bleu_score, 2),
            'rougeScore': round(max_rouge_score, 2),
            'meteorScore': round(max_meteor_score, 2),
            'bertScore': round(max_bert_score, 2),
            'avgBleuScore': round(avg_bleu_score, 2),
            'avgRougeScore': round(avg_rouge_score, 2),
            'avgMeteorScore': round(avg_meteor_score, 2),
            'avgBertScore': round(avg_bert_score, 2),
            'keywordMatchCount': keyword_match_count,
            'totalKeywords': len(clean_question_tokens),
            'keywordMatchRate': round(keyword_match_rate * 100, 1),
            'groundTruthCount': len(ground_truth_list),
            'allBleuScores': [round(score, 2) for score in bleu_scores],
            'allRougeScores': [round(score, 2) for score in rouge_scores],
            'allMeteorScores': [round(score, 2) for score in meteor_scores],
            'allBertScores': [round(score, 2) for score in bert_scores]
        }
    }

@app.route('/api/update-eval', methods=['POST'])
def update_eval():
    try:
        data = request.json
        category_key = data.get('categoryKey')
        question_text = data.get('questionText')
        ground_truth = data.get('groundTruth')
        full_eval_data = data.get('fullEvalData')
        
        if not all([category_key, question_text, ground_truth, full_eval_data]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # eval.json 파일 업데이트
        with open(EVAL_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(full_eval_data, f, ensure_ascii=False, indent=2)
        
        print(f"eval.json 업데이트 완료: {category_key} - {question_text[:50]}...")
        
        return jsonify({
            'success': True,
            'message': f'Ground truth updated for {category_key}',
            'updated_question': question_text[:50] + '...' if len(question_text) > 50 else question_text
        })
        
    except Exception as e:
        print(f"eval.json 업데이트 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/finetune', methods=['POST'])
def finetune():
    # 기존 파인튜닝 API (필요시 구현)
    return jsonify({'success': True, 'message': 'Fine-tuning endpoint'})

@app.route('/api/finetune/status', methods=['GET'])
def finetune_status():
    # 기존 상태 조회 API (필요시 구현)
    return jsonify({'is_running': False, 'progress': 0, 'message': 'No fine-tuning in progress'})

@app.route('/api/evaluate', methods=['POST'])
def evaluate_response():
    """BLEU 평가 API"""
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        ground_truth = data.get('groundTruth')
        
        if not all([question, answer, ground_truth]):
            return jsonify({'error': 'Missing required fields: question, answer, groundTruth'}), 400
        
        # 평가 실행
        evaluation_result = evaluate_question_answer_fit(question, answer, ground_truth)
        
        return jsonify({
            'success': True,
            'evaluation': evaluation_result
        })
        
    except Exception as e:
        print(f"평가 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate/batch', methods=['POST'])
def evaluate_batch():
    """배치 평가 API"""
    try:
        data = request.json
        evaluations = data.get('evaluations', [])
        
        if not evaluations:
            return jsonify({'error': 'No evaluations provided'}), 400
        
        results = []
        for eval_item in evaluations:
            question = eval_item.get('question')
            answer = eval_item.get('answer')
            ground_truth = eval_item.get('groundTruth')
            
            if all([question, answer, ground_truth]):
                evaluation_result = evaluate_question_answer_fit(question, answer, ground_truth)
                results.append({
                    'question': question,
                    'answer': answer,
                    'evaluation': evaluation_result
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"배치 평가 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"API 서버 시작 - eval.json 경로: {EVAL_JSON_PATH}")
    app.run(host='0.0.0.0', port=5001, debug=True) 