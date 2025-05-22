sentences = ["이 영화 정말 재미있어요", "스토리가 너무 지루했어요"]  # 학습할 문장 리스트와  
labels    = [1, 0]                                       # 각 문장에 대한 레이블(1=긍정, 0=부정)  

vocab   = list(set(" ".join(sentences).split()))        # 모든 문장을 합쳐 단어별로 분리한 후 중복 제거  
weights = [0] * len(vocab)                              # 각 단어에 대응하는 가중치를 0으로 초기화  
bias    = 0                                             # 편향을 0으로 초기화  
lr      = 0.1                                           # 학습률 설정  

for _ in range(5):                                      # 에폭(epoch) 5회 반복  
    for sent, label in zip(sentences, labels):         # 문장과 레이블을 짝지어 순회  
        x     = [sent.count(w) for w in vocab]         # 현재 문장의 단어 빈도 벡터 생성  
        score = sum(w * xi for w, xi in zip(weights, x)) + bias  # 가중합에 편향 더해 점수 계산  
        pred  = 1 if score > 0 else 0                  # 점수가 양수면 긍정, 아니면 부정 예측  
        error = label - pred                           # 실제값과 예측값 차이로 오차 계산  
        weights = [w + lr * error * xi for w, xi in zip(weights, x)]  # 가중치 갱신  
        bias   += lr * error                           # 편향 갱신  

tests = ["이 영화 재미없어요", "정말 최고예요"]         # 테스트할 새 문장 리스트  
for t in tests:                                        # 각 테스트 문장에 대해  
    x_test = [t.count(w) for w in vocab]               # 단어 빈도 벡터 생성  
    score  = sum(w * xi for w, xi in zip(weights, x_test)) + bias  # 점수 계산  
    result = "긍정 😊" if score > 0 else "부정 😢"        # 결과 판정  
    print(f"{t} → {result}")                           # 결과 출력  
