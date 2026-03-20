# Pipeline

## 개요

이 브랜치의 모델은 두 경로를 섞습니다.

1. `IRModule`
   각 sample보다 이전 시점의 `(N, T)` demand window 전체를 검색해 cosine similarity 기반 retrieval prediction을 만듭니다.
2. `IRVSTNet`
   demand, node id, weather, time을 임베딩하고 spatial attention + temporal BiLSTM으로 neural prediction을 만듭니다.

최종 출력은 두 prediction을 gate로 섞은 `(B, N)` 다음 시점 수요 예측입니다.

## 데이터 흐름

### 1. Dataset

`MyDataset`은 sample index `i`에 대해 다음 값을 반환합니다.

- `demands_series`: `demand_arr[i : i + time_step]`를 `(N, T)`로 전치한 값
- `labels`: `demand_arr[i + time_step]`, shape `(N,)`
- `time`: 다음 시점의 시간 feature, shape `(8,)`
- `weather`: 다음 시점의 기상 feature, shape `(F,)`
- `sample_idx`: 현재 sample의 절대 dataset index, scalar

`sample_idx`는 retriever가 검색 가능한 prefix를 `0..sample_idx-1`로 제한하는 데 사용합니다.

### 2. Split

`main.py`는 full timeline 기준으로 다음처럼 나눕니다.

- `warmup = model.IRModule.k`
- `train_end = int(len(dataset) * split.train_ratio)`
- retrieval-only prefix: `[0, warmup)`
- train: `[warmup, train_end)`
- test: `[train_end, len(dataset))`

즉 초반 `warmup` 구간은 학습에는 들어가지 않고 검색 candidate로만 남습니다.

### 3. Retrieval

`IRModule`은 full dataset 전체로 아래 DB를 만듭니다.

- `db_keys`: shape `(N, Samples, T)`
- `db_values`: shape `(N, Samples, 1)`
- `db_norms`: cosine normalization용 norm

query batch가 들어오면 batch 내부 각 sample마다 개별적으로 다음을 수행합니다.

1. `candidate_count = sample_idx`
2. prefix candidate를 `db[:, :candidate_count]`로 자름
3. 각 node에 대해 cosine similarity 계산
4. `top_k = min(k, candidate_count)` 적용
5. top-k label을 softmax weight로 가중합

retrieval output shape은 `(B, N)`입니다.

### 4. Neural Forecasting

`IRVSTNet`은 다음 feature를 더해 `(B, N, T, D)`를 만듭니다.

- demand embedding
- node embedding
- weather embedding
- time embedding

그 후:

1. 각 time slice에서 node 간 self-attention
2. 각 node별 temporal BiLSTM
3. linear projection으로 neural output `(B, N)` 생성

### 5. Fusion

`lambda_layer`가 node별 gate를 예측합니다.

- neural output: `out`
- retrieval output: `ir_out`
- final output: `lambda * out + (1 - lambda) * ir_out`

## 주요 모듈 설명

### `IRModule`

역할:

- causal retrieval
- query보다 이전 시간만 검색
- raw demand cosine similarity 사용

입력:

- `query_demands`: `(B, N, T)`
- `sample_idx`: `(B,)`

출력:

- `aggregated`: `(B, N)`
- `retrieved_indices`: `(B, N, k)`

`retrieved_indices`는 absolute dataset index가 아니라 prefix 내부 index입니다. 현재 구현에서는 항상 현재 `sample_idx`보다 작은 위치만 반환됩니다.

### `IRVSTNet`

역할:

- retrieval branch와 neural branch를 합친 forecasting model

입력:

- `demands_series`: `(B, N, T)`
- `weather`: `(B, F)`
- `time`: `(B, 8)`
- `sample_idx`: `(B,)`

출력:

- next-step demand prediction `(B, N)`

### `ModelTrainer`

역할:

- Hugging Face `Trainer`와 맞추기 위한 wrapper
- `L1Loss`로 training loss 계산

## 설정 인자 설명

기준 파일은 `configs/cosine_base_ir.yaml`입니다.

### `dataset`

`root`
: 원시 데이터 경로

`time_step`
: 입력 시계열 길이 `T`

`num_nodes`
: 총 demand 기준 상위 몇 개 grid cell을 사용할지

`size`
: 불러올 `grid(size).npy`의 size 값

### `split`

`train_ratio`
: full timeline 기준 train 종료 비율

### `model.IRModule`

`k`
: retrieval top-k 크기입니다. 동시에 warmup prefix 길이도 이 값으로 사용됩니다.

### `model.IRVSTNet`

`embedding_dim`
: demand/node/weather/time 공통 embedding 차원

`lstm.hidden_size`
: bidirectional LSTM의 hidden size

`lstm.num_layers`
: LSTM layer 수

`lstm.dropout`
: LSTM/attention dropout

`lstm.nhead`
: spatial attention의 head 수

### `train`

`per_device_train_batch_size`
: train batch size

`per_device_eval_batch_size`
: eval batch size

`num_train_epochs`
: 최대 epoch 수

`learning_rate`
: optimizer learning rate

`weight_decay`
: weight decay

`warmup_ratio`
: scheduler warmup 비율

`logging_steps`
: training log 주기

`eval_strategy`
: evaluation 주기

`save_strategy`
: checkpoint 저장 주기

`load_best_model_at_end`
: best eval checkpoint를 마지막에 다시 로드할지

`metric_for_best_model`
: best checkpoint를 고를 metric입니다. 현재는 `mape`를 사용합니다.

`greater_is_better`
: `mape`는 낮을수록 좋기 때문에 `false`를 사용합니다.

## 실행 체크리스트

학습 전에 최소한 다음을 확인하면 됩니다.

- `grid(size).npy`와 `meteorological_data.csv`가 `dataset.root` 아래에 있는지
- `model.IRModule.k < len(dataset) * split.train_ratio`인지
- GPU를 쓰면 `device=cuda`, CPU를 쓰면 `device=cpu`와 `+train.use_cpu=true`를 같이 주는지
- `remove_unused_columns=false`가 유지되는지

## 결과 파일

학습이 끝나면 Hydra output dir 아래에 다음이 생성됩니다.

- Trainer checkpoint
- `test_results.csv`
- 로그 및 설정 스냅샷
