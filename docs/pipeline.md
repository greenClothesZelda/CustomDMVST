# Pipeline

## 개요

현재 파이프라인은 ST-ResNet 단일 경로로 동작합니다.

1. `MyDataset`
   각 target 시점 `t`에 대해 closeness, period, trend keyframe을 모아 `(L_total, H, W)` 입력을 만듭니다.
2. `STResNet`
   세 branch의 residual CNN 출력을 parametric fusion으로 합치고, weather/time external feature를 map으로 투영해 더합니다.
3. `ModelTrainer`
   Hugging Face `Trainer`와 맞추기 위해 `loss`와 `predictions`를 dict로 반환합니다.

최종 출력은 `(B, 169)` shape의 다음 시점 full-grid 수요 예측입니다.

## 데이터 흐름

### 1. Dataset

실데이터 기준:

- `grid(7000).npy`: `(4368, 13, 13)`
- `meteorological_data.csv`: 4368행 hourly weather row

`MyDataset`은 sample index `i`에 대해 `target_idx = i + base_offset`를 잡고 다음을 반환합니다.

- `demands_series`: `(L_total, 13, 13)`
- `labels`: `origin_demand_arr[target_idx]`, shape `(169,)`
- `time`: target 시점의 시간 feature, shape `(8,)`
- `weather`: target 시점의 정규화된 기상 feature, shape `(4,)`
- `sample_idx`: dataset-local index, scalar

frame 순서는 아래와 같습니다.

1. closeness: `t-1, t-2, ..., t-len_c`
2. period: `t-period_interval, t-2*period_interval, ...`
3. trend: `t-trend_interval, t-2*trend_interval, ...`

### 2. Base Offset

`base_offset`은 keyframe이 모두 존재하도록 하는 최소 시작점입니다.

```text
base_offset = max(len_c, len_p * period_interval, len_t * trend_interval)
```

현재 기본 설정에서는:

- `len_c = 3`
- `len_p = 1`
- `len_t = 1`
- `period_interval = 24`
- `trend_interval = 168`

이므로 `base_offset = 168`입니다.

### 3. Split

`main.py`는 `len(dataset)`가 이미 `base_offset`을 반영한 상태라고 가정하고 단순 split을 적용합니다.

- `train_end = int(len(dataset) * split.train_ratio)`
- train: `[0, train_end)`
- test: `[train_end, len(dataset))`

weather normalization도 같은 train split 기준 통계로 계산됩니다.

## 모델 구조

### `STResNet`

입력:

- `demands_series`: `(B, L_total, 13, 13)`
- `weather`: `(B, 4)`
- `time`: `(B, 8)`
- `sample_idx`: `(B,)`, 현재 미사용

구성:

1. closeness branch
2. period branch
3. trend branch
4. parametric fusion weight `Wc`, `Wp`, `Wt`
5. external FC

각 branch는 다음 순서를 따릅니다.

1. `Conv2d`
2. residual unit stack
3. `Conv2d`

fusion 결과에 external map을 더한 뒤 `(B, 1, 13, 13)`을 flatten해서 `(B, 169)`를 반환합니다.

### `ModelTrainer`

역할:

- `DMVSTLoss` 계산
- 반환용 `predictions`에만 `ReLU` 적용
- `Trainer`가 읽을 수 있는 `{'predictions': ..., 'loss': ...}` dict 생성

loss는 raw output 기준으로 계산하고, 평가/저장용 prediction은 비음수로 정리합니다.

## 설정 인자 설명

기준 파일은 `configs/stresnet_base.yaml`입니다.

### `dataset`

`root`
: 원시 데이터 경로

`size`
: 불러올 `grid(size).npy`의 size 값

### `split`

`train_ratio`
: `base_offset`이 반영된 dataset length 기준 train 종료 비율

### `model.STResNet`

`nb_flow`
: 현재 구현은 `1`만 지원합니다.

`len_c`
: closeness frame 수

`len_p`
: period frame 수

`len_t`
: trend frame 수

`period_interval`
: period 간격

`trend_interval`
: trend 간격

`nb_filter`
: branch 내부 convolution 채널 수

`nb_residual_unit`
: branch별 residual unit 수

`use_external`
: weather/time external component 사용 여부

`external_hidden`
: external FC hidden 차원

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
: best checkpoint를 고를 metric입니다. 기본값은 `rmse`입니다.

`greater_is_better`
: `rmse`는 낮을수록 좋기 때문에 `false`를 사용합니다.

## 평가와 결과 파일

`runners/test.py`는 full-grid prediction과 label을 직접 비교합니다.

- `MAE`
- `RMSE`
- `MAPE`

산출물:

- `test_results.csv`
- `demand_error_analysis.png`
- `predictions_max_demand_node.png`
- `predictions_min_demand_node.png`
- `predictions_mid_demand_node.png`

## 실행 체크리스트

- `grid(size).npy`와 `meteorological_data.csv`가 `dataset.root` 아래에 있는지
- `remove_unused_columns=false`가 유지되는지
- CPU 실행 시 `device=cpu`와 `+train.use_cpu=true`를 같이 주는지
- 출력 경로가 현재 환경에서 writable 한지, 아니면 `hydra.run.dir=/tmp/...`로 override할지
