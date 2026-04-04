# CustomDMVST

현재 기본 경로는 `stresnet_base` 설정을 사용하는 ST-ResNet 학습 파이프라인입니다. 모델은 `closeness/period/trend` keyframe, residual CNN branch, parametric fusion, weather/time external feature를 이용해 다음 시점 전체 13x13 수요 맵을 예측합니다.

## 실행 방법

기본 학습:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config-name stresnet_base
```

CPU 스모크 테스트:

```bash
python main.py --config-name stresnet_base \
  hydra.run.dir=/tmp/customdmvst_stresnet_smoke \
  device=cpu \
  +train.use_cpu=true \
  +train.max_steps=1 \
  train.eval_strategy=steps \
  +train.eval_steps=1 \
  train.save_strategy=steps \
  +train.save_steps=1 \
  train.logging_steps=1
```

`outputs/`가 심볼릭 링크이거나 현재 환경에서 쓰기 불가면 위처럼 `hydra.run.dir=/tmp/...`를 함께 주면 됩니다.

## 배치 계약

`demands_series`
: `(B, L_total, 13, 13)`. 기본값 기준 `L_total = len_c + len_p + len_t = 5`입니다.

`labels`
: `(B, 169)`. 예측 대상 시점의 전체 13x13 수요 맵을 flatten한 값입니다.

`weather`
: `(B, 4)`. train split 기준 mean/std로 정규화된 기상 feature입니다.

`time`
: `(B, 8)`. `요일 one-hot 7 + hour/24`입니다.

`sample_idx`
: `(B,)`. 디버깅용 dataset-local index이며 ST-ResNet 본체에서는 사용하지 않습니다.

## 자주 바꾸는 인자

`model.STResNet.len_c`
: closeness frame 개수입니다. 기본값은 최근 3개 시점입니다.

`model.STResNet.len_p`
: period frame 개수입니다. 기본값은 전날 같은 시간 1개입니다.

`model.STResNet.len_t`
: trend frame 개수입니다. 기본값은 지난주 같은 시간 1개입니다.

`model.STResNet.period_interval`
: period 간격입니다. 현재 hourly 데이터 기준 기본값은 `24`입니다.

`model.STResNet.trend_interval`
: trend 간격입니다. 현재 hourly 데이터 기준 기본값은 `168`입니다.

`model.STResNet.nb_filter`
: 각 branch의 convolution 채널 수입니다.

`model.STResNet.nb_residual_unit`
: branch별 residual unit 개수입니다.

`train.per_device_train_batch_size`
: 학습 batch size입니다.

## 출력

학습 결과는 Hydra 출력 디렉터리에 저장됩니다.

- checkpoint: `outputs/stresnet_base/...`
- test prediction csv: `test_results.csv`
- visualization png: `demand_error_analysis.png`, `predictions_*.png`

## 테스트

단위 테스트:

```bash
pytest -q
```

## 문서

모델 구조, 데이터셋 keyframe 규칙, 설정 인자 설명은 [docs/pipeline.md](/home/jinsu/PycharmProjects/CustomDMVST/docs/pipeline.md)에 정리했습니다.
