# CustomDMVST

현재 브랜치는 `cosine_base_ir` 경로를 기준으로 동작합니다. retriever는 학습하지 않고, 각 query sample보다 이전 시점의 raw demand window만 cosine similarity로 검색합니다.

## 실행 방법

기본 학습:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config-name cosine_base_ir
```

CPU 스모크 테스트:

```bash
python main.py --config-name cosine_base_ir \
  device=cpu \
  +train.use_cpu=true \
  +train.max_steps=1 \
  train.eval_strategy=steps \
  +train.eval_steps=1 \
  train.save_strategy=steps \
  +train.save_steps=1 \
  train.logging_steps=1
```

## 자주 바꾸는 인자

`model.IRModule.k`
: retrieval top-k 크기입니다. 동시에 warmup 길이도 이 값으로 결정됩니다.

`split.train_ratio`
: 전체 timeline에서 train이 끝나는 위치입니다. train sample은 `[k, train_end)`, test sample은 `[train_end, len(dataset))`를 사용합니다.

`dataset.time_step`
: 각 sample의 입력 시계열 길이입니다.

`dataset.num_nodes`
: 총 수요량 기준 상위 몇 개 grid cell을 사용할지 정합니다.

`train.per_device_train_batch_size`
: forecasting model 학습 batch size입니다.

## 출력

학습 결과는 Hydra 출력 디렉터리에 저장됩니다.

- checkpoint: `outputs/cosine_base_ir/...`
- test prediction csv: `test_results.csv`

## 문서

모델 구조와 각 인자 설명은 [docs/pipeline.md](/home/jinsu/PycharmProjects/CustomDMVST/docs/pipeline.md)에 정리했습니다.
