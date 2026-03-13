example usage(single run):
base config 적용
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config-name config 
```
config override 적용
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config-name config train.epochs=10
```