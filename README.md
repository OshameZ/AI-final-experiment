# 多模态情感分析

这是一个多模态情感分析实验的示例项目，使用 BERT 对文本进行特征提取、ResNet 对图像进行特征提取，并对情感进行三分类（positive / neutral / negative）。

## 实验要求
- 设计一个多模态融合模型。
- 自行划分训练集和验证集，并调整超参数。
- 预测 `test_without_label.txt` 上的情感标签。
- 提交完整的代码、报告和测试结果。
- 代码需可执行，并保证结果可复现。

## 环境要求
建议使用Python3.11及以上版本。
所有依赖项已列在 `requirements.txt` 文件中，可使用以下命令安装：
```bash
pip install -r requirements.txt
```
## 代码文件结构
```bash
|-- data/
|   |-- 123.txt
|   |-- 123.jpg
|   |-- ...           # 其他 .txt 与 .jpg 文件
|-- train.txt
|-- test_without_label.txt
|-- multimodal.py
|-- requirements.txt
|-- README.md
```
- data/：用于存放 .txt（文本）和 .jpg（图像），文件以 guid 命名，供训练和测试读取
- train.txt：训练集列表文件，形如 guid,tag
- test_without_label.txt：测试集列表文件，形如 guid,null
- multimodal.py：项目的核心代码文件，包括模型、训练、验证、测试等逻辑
- requirements.txt：列出项目所需的依赖包
- README.md：当前说明文件


## 使用流程
1.准备数据

- 在 data/ 文件夹内放置所有 .txt 和 .jpg 文件，并确保其命名（形如 123.txt / 123.jpg）与 train.txt / test_without_label.txt 中的 guid 相对应。并确保train.txt与test_without_label.txt与py文件在同一目录下

2.安装依赖

```bash

pip install -r requirements.txt
```
如果缺少 CUDA 或存在版本冲突，可根据自己的环境相应修改 requirements.txt 中的版本号。

3.训练与验证

不带任何额外参数的运行方式（脚本默认超参数）：
```bash

python multimodal.py
```
代码会自动读取 train.txt，随机划分验证集并输出训练过程的损失与验证集准确率等信息。

4.自定义超参数

可以通过命令行传参的方式覆盖默认值。示例：
```bash

python multimodal.py \
  --train_file train.txt \
  --test_file test_without_label.txt \
  --data_dir data \
  --text_model_name bert-base-chinese \
  --batch_size 16 \
  --lr 2e-5 \
  --epochs 5 \
  --use_text \
  --use_image
  ```
常见可选参数包括：
- train_file / --test_file / --data_dir：数据路径配置
- batch_size：batch 大小
- lr：学习率
- epochs：训练轮数
- use_text / --use_image：是否使用文本或图像特征
- skip_test：是否跳过测试集推理
- 可以查看或修改 multimodal.py 中 ArgumentParser 的部分来调整所有可选参数。

5.测试集预测

- 默认情况下（未加 --skip_test），脚本在训练结束后会载入最优模型参数，对 test_without_label.txt 做推理，并将预测结果保存在 test_prediction.txt 中。
- 如果只想查看验证集效果或做消融实验，可在运行脚本时添加 --skip_test 参数，跳过测试集推理，避免生成新的预测文件：
```bash

python multimodal.py --skip_test
```
6.消融实验

仅使用文本特征：
```bash
python multimodal.py --use_text
```
仅使用图像特征：
```bash
python multimodal.py --use_image
```
同时使用文本与图像（完整多模态）：
```bash
python multimodal.py --use_text --use_image
```
通过比较在验证集上的准确率，即可完成消融实验的性能对比。


## 参考
- PyTorch
- torchvision
- HuggingFace Transformers

