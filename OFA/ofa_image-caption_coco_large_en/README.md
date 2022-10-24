---
backbone:
- OFA
datasets:
  evaluation:
  - modelscope/coco_2014_caption
  test:
  - modelscope/coco_2014_caption
  train:
  - modelscope/coco_2014_caption
domain:
- multi-modal
frameworks:
- pytorch
indexing:
  results:
  - dataset:
      args: default
      name: MSCOCO Image Captioning
      type: Image-Text Pair
    metrics:
    - args: default
      description: CIDEr score
      type: CIDEr
      value: 154.9
    task:
      name: Image Captioning
license: Apache License 2.0
metrics:
- CIDEr
tags:
- Alibaba
- ICML2022
- arxiv:2202.03052
tasks:
- image-captioning

widgets:
  - task: image-captioning
    inputs:
      - name: image
        title: 图片
        type: image
        validator:
        max_resolution: 3000*3000
        max_size: 10M
    examples:
      - name: 1
        title: 示例1 
        inputs:
        - data: https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg
          name: image
      - name: 2
        title: 示例2
        inputs:
        - data: https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/vqa2.jpg
          name: image
      - name: 3
        title: 示例3
        inputs:
        - data: https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/vqa1.jpg
          name: image
      - name: 4
        title: 示例4
        inputs:
        - data: https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/vqa3.jpg
          name: image
      - name: 5
        title: 示例5
        inputs:
        - data: https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/vqa4.jpeg
          name: image
    inferencespec:
      cpu: 4
      gpu: 1
      gpu_memory: 16000
      memory: 43000
  
---
## News

- 2022年09月: 上线[Huge模型](https://modelscope.cn/models/damo/ofa_image-caption_coco_huge_en/summary)，欢迎试用。

# OFA-图像描述(英文)


## 图像描述是什么？
如果你希望为一张图片配上一句文字，或者打个标签，OFA模型就是你的绝佳选择。你只需要输入任意1张你的图片，**3秒内**就能收获一段精准的描述。**本页面右侧**提供了在线体验的服务，欢迎使用！

注：本模型为OFA-图像描述的**Large**模型，参数量约为4.7亿。还有[Huge模型](https://modelscope.cn/models/damo/ofa_image-caption_coco_huge_en/summary)可以试用。
<br><br>

## 快速玩起来
玩转OFA只需区区以下6行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角`Notebook`按钮，我们为你提供了配备了GPU的环境，你只需要在notebook里输入提供的代码，就可以把OFA玩起来了！

<p align="center">
    <img src="resources/donuts.jpg" alt="donuts" width="200" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_large_en')
result = img_captioning({'image': 'https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg'})
print(result[OutputKeys.CAPTION]) # 'a bunch of donuts on a wooden board with popsicle sticks'
```
<br>

## OFA是什么？
OFA(One-For-All)是通用多模态预训练模型，使用简单的序列到序列的学习框架统一模态（跨模态、视觉、语言等模态）和任务（如图片生成、视觉定位、图片描述、图片分类、文本生成等），详见我们发表于ICML 2022的论文：[OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)，以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA](https://github.com/OFA-Sys/OFA)。

<p align="center">
    <br>
    <img src="resources/OFA_logo_tp_path.svg" width="150" />
    <br>
<p>
<br>

<p align="center">
        <a href="https://github.com/OFA-Sys/OFA">Github</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2202.03052">Paper </a>&nbsp ｜ &nbspBlog
</p>

<p align="center">
    <br>
        <video src="https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/resources/modelscope_web/demo.mp4" loop="loop" autoplay="autoplay" muted width="100%"></video>
    <br>
</p>


## 为什么OFA是图像描述的最佳选择？
OFA在图像描述（image captioning）任务的权威数据集Microsoft COCO Captions官方榜单成功登顶（想看榜单，点[这里](https://competitions.codalab.org/competitions/3221#results)），并在经典测试集Karpathy test split取得CIDEr 154.9的分数。具体如下：

<table border="1" width="100%">
    <tr align="center">
        <th>Stage</th><th colspan="4">Cross Entropy Optimization</th><th colspan="4">CIDEr Optimization</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td><td>SPICE</td><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td><td>SPICE</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub></td><td>41.0</td><td>30.9</td><td>138.2</td><td>24.2</td><td>42.8</td><td>31.7</td><td>146.7</td><td>25.8</td>
    </tr>
    <tr align="center">
        <td><b>OFA<sub>Large</sub></b></td><td>42.4</td><td>31.5</td><td>142.2</td><td>24.5</td><td>43.6</td><td>32.2</td><td>150.7</td><td>26.2</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Huge</sub></td><td>43.9</td><td>31.8</td><td>145.3</td><td>24.8</td><td>44.9</td><td>32.5</td><td>154.9</td><td>26.6</td>
    </tr>
</table>
<br>

## 相关论文以及引用
如果你觉得OFA好用，喜欢我们的工作，欢迎引用：
```
@article{wang2022ofa,
  author    = {Peng Wang and
               An Yang and
               Rui Men and
               Junyang Lin and
               Shuai Bai and
               Zhikang Li and
               Jianxin Ma and
               Chang Zhou and
               Jingren Zhou and
               Hongxia Yang},
  title     = {OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence
               Learning Framework},
  journal   = {CoRR},
  volume    = {abs/2202.03052},
  year      = {2022}
}
```