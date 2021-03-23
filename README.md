# involution.pytorch ([内卷](https://zh.wikipedia.org/wiki/%E5%86%85%E5%8D%B7%E5%8C%96))
A PyTorch implementation of Involution using [einops](https://github.com/arogozhnikov/einops)

An unofficial pytorch implementation of involution [paper](https://arxiv.org/pdf/2103.06255.pdf). Official implementation can be found [here](https://github.com/d-li14/involution).

<img src="https://github.com/shuuchen/involution.pytorch/blob/main/images/invo.png" width="480" height="220" />


## Requirements
```
pytorch >= 1.4.0
einops >= 0.3.0
```

## Usage
* An example:
```shell
>>> import torch
>>> from involution import Involution
>>>
>>> x = torch.rand(2,8,5,5)
>>> i = Involution(8)
>>> i(x).size()
torch.Size([2, 8, 5, 5])
```

## TODO
- [ ] ImageNet training & performance checking


## References
- [official implementation](https://github.com/d-li14/involution)
- [original paper](https://arxiv.org/pdf/2103.06255.pdf)
- [einops](https://github.com/arogozhnikov/einops)
- [利用Pytorch实现卷积操作](https://zhuanlan.zhihu.com/p/349683405)
