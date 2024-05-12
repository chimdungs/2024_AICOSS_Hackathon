# 2023~2024 UOS AICOSS Hackathon 
![image](https://github.com/chimdungs/2024_AICOSS_Hackathon/assets/138076274/bdad8017-acf8-45d8-851b-ea7e0a11ac46)

## Hyperparmaeter Setting
### -model 1
```python
CFG = {
    'IMG_SIZE':256, # (fixed for model 1~5)
    'EPOCHS':12, #(12~30) various for mode 1~5
    'LEARNING_RATE':1e-3, # default = 3e-4(0.0003)
    'BATCH_SIZE':32,
    'SEED':42,
    'BETA':1.0 #cutmix hyperparameter
}
```

## Preprocessing

1. RandomHorizontal Flip

2. Cutmix
   - cutmix probability setting : 0.7
```python
def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율
    cut_w = int(W * cut_rat)  # 패치의 너비
    cut_h = int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값 추출(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
```

```python
X,y = next(iter(train_loader))
X = X.to(device)

lam = np.random.beta(1.0, 1.0)  # 베타 분포에서 lam 값을 가져옴
rand_index = torch.randperm(X.size()[0]).to(device) # batch_size 내의 인덱스가 랜덤하게 셔플

print(lam)
print(rand_index)

```
```python
import matplotlib.pyplot as plt
16]
def cutmix_plot(train_loader):
    fig , axes = plt.subplots(1,3)
    fig.set_size_inches(15,12)
    
    for i in range(3):
        for inputs, targets in train_loader:
            inputs = inputs
            targets = targets
            break

        lam = np.random.beta(1.0, 1.0) 
        rand_index = torch.randperm(inputs.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        axes[i].imshow(inputs[1].permute(1, 2, 0).cpu())
        axes[i].set_title(f'λ : {np.round(lam,3)}')
        axes[i].axis('off')
    return

cutmix_plot(train_loader)
```
-> cutmix result example
![image](https://github.com/chimdungs/2024_AICOSS_Hackathon/assets/138076274/453dfb72-2d12-446b-b61d-19a298bb5b08)


3. RandomAffine
4. Resize & Normalization

## Loss
### Baseline = BCELoss
### Focal Loss
  caused by data imbalance, I chose the focal loss.
  - setting: alpha = 0.25, gamma = 2, mean reduction
- Implementation
```python
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,#default = 2
                 temperature = 1.0,
                 reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss()
        self.temperature = temperature

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        logits = logits / self.temperature
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

  ```
    
## Etc.
### Temperature
- for label smoothing effection, use logit / temperature

## Inference
