## bp神经网反向传播

### 符号定义：

$X$: 输入矩阵，维度为$n\times t$ 。$n$ 为数据个数，$t$ 为特征向量维数。

$y^{m}$: 第$m$层神经元的输出矩阵。维度为$1 \times size(m)$

$y_{i}^{m}$: 第$m$层第$i$个神经元的输出值。

$\omega^{m}$: 第$m-1$ 层神经元到第$m$层神经元的权重。维度为 $size(m-1)\times size(m)$

$\omega_{ji}^{m}$: 第$m-1$ 层第$j$个神经元到第$m$层第$i$个神经元的权重

$\epsilon_{i}^{m}$: 第$m$层第$i$个神经元的线性组合。即$\epsilon _{i}^{m}= \sum _{j=1}^{size(m-1)}\omega_{ji}^{m}y_{j}^{m-1}+b_{i}^{m}$

$\epsilon^{m}$: 第$m$层神经元的线性组合。$1 \times size(m)$

$b^{m}$: 第$m$层神经元的偏置。维度为$1 \times size(m)$

$D$：神经元层数。标量

$l_{i}$: 输出层第$i$个神经元所造成的。维度为$1 \times size(D)$

$L$: 总损失。标量



$Act_{m}:$ 第$m$层神经元的激活函数

$Loss$: 损失函数

### 运算定义：

$y^{m} = Act_{m}(y^{m-1}·\omega^{m}+b^{m})$

$L = \sum_{i=1}^{size(D)}l_{i}=\sum_{i=1}^{size(D)}Loss(y_{i}^{D})$



### 反向传播推理：

$\omega_{kj}^{m}$: 第$m-1$ 层第$k$个神经元到第$m$层第$j$个神经元的权重
$$
\frac{\partial L}{\partial w_{kj}^{m}}=\frac{\partial L}{\partial \epsilon_{j}^{m}}\frac{\partial \epsilon}{\partial w_{kj}^{m}}=\frac{\partial L}{\partial \epsilon_{j}^{m}}y_{k}^{m-1}
$$
考虑第$m-1$ 层所有神经元到第$m$层第$j$个神经元的权重
$$
\frac{\partial L}{\partial w_{·j}^{m}}=\frac{\partial L}{\partial \epsilon_{j}^{m}}\frac{\partial \epsilon_{j}^{m}}{\partial w_{·j}^{m}}=\frac{\partial L}{\partial \epsilon_{j}^{m}}(y^{m-1})^{T}
$$
考虑矩阵
$$
d\omega^{m}=\frac{\partial L}{\partial w^{m}}=\frac{\partial L}{\partial \epsilon^{m}}\frac{\partial \epsilon^{m}}{\partial w^{m}}=(y^{m-1})^{T}\frac{\partial L}{\partial \epsilon^{m}}
$$
再来考虑 $\frac{\partial L}{\partial \epsilon^{m}}$，先考虑其中任意一个值
$$
\frac{\partial L}{\partial \epsilon_{j}^{m}} = \sum_{i=1}^{size(m+1)}\frac{\partial L}{\partial \epsilon_{i}^{m+1}}\frac{\partial \epsilon_{i}^{m+1}}{\partial y_{j}^{m}}\frac{\partial y_{j}^{m}}{\partial \epsilon_{j}^{m}}=\frac{\partial y_{j}^{m}}{\partial \epsilon_{j}^{m}}\sum_{i=1}^{size(m+1)}\frac{\partial L}{\partial \epsilon_{i}^{m+1}}\omega_{ji}^{m+1}=Act^{'}_{m}(\epsilon_{j}^{m})(\frac{\partial L}{\partial \epsilon^{m+1}} ·(\omega_{j·}^{m+1})^{T})
$$
考虑矩阵
$$
\frac{\partial L}{\partial \epsilon^{m}}=Act_{m}^{'}(\epsilon^{m})\odot[\frac{\partial L}{\partial \epsilon^{m+1}} ·(\omega^{m+1})^{T}]
$$
当$m=D$ 时
$$
\frac{\partial L}{\partial \epsilon^{D}_{i}}=\frac{\partial L}{\partial y^{D}_{i}}\frac{\partial y^{D}_{i}}{\partial \epsilon^{D}_{i}}=Loss^{'}(y^{D}_{i})· Act^{’}_{D}(\epsilon^{D}_{i})
$$
考虑矩阵
$$
\frac{\partial L}{\partial \epsilon^{D}_{i}}=Loss^{'}(y^{D})\odot Act^{’}_{D}(\epsilon^{D})
$$
考虑$b^{m}$
$$
db^{m} = \frac{\partial L}{\partial \epsilon^{m}}
$$
如果综合，即$b^{m}$为标量
$$
db^{m} = \sum_{i=1}^{size(m)}\frac{\partial L}{\partial \epsilon_{i}^{m}}
$$

### 正向传播

$$
y_{0} = X
$$


$$
y^{m} = Act_{m}(\epsilon^{m})= Act_{m}(y^{m-1}·\omega^{m}+b^{m})
$$
记录所有 $\epsilon^{m}$ 和 $y^{m}$



### 反向传播

step1：计算
$$
\frac{\partial L}{\partial \epsilon^{D}_{i}}=Loss^{'}(y^{D})\odot Act^{’}_{D}(\epsilon^{D})
$$
step2：自最深层向前遍历依次计算 `for m from D to 1`
$$
d\omega^{m}=\frac{\partial L}{\partial w^{m}}=\frac{\partial L}{\partial \epsilon^{m}}\frac{\partial \epsilon^{m}}{\partial w^{m}}=(y^{m-1})^{T}\frac{\partial L}{\partial \epsilon^{m}}
$$

$$
db^{m} = \frac{\partial L}{\partial \epsilon^{m}} 或 db^{m} = \sum_{i=1}^{size(m)}\frac{\partial L}{\partial \epsilon_{i}^{m}}
$$

$$
\omega^{m} = \omega^{m} - \alpha d\omega^{m} //不考虑L1、L2回归
$$

$$
b^{m} = b^{m} - \alpha db^{m}
$$

$$
\frac{\partial L}{\partial \epsilon^{m-1}}=Act_{m}^{'}(\epsilon^{m-1})\odot[\frac{\partial L}{\partial \epsilon^{m}} ·(\omega^{m})^{T}]
$$

