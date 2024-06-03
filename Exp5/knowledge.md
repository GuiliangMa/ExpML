# 有关SVM的相关知识备忘录

## SVM基本表示形式

SVM线性平面方程可以描述为：$w^Tx+b=0$，假设间隔为$d$

通过一些列等价变化可以将分类的边界变为：
$$
\left\{
\begin{matrix}
\mathbf{w}^T\mathbf{x}+b \geqslant +1，	y_i=+1\\
\mathbf{w}^T\mathbf{x}+b \leqslant -1，	y_i=-1
\end{matrix}
\right.
$$
通过点到超平面的距离公式可以得知
$$
r = \frac{|\mathbf{w}^T+b|}{||\mathbf{w}||}\\
d = \frac{|1|}{||\mathbf{w}||}+\frac{|-1|}{||\mathbf{w}||} = \frac{2}{||\mathbf{w}||}
$$
SVM的过程即最大化 间隔$d$，即：
$$
\underset{\mathbf{w},b}{max}\quad \frac{2}{||\mathbf{w}||} \\
s.t.\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)-1\geqslant0\quad i=1,2,...,m
$$
该问题等价于：
$$
\underset{\mathbf{w},b}{min}\quad \frac{1}{2}||\mathbf{w}||^2 \\
s.t.\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)-1\geqslant0\quad i=1,2,...,m
$$
该表示即为支持向量机（Support Vector Machine）的基本表示形式，当然此时以及可以用来求解 $\omega,b$ ，比如采用**合页损失函数（hinge loss）**



### 合页损失函数

梯度下降是一种优化算法，用于通过不断调整参数来最小化损失函数。以下是梯度下降过程的详细解释：

#### 1. 初始化参数

首先，算法需要初始化参数（在我们的 SVM 实现中是权重 $\mathbf{w}$ 和偏置 $b$）。这些参数可以随机初始化或设置为零。

#### 2. 计算损失函数

损失函数衡量模型预测与实际结果之间的差异。在支持向量机中，常用的损失函数是合页损失函数（hinge loss），加上一个正则化项来防止过拟合。

合页损失函数的形式如下：
$$
L = C \sum_{i=1}^n \max(0, 1 - y_i (\mathbf{w} \cdot \mathbf{x}_i + b)) + \frac{1}{2} ||\mathbf{w}||^2
$$
其中，$y_i$ 是第 $ i $ 个样本的标签， $ \mathbf{x}_i $ 是第 $ i $ 个样本的特征向量， $ \lambda $ 是正则化参数。

#### 3. 计算梯度

梯度是损失函数相对于参数的导数，表示损失函数在当前参数值下变化的方向和速度。在我们的 SVM 实现中，梯度的计算如下：

- 当样本满足 $ y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 $ 时，损失函数对参数的梯度为：
  $$
  \frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} \\
  \frac{\partial L}{\partial b} = 0
  $$
  
- 当样本不满足 $ y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 $ 时，损失函数对参数的梯度为：
  $$
  \frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} -C y_i \mathbf{x}_i \\
   \frac{\partial L}{\partial b} = -Cy_i
  $$

#### 4. 更新参数

利用梯度更新参数。更新的规则是沿着梯度的反方向调整参数，因为梯度指向的是损失函数增加最快的方向，而我们希望减少损失。

更新公式如下：
$$
\mathbf{w} = \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}\\
b = b - \eta \frac{\partial L}{\partial b}
$$
其中， $ \eta $ 是学习率，决定了每次更新的步长。

#### 5. 重复迭代

重复步骤 2 到 4，直到达到预定的迭代次数或者损失函数的变化趋于平缓，即收敛。

## SVM的对偶形式

基于凸规划的表示形式，可以采用拉格朗日乘子法来对原问题进行求解

原问题：
$$
\underset{\mathbf{w},b}{min}\quad \frac{1}{2}||\mathbf{w}||^2 \\
s.t.\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)-1\geqslant0\quad i=1,2,...,m
$$
引入拉格朗日因子$\lambda$后形式如下：
$$
{L(\mathbf{w},b;\lambda)} = \frac{1}{2}||\mathbf{w}||^2+\sum_{i=1}^{m}\lambda_i(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))\\
\nabla_{\mathbf{w},b}L(\mathbf{w},b;\lambda) = \mathbf{0}\\
\nabla_\mathbf{w} L(\mathbf{w},b;\lambda) = \mathbf{w}-\sum_{i=1}^m\lambda_iy_i\mathbf{x}_i=\mathbf{0}\quad \Rightarrow \quad \mathbf{w}^*= \sum_{i=1}^m\lambda_iy_i\mathbf{x}_i\\
\nabla_b L(\mathbf{w},b;\lambda) = \sum_{i=1}^m\lambda_iy_i = 0
$$
带入即可得到其拉格朗日对偶问题：
$$
\underset{\Lambda}{max} \quad L(\mathbf{w}*,b*;\mathbf{\Lambda})\\
s.t. 
\begin{matrix}
\quad \nabla_{\mathbf{w},b} L(\mathbf{w},b;\mathbf{\Lambda}) = \mathbf{0} \\
\quad \quad \lambda_i \geq 0,\quad i=1,...,m
\end{matrix}
$$
带入展开后可得：
$$
\underset{\mathbf{\Lambda}}{max} \quad \sum_{i=1}^m \lambda_i-\frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m\lambda_i\lambda_jy_iy_j\mathbf{x}_i^T\mathbf{x}_j \\
s.t. 
\begin{matrix}
\quad \sum_{i=1}^m \lambda_iy_i = 0 \\
\quad \quad \lambda_i \geq 0,\quad i=1,...,m
\end{matrix}
$$
引入矩阵形式则为：
$$
\underset{\mathbf{\Lambda}}{max} \quad \mathbf{1}^T\mathbf{\Lambda}-\frac{1}{2}\mathbf{\Lambda}^TD\mathbf{\Lambda} \\
s.t. 
\begin{matrix}
\quad y^T\mathbf{\Lambda} = \mathbf{0} \\
\quad \quad \mathbf{\Lambda} \geq 0
\end{matrix}
$$
其中 $D_{ij} = y_iy_j\mathbf{x_i^T}\mathbf{x_j} = (\mathbf{y} \mathbf{y^T})\odot(XX^T)$