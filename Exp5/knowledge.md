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

该问题为一个QR问题，可以采用内点法或者SMO进行求解，由于原问题是由拉格朗日乘数变化而来，因此在强对偶的前提下一定满足KKT定理，因此对于那些在 $\mathbf{w} \cdot \mathbf{x}_i + b = \pm  1$ 上的点对应的$\lambda_i$不为0， 其余的$\lambda_i$ 为0，那些在线上的点称之为支持向量，因此对于一个模型我们可以仅保存支持向量即可。

由于后续会存在利用核函数进行升维度的操作，因此$\mathbf{w}$ 的维度并不好确定，但是根据拉普拉斯函数的约束条件我们可以得知：
$$
\quad \mathbf{w}^*= \sum_{i=1}^m\lambda_iy_i\mathbf{x}_i\\
$$
如果我们将支持向量进行存储，那么首先可以根据支持向量之间的关系求解 $b$，设支持向量的集合为 $X_{sv},y_{sv}$，则：
$$
b = avg(y_{sv}-X_{sv}\mathbf{w}) = avg(y_{sv}-X_{sv}\sum_{i=1}^{sv}\lambda_iy_{svi}\mathbf{x_{sv}}_i)
$$
那么对于预测即为，在此处将支持向量集合写作 $X,y$
$$
y_{pred} = \sum_{i=1}^m \lambda_{i}y_{i}\mathbf{x}_{i}^T\mathbf{x}+b
$$
引入核函数则为：
$$
y_{pred} = \sum_{i=1}^m \lambda_{i}y_{i}K(\mathbf{x}_{i},\mathbf{x})+b
$$


## 核函数（Kernel Functions）

在实际应用中，许多数据集不是线性可分的，这意味着没有一个直线或平面能够完全正确地分类数据点。为了解决这个问题，SVM使用了一种叫做核技巧的方法，允许SVM在更高维的空间中有效地进行工作，而无需直接计算高维空间中的点。

核函数可以将输入数据映射到一个高维空间，其中数据点更有可能是线性可分的。核函数的选择取决于数据的特性和问题类型。常见的核函数包括：

- **线性核（Linear Kernel）**：没有进行任何映射，保持数据在原始空间中不变，适用于线性可分的数据集。
- **多项式核（Polynomial Kernel）**：将数据映射到一个多项式特征空间，适用于非线性问题。
- **径向基函数核（RBF，也称为高斯核）**：一种非常流行的核，可以映射到无限维的空间，非常适合处理那些在原始空间中呈复杂分布的数据。
- **Sigmoid核**：类似于神经网络中使用的激活函数。

核函数 $K(\mathbf{x},\mathbf{y})$ 是一个函数核函数 ，对于所有 $\mathbf{x}$ 和 $\mathbf{y}$ 在输入空间中，它返回这两个点在高维特征空间中的内积，即： 
$$
K(\mathbf{x},\mathbf{y}) = <\phi(\mathbf{x}),\phi(\mathbf{y})>
$$


这里的$<,>$ 表示内积，而 $\phi$ 表示从输入空间到某个高维特征空间的映射。

常见的核函数：

线性核：
$$
K(\mathbf{x},\mathbf{y}) = \mathbf{x}^T\mathbf{y}
$$
多项式核：
$$
K(\mathbf{x},\mathbf{y}) = (\mathbf{x}^T\mathbf{y}+c)^d
$$
高斯核（RBF）：
$$
K(\mathbf{x},\mathbf{y}) = exp(-\frac{||\mathbf{x}-\mathbf{y}||^2}{2\sigma^2})
$$
拉普拉斯核：
$$
K(\mathbf{x},\mathbf{y}) = exp(-\frac{||\mathbf{x}-\mathbf{y}||}{\sigma})
$$
Sigmoid核：
$$
K(\mathbf{x},\mathbf{y}) = tanh(\alpha\mathbf{x}^T\mathbf{y}+c)
$$
在实际应用时，只需要将$D_{ij} = y_iy_j\mathbf{x_i^T}\mathbf{x_j} = (\mathbf{y} \mathbf{y^T})\odot(XX^T)$ 修改为 $D_{ij} = y_iy_j\mathbf{x_i^T}\mathbf{x_j} = (\mathbf{y} \mathbf{y^T})\odot K(\mathbf{x}_i,\mathbf{x}_j)$ 即可

## 软间隔

在实际应用中，数据往往不是完全线性可分的，这就需要使用软间隔（soft margin）的概念来处理轻微的数据重叠。

松弛变量（Slack Variables）

对于每一个训练样本，引入一个松弛变量$\xi_i$。这个变量表示第$i$个数据点违反边界的程度。如果，则$\xi_i=0$该点正确分类并且在边界外。如果 ，则该$\xi_i>0$点在边界内或者被错误分类。

那么原问题要优化的函数则变为：
$$
\underset{\mathbf{w},b,\xi_{i}}{min}\quad \frac{1}{2}||w||^2+C\sum_{i=1}^m \xi_i\\
s.t. \quad y_{i}(\mathbf{w}^T\mathbf{x}_i+b)\geq 1 -\xi_{i} \\
\quad \quad \xi_i\geq0\quad i=1,2,...,m
$$
再次引入拉格朗日乘子：
$$
L(\mathbf{w},b,\mathbf{\Xi};\mathbf{\Lambda},\mathbf{\Mu}) = \frac{1}{2}||w||^2+C\sum_{i=1}^m\xi_i+\sum_{i=1}^m\lambda_i(1-\xi_i-y_i(\mathbf{w}^T\mathbf{x}_i+b))-\sum_{i=1}^m\mu_i\xi_i
$$
其中 $\lambda_i\geq0,\mu_i\geq0$ 均为拉格朗日乘子，令$L(\mathbf{w},b,\mathbf{\Xi};\mathbf{\Lambda},\mathbf{\Mu})$，对$\mathbf{w},b,\mathbf{\Xi}$ 求偏导可得：
$$
\mathbf{w} = \sum_{i=1}^m \lambda_iy_i\mathbf{x}_i\\
0 = \sum_{i=1}^m \lambda_iy_i \\
C = \lambda_i+\mu_i
$$


带入后可以得到其对偶问题
$$
\underset{\Lambda}{max}\quad\sum_{i=1}^m\lambda_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\lambda_i\lambda_jy_iy_jK(\mathbf{x}_i,\mathbf{x}_j))\\
s.t.\quad \sum_{i=1}^m\lambda_iy_i = 0\\
\quad \quad 0\leq\lambda_i\leq C,\quad i=1,2,...,m
$$
则应满足KKT 条件即：
$$
\left\{
\begin{matrix}
\lambda_i\geq0,\\
\mu_i\geq0,\\
y_if(\mathbf{x}_i)-1+\xi_i\geq0,\\
\lambda_i(y_if(\mathbf{x}_i)-1+\xi_i)=0,\\
\xi_i\geq0,\\
\mu_i\xi_i = 0.
\end{matrix}
\right.
$$
整体优化表达式与无软间隔的表达式除了在引入的惩罚因子和拉格朗日乘子$\mu_i$ 存在约束外，最本质的在于对于$\lambda_i$ 的约束存在了一个上限$C$

## SMO 解释

正如上述软间隔所描述的约束优化表达式
$$
\underset{\Lambda}{max}\quad\sum_{i=1}^m\lambda_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\lambda_i\lambda_jy_iy_jK(\mathbf{x}_i,\mathbf{x}_j))\\
s.t.\quad \sum_{i=1}^m\lambda_iy_i = 0\\
\quad \quad 0\leq\lambda_i\leq C,\quad i=1,2,...,m
$$
对KKT 条件进行一个简单的转换，则会得到：
$$
\lambda_i=0⇔y_if(\mathbf{x}_i)≥1\\
0<\lambda_i<C⇔y_if(\mathbf{x}_i)=1\\
\lambda_i=C⇔y_if(x_i)≤1
$$
在 SMO（序列最小化）方法出现之前，人们依赖于二次规划求解工具（例如基于内点法的 `cvxopt`）来解决上述的优化问题，训练 SVM。这些工具需要具有强大计算能力的计算机进行支撑，实现也比较复杂。SMO 算法将优化问题分解为容易求解的若干小的优化问题，来训练 SVM。简言之，SMO 仅关注 **$\Lambda $对** 和 **偏置 $b$ 的求解更新，进而求解出权值向量 $\mathbf{w}$的隐式表达，得到决策边界（分割超平面），从而大大减少了运算复杂度。(理论上会减少运算量和复杂度)

SMO会选择一对$\lambda_i$ 和 $\lambda_j$，并固定其他参数（将其他参数认为是常数），则约束条件就会变为：
$$
\lambda_iy_i+\lambda_jy_j = -\sum_{k\neq i,j}\lambda_ky_k\\
0\leq \lambda_i \leq C\\
0\leq \lambda_j \leq C\\
$$
那么原问题可以替换为：
$$
\underset{\lambda_i,\lambda_j}{\mathbf{max}} \quad (\lambda_i+\lambda_j)-[\frac{1}{2}K_{ii}\lambda_i^2+\frac{1}{2}K_{jj}\lambda_j^2+y_iy_jK_{ij}\lambda_i\lambda_j]-[y_i\lambda_i\sum_{k\neq i,j}y_k\lambda_kK_{ki}+y_j\lambda_j\sum_{k\neq i,j}y_k\lambda_kK_{kj}]\\
s.t. \quad \lambda_iy_i+\lambda_jy_j = -\sum_{k\neq i,j}\lambda_ky_k\\
0\leq \lambda_i \leq C\\
0\leq \lambda_j \leq C\\
$$
https://yoyoyohamapi.gitbooks.io/mit-ml/content/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/codes/%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C.html