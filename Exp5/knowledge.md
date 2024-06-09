# 有关SVM的相关知识备忘录

## SVM基本表示形式

SVM线性平面方程可以描述为：$\mathbf{w}^T\mathbf{x}+b=0$，假设间隔为$d$

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

### 理论

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
从数学上来看，$\lambda_i,\lambda_j$ 的取值 要落在 0-C取值的正方形中，且要落在对应的一条线段上。

若放缩$\lambda_i$ 的取值边界，则有 $L\leq \lambda_i \leq H$

由此可以求出 $\lambda_i$ 的上下界：

- $y_i \neq y_j$ 时：
  $$
  L = max(0,\lambda_i-\lambda_j),\quad H=min(C,C+\lambda_j-\lambda_i)
  $$
  
- $y_i = y_j$ 时：
  $$
  L = max(0,C+\lambda_j-\lambda_i),\quad H=min(C,\lambda_j+\lambda_i)
  $$
  定义优化函数为：
  $$
  \Psi = (\lambda_i+\lambda_j)-[\frac{1}{2}K_{ii}\lambda_i^2+\frac{1}{2}K_{jj}\lambda_j^2+y_iy_jK_{ij}\lambda_i\lambda_j]
  $$
  

​	由于$\lambda_i$ 和 $\lambda_j$ 有线性关系，因此可以消去$\lambda_i $，进而令 $\Psi$ 对 $\lambda_j$ 求二阶导，并令二阶导为0可以得到：
$$
\lambda_{jnew}(2K_{ij}-K_{ii}-K_{jj}) = \lambda_{jold}(2K_{ij}-K_{ii}-K_{jj})-y_j(E_i-E_j) \\
E_i = f(\mathbf{x}_i) -y_i
$$
令：$\eta = 2K_{ij} - K_{ii} - K_{jj}$

因此：
$$
\lambda_{jnew} = \lambda_{jold} - \frac{y_j(E_i-E_j)}{\eta}
$$
但是还需要考虑上下界截断：
$$
\lambda_{jnewclipped} = 
\left\{
\begin{matrix}
H,\quad if \quad \lambda_{jnew} \geq H\\
\lambda_{jnew}, \quad if \quad L<\lambda_{jnew}<H \\
L,\quad if \quad \lambda_{jnew} \leq L
\end{matrix}
\right.
$$
从而得到 $\lambda_i$ 的更新：
$$
\lambda_{inew} = \lambda_{iold}+y_iy_j(\lambda_{jold}-\lambda_{jnewclipped})
$$
令：
$$
b_1 = b_{old}-E_i-y_iK_{ii}(\lambda_{inew}-\lambda_{iold})-y_jK_{ij}(\lambda_{jnewclipped}-\lambda_{jold})\\
b_2 = b_{old}-E_j-y_iK_{ij}(\lambda_{inew}-\lambda_{old})-y_jK_{jj}(\lambda_{jnewclipped}-\lambda_{jold})
$$
则 $b$ 的更新为：
$$
b_{new} = \left\{
\begin{matrix}
b_1,\quad if\quad 0<\lambda_{inew}<C\\
b_2,\quad if \quad 0<\lambda_{jnewclipped}<C\\
\frac{b_1+b_2}{2},otherwise
\end{matrix}
\right.
$$

### 启发式选择

如果两个拉格朗日乘子其中之一违背了 KKT 条件，此时，每一次乘子对的选择，都能使得优化目标函数减小。

若 $\lambda_i =0 $，可知样本 $\mathbf{x}_i$ 不会对模型产生影响

若 $\lambda_i = C$，样本 $\mathbf{x}_i$ 不会是支持向量

若 $0<\lambda_i<C$，则$\lambda_i$ 没有落在边界上，当下式满足时，$\lambda_i$ 会违反KKT条件：
$$
\lambda_i<C \quad and \quad y_if(\mathbf{x}_i) -1 <0\\
\lambda_i>0 \quad and \quad y_if(\mathbf{x}_i) -1 >0\\
$$
由于上式过于严苛，可以考虑设置一个容忍区间 $[-\tau,\tau]$，并考虑令：
$$
R_i = y_iE_i= y_i(f(\mathbf{x}_i)-y_i) = y_if(\mathbf{x}_i)-1
$$
可以将违反KKT条件表达式写为：
$$
\lambda_i<C \quad and \quad R_i<-\tau\\
\lambda_i>0 \quad and \quad R_i >\tau
$$
则启发式选择$\lambda_i,\lambda_j$ 可以看作两层循环：

外层循环中，如果当前没有 $\lambda$ 对的变化，意味着所有的 $\lambda_i$ 都遵循了KKT条件，需要在整个样本集上进行迭代。否则，只需要选择在处在边界内 $0<\lambda_i<C$、并且违反了KKT条件的$\lambda_i$。

内层循环中，选出使得 $E_i-E_j$ 达到最大的 $\lambda_j$





## 特殊处理：

这两段代码段实现了SMO算法的两个不同版本，时间差异巨大的原因在于它们采用了不同的启发式方法来选择需要优化的拉格朗日乘子对，并且处理方式有所不同。

### 第一段循环：

```python
while num_changed > 0 or examine_all:
    num_changed = 0
    if examine_all:
        for i in range(n_samples):
            _, j, E_i, E_j = choose_alpha_pair(i)
            if j is not None:
                if update_alpha_pair(i, j, E_i, E_j):
                    num_changed += 1
    else:
        for i in range(n_samples):
            if 0 < self.Lambda[i] < C:
                _, j, E_i, E_j = choose_alpha_pair(i)
                if j is not None:
                    if update_alpha_pair(i, j, E_i, E_j):
                        num_changed += 1
    examine_all = (examine_all == False)
```

#### 特点：

1. **检查所有样本（examine_all = True）**：在这种情况下，所有样本都会被检查，时间复杂度较高，因为每个样本都需要进行一次拉格朗日乘子的选择和更新。
2. **只检查非边界样本（examine_all = False）**：在这种情况下，只检查那些拉格朗日乘子在（0, C）范围内的样本，减少了需要检查的样本数。

#### 时间差异原因：

- 当`examine_all`为True时，每次迭代都会遍历所有样本，消耗大量时间。
- 当`examine_all`为False时，只遍历非边界样本，减少了遍历次数，但如果条件不满足，需要反复切换`examine_all`状态，导致额外的迭代和检查。

### 第二段循环：

```python
iteration = 0
while iteration < max_iter:
    num_changed = 0
    for i in range(n_samples):
        E_i = compute_error(i)
        if (y[i] * E_i < -tolerance and self.Lambda[i] < C) or (y[i] * E_i > tolerance and self.Lambda[i] > 0):
            j = choose_alpha_pair(i, E_i)
            if update_alpha_pair(i, j):
                num_changed += 1
    if num_changed == 0:
        break
    iteration += 1
```

#### 特点：

1. **固定迭代次数（max_iter）**：每次迭代遍历所有样本，但通过`max_iter`限制最大迭代次数。
2. **直接选择对违反KKT条件的样本对进行优化**：减少不必要的迭代次数。

#### 时间差异原因：

- 每次迭代都遍历所有样本，但直接对违反KKT条件的样本对进行优化，这样在每次迭代中更高效。
- 在更新拉格朗日乘子对时，如果没有发生变化（num_changed == 0），则直接退出循环，减少了不必要的迭代次数。

### 总结：

- **第一段循环**：采用检查所有样本和非边界样本交替进行的方法，导致在每次迭代中都可能进行大量无效的检查，从而影响性能。
- **第二段循环**：每次迭代都遍历所有样本，但只对违反KKT条件的样本对进行优化，同时通过设置最大迭代次数限制总的迭代次数，优化了算法的效率。

因此，第二段代码在时间复杂度和效率上更优，因为它减少了不必要的检查和迭代次数，更快地收敛到解决方案。
