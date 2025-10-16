# anchors-linear-regression

# 🧩 Giải thích cơ chế của Anchors (High-Precision Model-Agnostic Explanations)

## 1️⃣ Mục tiêu của Anchors  

Giả sử ta có một mô hình black box $\mathbf{f}: X \to Y$ và một mẫu đầu vào $x \in X$.  
Mục tiêu của **local model-agnostic interpretability** (giải thích cục bộ, độc lập mô hình) là giúp người dùng **hiểu tại sao mô hình lại dự đoán f(x)** cho mẫu cụ thể này.  

**Ý tưởng**: mô hình có thể **quá phức tạp để giải thích toàn cục**, nhưng nếu ta **“phóng to” (zoom in)** vào một dự đoán riêng lẻ, thì có thể tìm được **một số điều kiện (rules)** đơn giản mà khi thỏa mãn, mô hình sẽ gần như luôn dự đoán cùng một kết quả.

---

## 2️⃣ Biểu diễn có thể hiểu được (Interpretable Representation)

Trong các bài toán khác nhau, **biểu diễn có thể hiểu được** sẽ khác nhau, nhưng điểm chung là: **con người có thể đọc và hiểu ý nghĩa của từng đặc trưng (feature)**.

| Loại dữ liệu | Biểu diễn mô hình (machine representation) | Biểu diễn có thể hiểu được (interpretable representation) |
|--------------|---------------------------------------------|-------------------------------------------------------------|
| Văn bản | vector TF-IDF, embeddings | các từ (tokens) xuất hiện trong câu |
| Ảnh | ma trận pixel, đặc trưng CNN | các vùng (superpixels) trong ảnh |
| Dữ liệu bảng (tabular) | vector số thực | các cột có ý nghĩa như “tuổi”, “thu nhập”, “giới tính”, “petal width”... |

**Ví dụ:**  
Trong tập dữ liệu **Iris**, một mẫu có thể là `[5.1, 3.5, 1.4, 0.2]`, nhưng khi hiển thị ở dạng có thể hiểu được, ta gọi nó là:  
> Sepal length = 5.1, Sepal width = 3.5, Petal length = 1.4, Petal width = 0.2  

→ Đây chính là dạng biểu diễn con người hiểu được.

---

## 3️⃣ Định nghĩa Rule (A) và Anchor  

Một **rule** (quy tắc) là một điều kiện đơn giản mô tả đặc trưng nào đó của dữ liệu, ví dụ:  
- “petal length < 2.0”  
- “từ ‘not’ xuất hiện trong câu”  

Tập hợp nhiều rule được gọi là **A**.  
Nói cách khác, $A$ là **một nhóm các điều kiện** (predicates).  
Khi ta viết $A(x) = 1$, điều này có nghĩa là **tất cả các điều kiện trong tập A đều đúng với mẫu x**.  

> 🟢 **Ví dụ:**  
> Nếu $A = \{\text{petal length < 2.0}, \text{petal width < 0.5}\}$ 
> và mẫu x có petal length = 1.4, petal width = 0.2  
> ⇒ $A(x) = 1$ vì cả hai điều kiện đều đúng.

---

Công thức chính thức định nghĩa một **Anchor** như sau:

> 𝔼<sub>D(z|A)</sub> [<b>1</b><sub>f(x)=f(z)</sub>] ≥ τ, A(x)=1

**Giải thích:**
- $D(z|A)$: là **phân phối của các mẫu z** giống x ở chỗ đều thỏa các điều kiện trong A.
- $\mathbf{1}_{f(x)=f(z)}$: là hàm kiểm tra — bằng 1 nếu mô hình cho cùng dự đoán ở x và z, ngược lại bằng 0.
- Lấy **trung bình (kỳ vọng 𝔼)** của giá trị đó chính là tỉ lệ phần trăm các mẫu z có dự đoán giống x.
- $\tau$: là ngưỡng (ví dụ 0.95).

👉 **Hiểu nôm na:** Nếu một tập điều kiện A (anchor) đúng với mẫu x, thì trong **hầu hết các trường hợp tương tự**, mô hình vẫn dự đoán cùng kết quả.

---

### 🌸 Ví dụ với bài toán phân loại hoa Iris
Giả sử mô hình dự đoán cho mẫu x = [5.1, 3.5, 1.4, 0.2] là **“Setosa”**.  
Nếu ta chọn:
$A = \{\text{petal length < 2.0}, \text{petal width < 0.5}\}$
**thì tất cả các hoa thỏa điều kiện A (tức D(z|A))** cũng thường được mô hình dự đoán là “Setosa”.

Nếu 97% các mẫu thỏa A được mô hình dự đoán cùng nhãn:
**𝔼<sub>D(z|A)</sub>[<b>1</b><sub>f(x)=f(z)</sub>] = 0.97 ≥ τ = 0.95**
→ A là **anchor** cho dự đoán này.

---

## 4️⃣ Tính toán hiệu quả các Anchors  

Công thức (2) định nghĩa độ chính xác của một anchor:
$\text{prec}(A) = \mathbb{E}_{D(z|A)} [\mathbf{1}_{f(x)=f(z)}]$

Tức là **tỉ lệ các mẫu trong không gian điều kiện A mà mô hình vẫn cho cùng dự đoán với mẫu x**.

---

### ⚙️ Vấn đề trong thực tế  

Trong thực tế, ta **không thể tính toán kỳ vọng này chính xác**, vì:
- Phân phối $D(z|A)$ có **vô hạn khả năng mẫu z**.  
- Ta không biết phân phối thực của dữ liệu (đặc biệt trong mô hình black box).  

> 🔹 **Ví dụ nhỏ:**  
> Với rule A = {petal length < 2.0}, ta có thể sinh ra vô số mẫu z khác nhau (các giá trị sepal length, sepal width biến thiên liên tục).  
> → Không thể duyệt hết để tính chính xác tỉ lệ mẫu có cùng dự đoán.

---

### 🔢 Định nghĩa xác suất  

Do đó, ta chuyển sang định nghĩa **xác suất**:

$P(\text{prec}(A) \ge \tau) \ge 1 - \delta$

**Giải thích:**  
Ta không yêu cầu tính chính xác tuyệt đối nữa, mà chỉ cần **với xác suất ít nhất 1 − δ**, rule A có độ chính xác ≥ τ.  

Ví dụ:  
- τ = 0.95 (muốn A đúng 95% mẫu)  
- δ = 0.05 (chấp nhận sai lệch 5%)  
→ Tức là ta **tin 95% rằng anchor A giữ vững dự đoán của mô hình**.

---

## 5️⃣ Tối ưu hóa việc chọn Anchor  

Khi nhiều tập rule **A** đều đạt yêu cầu chính xác (tức prec(A) ≥ τ), ta **không chỉ chọn 1 rule đơn lẻ**, mà tìm **những rule có độ bao phủ (coverage) lớn nhất** — tức là **đúng cho nhiều mẫu nhất trong không gian đầu vào**.

Công thức tối ưu hóa:

$$\max_{\text{A s.t.} \quad P(\text{prec}(A) \geq \tau) \geq 1 - \delta} \text{cov}(A)$$

**Ta muốn **tìm một tập rule A** sao cho:**  
> - A phải đủ mạnh để giữ ổn định dự đoán (độ chính xác ≥ τ với xác suất cao).  
> - Trong số các rule đạt điều kiện trên, **chọn rule có phạm vi áp dụng lớn nhất** (tức đúng cho nhiều mẫu nhất có thể).  

Nói cách khác:  
> “Tìm ra bộ điều kiện vừa đủ mạnh để mô hình gần như chắc chắn không đổi kết quả, vừa đủ tổng quát để không chỉ áp dụng cho duy nhất một mẫu.”

---

## 6️⃣ Beam Search cho việc xây dựng Anchors và ví dụ

⚙️ Beam Search trong Anchors
======================

🎯 Mục tiêu
-----------

Tìm anchor (tập các điều kiện đủ mạnh để giải thích dự đoán của mô hình) sao cho:

 > Precision(A)≥τ

và

 > Coverage(A) là lớn nhất có thể.

🧠 Lý do cần Beam Search
------------------------

**Greedy Search** (cũng là 1 cách để tìm anchors) chỉ giữ 1 rule duy nhất tại mỗi bước → dễ mắc kẹt nếu chọn sai ban đầu, và chỉ trả về rule ngắn nhất thỏa điều kiện chính xác mà không quan tâm coverage.

**Beam Search** cải tiến bằng cách:

- Giữ lại B rule tốt nhất (beam width) ở mỗi vòng.
- Duy trì song song nhiều hướng mở rộng.
- Ưu tiên rule có coverage cao nhất trong số những rule thỏa precision.

🔢 Thuật toán tổng quát
-----------------------

Algorithm 2 – Outline of the Beam Search

```
function BeamSearch(f, x, D, τ)
    hyperparameters B, ε, δ
    A* ← null 
    A0 ← ∅  
    loop
        At ← GenerateCands(At−1, cov(A*))  
        At ← B-BestCand(At, D, B, δ, ε)  

        if At = ∅ then break loop

        for all A ∈ At s.t. preclb(A, δ) > τ do  
            if cov(A) > cov(A*) then            
               A* ← A                         

    return A*                                   
```

# 🪜 Các bước chi tiết của Beam Search

## Bước 1 – Khởi tạo

**Anchor tốt nhất:** $A^* = \text{null}$

**Tập rule ban đầu:** $A_0 = \emptyset$

## Bước 2 – Sinh các rule ứng viên

Từ mỗi rule hiện tại, thêm 1 điều kiện (predicate) để tạo rule mới.

**Ví dụ:**

Nếu $A_{t-1} = \{\text{PetalLength} < 2.0\}$

thì có thể sinh các rule mới:

$$A_t = \{\{\text{PetalLength} < 2.0, \text{PetalWidth} < 0.5\}, \{\text{PetalLength} < 2.0, \text{SepalLength} < 5.0\}, \ldots\}$$

## Bước 3 – Chọn B rule tốt nhất (KL-LUCB)

Ở bước này, thuật toán đánh giá độ chính xác (precision) của từng rule bằng cách lấy mẫu nhiễu $z \sim D(z|A)$.

**Công thức tính precision:**

$$\text{prec}(A) = E_{z \sim D(z|A)} [1_{f(x) = f(z)}]$$

Tức là:

$$\text{prec}(A) = \frac{\text{số mẫu } z \text{ mà } f(z) = f(x)}{\text{tổng số mẫu } z \text{ sinh ra}}$$

**KL-LUCB** (Kullback–Leibler Lower & Upper Confidence Bounds) giúp chọn B rule có độ chính xác tốt nhất, đảm bảo:

$$P\left(\min_{A \in A} \text{prec}(A) \geq \min_{A' \in A^*} \text{prec}(A') - \varepsilon\right) \geq 1 - \delta$$

Nói cách khác, các rule được chọn đều gần với rule tốt nhất, sai lệch tối đa $\varepsilon$ với xác suất tin cậy $1 - \delta$.

**Ví dụ chi tiết cho bước 3:**

Giả sử có 3 rule:

| Rule | Precision | Lower bound $\text{prec}_\text{lb}$ | Giữ lại? |
|------|-----------|-----------------------------------|----------|
| A₁   | 0.67      | 0.60                              | ✅       |
| A₂   | 0.67      | 0.59                              | ✅       |
| A₃   | 0.67      | 0.50                              | ❌       |

KL-LUCB giữ lại 2 rule tốt nhất (B=2) có confidence cao nhất — tức là A₁ và A₂.

## Bước 4 – Cập nhật anchor tốt nhất

Nếu rule $A$ có:

$$\text{prec}_\text{lb}(A) > \tau \text{ và } \text{cov}(A) > \text{cov}(A^*)$$

thì cập nhật $A^* = A$.

**Công thức tính Coverage:**

$$\text{cov}(A) = P(z \sim D)[A(z) = 1]$$

hay gần đúng bằng:

$$\text{cov}(A) = \frac{\text{số mẫu thỏa } A}{\text{tổng số mẫu trong } D}$$

## Bước 5 – Dừng khi không còn ứng viên

Khi $A_t = \emptyset$, thuật toán dừng và trả về $A^*$ — anchor cuối cùng.

---

# 🌼 Ví dụ chi tiết: Phân loại hoa Iris

## Dữ liệu

**Mẫu $x$:**

| Thuộc tính   | Giá trị |
|--------------|--------|
| SepalLength  | 4.9    |
| SepalWidth   | 3.0    |
| PetalLength  | 1.4    |
| PetalWidth   | 0.2    |

**Mô hình dự đoán:** $f(x) = \text{Setosa}$

Ta muốn tìm anchor $A$ sao cho:

$$\text{prec}(A) \geq \tau = 0.95$$

và coverage cao nhất.

**Giả sử:** $B = 2, \varepsilon = 0.05, \delta = 0.1$

### 🔹 Bước 1 – Khởi tạo

$$A_0 = \emptyset, A^* = \text{null}$$

### 🔹 Bước 2 – Sinh rule ứng viên ban đầu

Giả sử ta sinh 3 rule cơ bản:

| Rule ID | Rule                | Mẫu nhiễu z (minh họa)            | f(z)                                     | Giống f(x)? |
|---------|---------------------|-----------------------------------|------------------------------------------|-------------|
| A₁      | PetalLength < 2.0   | (1.6, 0.3), (1.8, 0.4), (3.0, 1.0) | Setosa, Setosa, Versicolor              | ✅✅❌      |
| A₂      | PetalWidth < 0.5    | (0.3, 0.2), (0.4, 0.3), (1.0, 0.6) | Setosa, Setosa, Versicolor              | ✅✅❌      |
| A₃      | SepalLength < 5.0   | (4.7, 3.0), (4.9, 2.8), (5.5, 3.1) | Setosa, Setosa, Versicolor              | ✅✅❌      |

⚠️ **Ghi chú:**

- Mẫu nhiễu $z$ chỉ chứa 2 đặc trưng liên quan đến rule hiện tại (ví dụ chỉ PetalLength và PetalWidth).
- Các đặc trưng khác được cố định từ mẫu gốc $x$, để mô hình vẫn có thể dự đoán hợp lệ.

**Công thức precision:**

$$\text{prec}(A) = \frac{\text{số mẫu có } f(z) = f(x)}{\text{tổng số mẫu } z}$$

| Rule | Precision | Coverage (ước lượng) |
|------|-----------|---------------------|
| A₁   | 2/3 = 0.67 | 0.60                |
| A₂   | 2/3 = 0.67 | 0.55                |
| A₃   | 2/3 = 0.67 | 0.40                |

Chưa rule nào đạt $\tau = 0.95$. KL-LUCB chọn A₁, A₂ để mở rộng.

### 🔹 Bước 3 – Sinh tổ hợp mới

Kết hợp 2 rule đầu:

$$A_4 = \{\text{PetalLength} < 2.0, \text{PetalWidth} < 0.5\}$$

Giả sử sinh 5 mẫu nhiễu thỏa $A_4$:

| z  | PetalLength | PetalWidth | f(z)       |
|----|-------------|------------|------------|
| z₁ | 1.5         | 0.4        | Setosa     |
| z₂ | 1.8         | 0.3        | Setosa     |
| z₃ | 1.2         | 0.4        | Setosa     |
| z₄ | 1.7         | 0.4        | Setosa     |
| z₅ | 1.9         | 0.4        | Versicolor |

$$\text{prec}(A_4) = \frac{4}{5} = 0.8, \quad \text{cov}(A_4) = 0.52$$

### 🔹 Bước 4 – Sinh rule mạnh hơn

Thêm điều kiện mới:

$$A_5 = \{\text{PetalLength} < 2.0, \text{PetalWidth} < 0.5, \text{SepalLength} < 5.5\}$$

Giả sử 5 mẫu nhiễu đều có f(z)=Setosa:

$$\text{prec}(A_5) = \frac{5}{5} = 1.0$$

$$\text{cov}(A_5) = \frac{50}{100} = 0.50$$

(ước lượng: 50 trong 100 mẫu trong D thỏa rule này)

→ Rule này đạt $\text{prec} \geq \tau$, cập nhật:

$$A^* = A_5$$

---

## ✅ Kết quả cuối cùng

$$A^* = \{\text{PetalLength} < 2.0, \text{PetalWidth} < 0.5, \text{SepalLength} < 5.5\}$$

**Precision = 1.0, Coverage = 0.50**

→ **Giải thích của mô hình:**

"Nếu PetalLength < 2.0, PetalWidth < 0.5, và SepalLength < 5.5, thì mẫu gần như chắc chắn là Setosa."

## 7️⃣ Anchors cho bài toán Linear Regression

Mặc dù **Anchors** ban đầu được thiết kế cho bài toán **phân loại**, ta vẫn có thể áp dụng cho **bài toán hồi quy tuyến tính (Linear Regression)**.

---

## 🎯 Ý tưởng chính

Thay vì giải thích vì sao mô hình dự đoán một **lớp nhãn cụ thể**, ta sẽ:

1. **Chuyển bài toán hồi quy thành phân loại nhị phân** dựa trên ngưỡng (ví dụ: giá trị dự đoán lớn hơn trung vị thì là *High*, ngược lại là *Low*).  
2. **Sử dụng Anchors Tabular Explainer** để tìm ra những điều kiện trên các đặc trưng *(TV, Radio, Newspaper)* khiến mô hình dự đoán “High” với **độ tin cậy cao**.

---

## 💡 Cách làm này giúp ta hiểu

- “**Khi nào mô hình dự đoán doanh số cao?**”  
- “**Những yếu tố nào của chiến dịch quảng cáo đủ mạnh để giữ ổn định dự đoán ‘Sales cao’?**”

---

## ⚙️ Code minh họa với tập dữ liệu Advertising

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from anchor import anchor_tabular
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("advertising.csv")
X = data[["TV", "Radio", "Newspaper"]].values
y = data["Sales"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


feature_names = ["TV", "Radio", "Newspaper"]

explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=["Low", "High"],
    feature_names=feature_names,
    train_data=X_train
)


threshold = np.median(y_train)
def predict_fn(x):
    preds = model.predict(x)
    return (preds > threshold).astype(int)


instance = X_test[0]
print(f"Instance cần xử lý:\n X = [TV = {instance[0]:.2f}, Radio = {instance[1]:.2f}, Newspaper = {instance[2]:.2f}]")
exp = explainer.explain_instance(instance, predict_fn, threshold=0.95)

print("Prediction class:", "High" if predict_fn(instance.reshape(1, -1))[0] else "Low")
print("Anchor explanation:", exp.names())
print("Precision:", exp.precision())
print("Coverage:", exp.coverage())


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = X_test[:, 0]  # TV
y = X_test[:, 1]  # Radio
z = X_test[:, 2]  # Newspaper
preds = model.predict(X_test)

# Vẽ điểm dựa trên phân loại (Low/High)
colors = ['red' if p > threshold else 'blue' for p in preds]

ax.scatter(x, y, z, c=colors, s=50, edgecolors='k')
ax.scatter(instance[0], instance[1], instance[2], c='black', s=200, marker='*', label='Explained instance')

ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Newspaper')
ax.set_title('3D Visualization of Anchors (TV, Radio, Newspaper)')
plt.legend()
plt.show()
```

**Output**
```
Instance cần xử lý: 
 X = [TV = 163.30, Radio = 31.60, Newspaper = 52.90]    
Prediction class: High  
Anchor explanation: ['TV > 150.65', 'Radio > 21.20']    
Precision: 1.0  
Coverage: 0.2696    
```

[chèn ảnh]

## 📘 Diễn giải kết quả mô hình

### Tại sao mô hình dự đoán "Sales cao" cho mẫu này?

Mô hình đã phân tích mẫu dữ liệu sau:
- **Chi phí quảng cáo trên TV**: 163.3
- **Chi phí quảng cáo trên Radio**: 31.6
- **Chi phí quảng cáo trên Newspaper**: 52.9

Và đưa ra kết luận: **Doanh số sẽ cao (High)**

### 🧱 Quy tắc giải thích (Anchor)

**Mô hình dựa vào điều kiện chính sau:**

Nếu **TV > 150.65** VÀ **Radio > 21.20** → Doanh số sẽ cao

Nói cách khác: Khi chi phí quảng cáo TV vượt 150.65 (đơn vị) **và** Radio vượt 21.20 (đơn vị), thì doanh số sẽ tăng cao.

### Mức độ tin cậy

#### ✅ Độ chính xác (Precision): 100%
Quy tắc này rất đáng tin cậy! Tất cả các trường hợp thỏa mãn điều kiện trên đều thực sự cho ra doanh số cao. Không có ngoại lệ hay sai sót.

#### 📊 Phạm vi áp dụng (Coverage): 27%
Quy tắc này chỉ áp dụng cho khoảng **27% các mẫu** trong tập dữ liệu. Có nghĩa là còn 73% các trường hợp doanh số cao khác không tuân theo quy tắc này - chúng có những yếu tố khác quyết định.
