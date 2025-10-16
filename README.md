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

🪜 Các bước chi tiết của Beam Search
------------------------------------

### Bước 1 – Khởi tạo

Anchor tốt nhất ban đầu: A* = null

Tập rule ban đầu: A0 = ∅

### Bước 2 – Sinh các rule ứng viên

Từ mỗi rule hiện tại, thêm 1 điều kiện (predicate) để tạo rule mới.

Ví dụ:
Nếu At−1 = {PetalLength < 2.0}, thì có thể sinh ra các rule mới:

```
At = {
    {PetalLength < 2.0, PetalWidth < 0.5},
    {PetalLength < 2.0, SepalLength < 5.0},
    …
}
```

### Bước 3 – Chọn B rule tốt nhất (KL-LUCB)

Ở bước này, thuật toán đánh giá độ chính xác (precision) của từng rule bằng cách lấy mẫu nhiễu z ∼ D(z∣A).

**Công thức:**

> prec(A) = E<sub>z ~ D(z|A)</sub>[<b>1</b><sub>f(x)=f(z)</sub>]

**Tức là:**

$\text{prec}(A) = \dfrac{\text{số mẫu } z \text{ có } f(z) = f(x)}{\text{tổng số mẫu } z \text{ được sinh ra}}$


**KL-LUCB** (Kullback–Leibler Lower & Upper Confidence Bounds) giúp chọn B rule có độ chính xác tốt nhất, đảm bảo:

$P(min_{A ∈ A} prec(A) ≥ min_{A' ∈ A*} prec(A') − ε) ≥ 1 − δ$

Nói cách khác, các rule được chọn đều gần với rule tốt nhất, sai lệch tối đa ε với xác suất tin cậy 1 − δ.

#### Ví dụ chi tiết cho bước 3

Giả sử ta đang làm việc với mẫu Iris Setosa:

| Thuộc tính | Giá trị |
|-----------|--------|
| SepalLength | 4.9 |
| SepalWidth | 3.0 |
| PetalLength | 1.4 |
| PetalWidth | 0.2 |

Mô hình dự đoán: f(x) = Setosa

Sinh 3 rule ứng viên:

| Rule ID | Rule | Mẫu nhiễu z (PetalLength, PetalWidth) | f(z) | Giống f(x)? |
|---------|------|---------------------------------------|------|-------------|
| A₁ | PetalLength < 2.0 | (1.6, 0.3), (1.8, 0.4), (3.0, 1.0) | Setosa, Setosa, Versicolor | ✅✅❌ |
| A₂ | PetalWidth < 0.5 | (0.3, 0.2), (0.4, 0.3), (1.0, 0.6) | Setosa, Setosa, Versicolor | ✅✅❌ |
| A₃ | SepalLength < 5.0 | (4.7, 3.0), (4.9, 2.8), (5.5, 3.1) | Setosa, Setosa, Versicolor | ✅✅❌ |

Công thức tính precision:

$\text{prec}(A) = \dfrac{\text{số mẫu } z \text{ có } f(z) = f(x)}{\text{tổng số mẫu } z \text{ được sinh ra}}$

Kết quả:

| Rule | Precision | Coverage (ước lượng) |
|------|-----------|---------------------|
| A₁ | 2/3 = 0.67 | 0.60 |
| A₂ | 2/3 = 0.67 | 0.55 |
| A₃ | 2/3 = 0.67 | 0.40 |

Chưa rule nào đạt τ = 0.95.
KL-LUCB chọn A₁, A₂ để mở rộng.

### Cách tính Coverage chi tiết

Công thức chính thức:

$cov(A) = P_{z ~ D} [A(z) = 1]$

Tức là tỷ lệ mẫu trong toàn bộ phân phối D (hoặc trong tập dữ liệu) thỏa điều kiện của rule.

Trong thực tế, ta ước lượng bằng tần suất:

$\text{cov}(A) = \dfrac{\text{số mẫu thỏa rule } A}{\text{tổng số mẫu trong } D}$


**Ví dụ:**

Giả sử tập dữ liệu có 100 mẫu, trong đó:

- 60 mẫu có PetalLength < 2.0 → cov(A₁) = 0.6
- 55 mẫu có PetalWidth < 0.5 → cov(A₂) = 0.55
- 40 mẫu có SepalLength < 5.0 → cov(A₃) = 0.4

Đây là cách ước lượng coverage thực tế trong ví dụ ở trên.

### Bước 4 – Cập nhật anchor tốt nhất

**Nếu rule A có**:

$prec(A) > \tau \ \text{và} \ cov(A) > cov(A^*)$

**thì cập nhật**: A* = A

#### Ví dụ:

Kết hợp 2 rule đầu:

$A₄ = {PetalLength < 2.0, PetalWidth < 0.5}$

Giả sử sinh 5 mẫu nhiễu thỏa A₄:

| z | PetalLength | PetalWidth | f(z) |
|---|-------------|-----------|------|
| z₁ | 1.5 | 0.4 | Setosa |
| z₂ | 1.8 | 0.3 | Setosa |
| z₃ | 1.2 | 0.4 | Setosa |
| z₄ | 1.7 | 0.4 | Setosa |
| z₅ | 1.9 | 0.4 | Versicolor |

Tính toán:

$prec(A₄) = 4/5 = 0.8, cov(A₄) = 0.52$

Thêm điều kiện mới:

$A₅ = {PetalLength < 2.0, PetalWidth < 0.5, SepalLength < 5.5}$

Giả sử 5 mẫu nhiễu đều có f(z) = Setosa:

$prec(A₅) = 5/5 = 1.0, cov(A₅) = 50/100 = 0.50$

(ước lượng: có 50 mẫu trong 100 mẫu dữ liệu thỏa điều kiện này)

→ Rule này đạt prec(A₅) ≥ τ = 0.95, cập nhật: A* = A₅

✅ Kết quả cuối cùng
--------------------

$A* = {PetalLength < 2.0, PetalWidth < 0.5, SepalLength < 5.5}$

Precision = 1.0, Coverage = 0.50

→ Giải thích của mô hình:

"Nếu PetalLength < 2.0, PetalWidth < 0.5, và SepalLength < 5.5, thì mẫu gần như chắc chắn là Setosa."
