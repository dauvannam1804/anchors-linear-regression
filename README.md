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
