# BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG MODULE CHẤM ĐIỂM PHÁT ÂM (AI PRONUNCIATION SCORING)

**Dự án:** Hệ thống hỗ trợ học từ vựng tiếng Anh thông minh (Word Wise)  
**Module:** Luyện phát âm từ & câu (Pronunciation Trainer)  
**Phương pháp:** Unsupervised Learning (Whisper ASR + Phonetic Matching + DTW)  
**Dataset kiểm thử:** Speechocean762 (Test Set - Cleaned N=2000)  
**Ngày báo cáo:** 11/01/2026

---

## 1. Tổng quan phương pháp

Module chấm điểm phát âm được tôi xây dựng dựa trên kiến trúc **Unsupervised (Không giám sát)**, giúp hệ thống hoạt động độc lập mà không yêu cầu huấn luyện lại mô hình Acoustic trên dữ liệu gán nhãn tốn kém. Quy trình xử lý bao gồm các bước chính:

1.  **ASR Engine:** Sử dụng mô hình **OpenAI Whisper Base** để nhận dạng văn bản từ giọng nói người dùng và trích xuất dấu thời gian (timestamp).
2.  **Phonetic Conversion:** Chuyển đổi văn bản gốc (Reference) và văn bản nhận dạng (Hypothesis) sang chuẩn IPA (International Phonetic Alphabet) để so sánh ở cấp độ âm vị học.
3.  **Alignment:** Ứng dụng thuật toán **DTW (Dynamic Time Warping)** để so khớp tối ưu chuỗi âm vị của người dùng với chuỗi chuẩn.
4.  **Scoring:** Tính toán điểm số tổng hợp dựa trên trọng số (Weighted Scoring) của 4 tiêu chí: Độ chính xác (Accuracy), Độ trôi chảy (Fluency), Độ đầy đủ (Completeness) và Ngữ điệu (Prosody).

---

## 2. Kết quả Benchmark định lượng

Kết quả đánh giá được thực hiện trên 2000 mẫu dữ liệu hợp lệ từ tập Test của bộ dữ liệu chuẩn Speechocean762.

| Metric (Tiêu chí) | MAE (Sai số tuyệt đối trung bình) | PCC (Hệ số tương quan Pearson) |
| :---------------- | :-------------------------------- | :----------------------------- |
| **Accuracy**      | **12.41**                         | **0.501**                      |
| **Fluency**       | 10.81                             | 0.469                          |
| **Completeness**  | 4.72                              | -                              |
| **Total Score**   | **5.59**                          | **0.594**                      |

### Biểu đồ phân tích:

- **Total Score Correlation (PCC ~0.60):** Các điểm dữ liệu phân bố tập trung quanh đường chéo lý tưởng (Ideal Line). Điều này cho thấy mô hình của tôi có khả năng phân loại năng lực phát âm của người học (từ Yếu đến Giỏi) tương đồng với đánh giá của chuyên gia con người.
- **Error Distribution:** Độ lệch trung bình (Mean Diff) là **-1.23**, cho thấy thuật toán chấm điểm rất cân bằng, không bị thiên kiến chấm quá cao hay quá thấp. Phân phối lỗi tập trung chủ yếu trong khoảng tin cậy **±5 điểm**.

---

## 3. Đánh giá chi tiết: Ưu điểm và Nhược điểm

### 3.1. Ưu điểm (Pros)

1.  **Độ chính xác tổng thể cao (Low MAE):**
    - Sai số trung bình chỉ **~5.6 điểm** trên thang 100. Đây là kết quả rất ấn tượng đối với một hệ thống không giám sát. Trong thực tế đánh giá ngôn ngữ, sự chênh lệch giữa hai giám khảo con người cũng thường rơi vào khoảng 3-5 điểm.
2.  **Khả năng xử lý lỗi tốt (Robustness):**
    - Module đã xử lý thành công các trường hợp biên như audio rác, nhiễu nền hoặc im lặng (gán điểm 0 hoặc điểm sàn thấp), tránh được lỗi "Over-estimation" (chấm điểm ảo) thường gặp ở các hệ thống ASR thông thường.
3.  **Phản hồi chi tiết & Trực quan (Granular Feedback):**
    - Hệ thống hỗ trợ bôi màu từng từ (Xanh/Đỏ) và nghe lại từng từ (Word-level timestamp). Đây là tính năng cốt lõi tôi hướng tới để giúp người học cải thiện, quan trọng hơn là một con số điểm tổng quát.
4.  **Tính tổng quát hóa (Generalization):**
    - Nhờ sử dụng Whisper và IPA Converter, module có thể chấm điểm bất kỳ câu văn nào mà không cần train lại (Zero-shot), giúp tôi dễ dàng mở rộng nội dung bài học trong tương lai.

### 3.2. Nhược điểm và Hạn chế (Cons & Limitations)

1.  **Vấn đề "Whisper Hallucination" (Tự sửa lỗi):**
    - Whisper là mô hình mạnh về ngữ nghĩa. Khi người dùng phát âm sai nhẹ (ví dụ: _sheet_ thành _shit_), Whisper có thể tự động sửa thành từ đúng dựa trên ngữ cảnh câu. Điều này khiến điểm Accuracy đôi khi cao hơn thực tế (giới hạn PCC Accuracy ở mức 0.5).
2.  **Khó khăn với từ ngắn/số đếm:**
    - Các từ đơn âm tiết hoặc số đếm (One, Two...) không có ngữ cảnh đi kèm thường bị nhận diện kém chính xác hơn câu dài.
3.  **Hạn chế về Prosody (Ngữ điệu):**
    - Module hiện tại tính ngữ điệu dựa trên độ biến thiên năng lượng (RMS) và tốc độ nói. Cách tiếp cận này đơn giản và hiệu quả nhưng chưa thể bắt được các lỗi tinh tế về cao độ (Pitch/Intonation) như lên giọng cuối câu hỏi.

---

## 4. Tại sao có các hạn chế trên? (Nguyên nhân)

1.  **Bản chất của công nghệ ASR (Automatic Speech Recognition):** Whisper được tối ưu để **hiểu** ý người nói ("What was said"), không phải để **đánh giá** cách họ nói ("How it was said"). Để khắc phục triệt để cần dùng các mô hình GOP (Goodness of Pronunciation) chuyên dụng, nhưng bù lại sẽ phức tạp và tốn tài nguyên hơn nhiều.
2.  **Đặc thù dữ liệu Speechocean762:** Dữ liệu gán nhãn của con người (Ground Truth) là các số nguyên (1-10) và mang tính chủ quan cảm tính. Việc thuật toán tính toán ra số thực (74.5, 81.2...) dẫn đến việc khó khớp tuyệt đối về mặt toán học (PCC bị giới hạn).

---

## 5. Kết luận: Khả năng ứng dụng thực tế

**Câu hỏi:** _Module này có đủ tiêu chuẩn để tích hợp vào Hệ thống hỗ trợ học từ vựng tiếng Anh thông minh của tôi không?_

**Trả lời: CÓ.**

### Lý do:

1.  **Mục đích sử dụng:** Hệ thống của tôi là **Ứng dụng Luyện tập (Learning Tool)**, không phải là **Hệ thống Thi cử (Testing System)**.
    - Đối với nhu cầu luyện tập: Người dùng cần phản hồi nhanh (Real-time feedback), chỉ ra được từ nào đọc sai và một con số điểm mang tính khích lệ/tham khảo. Sai số **5-7%** là hoàn toàn chấp nhận được cho mục tiêu giáo dục này.
2.  **Trải nghiệm người dùng:** Việc tôi tích hợp thành công Word-level Timestamp và bôi màu lỗi (Visual Feedback) mang lại giá trị thực tiễn cao hơn nhiều so với việc chỉ cố gắng tối ưu chỉ số PCC thêm một vài phần trăm.
3.  **Hiệu năng/Chi phí:** Giải pháp hiện tại chạy nhanh, nhẹ, dễ deploy và không tốn chi phí huấn luyện lại (Retraining cost) khi thêm từ vựng mới.

**Định hướng cải thiện:**

- Tôi sẽ hiển thị điểm số dưới dạng thang điểm rộng (ví dụ: sao, hoặc làm tròn chục) để giảm cảm giác sai số cho người dùng.
- Tập trung phát triển tính năng "Click to listen" để người dùng tự nghe lại, so sánh giọng mình với giọng mẫu, bù đắp cho các hạn chế về ngữ điệu của AI.

---

_Báo cáo được tổng hợp dựa trên kết quả benchmark ngày 11/01/2026._
