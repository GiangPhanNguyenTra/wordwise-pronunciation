import nltk

# 1. Tải bộ tách từ và câu (Bắt buộc cho g2p_en hoạt động)
nltk.download('punkt')
nltk.download('punkt_tab') # NLTK bản mới tách cái này ra, cần tải thêm để tránh lỗi

# 2. Tải từ điển phát âm CMU (Bắt buộc vì code bạn có dòng: from nltk.corpus import cmudict)
nltk.download('cmudict')

# 3. Tải bộ gán nhãn từ loại (POS Tagger - giúp phân biệt danh từ/động từ để phát âm đúng)
nltk.download('averaged_perceptron_tagger_eng')
# Nên tải thêm bản gốc này vì một số thư viện cũ vẫn gọi tên này
nltk.download('averaged_perceptron_tagger')