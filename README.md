# ViRAG - Hệ thống Tóm tắt và Hỏi đáp tiếng Việt

ViRAG là hệ thống RAG (Retrieval-Augmented Generation) cho tài liệu tiếng Việt. Ứng dụng cho phép người dùng tải văn bản hoặc PDF, tóm tắt nội dung, và đặt câu hỏi dựa trên tài liệu đã tải lên.

Hệ thống được tách thành 3 phần chính:

```text
Frontend tĩnh  <->  Backend FastAPI  <->  Model Server trên Colab
                                      <->  Ollama refine server (tuỳ chọn)
```

## Tính năng

- Tải tài liệu từ văn bản nhập trực tiếp hoặc file PDF.
- Trích xuất text từ PDF bằng PyMuPDF.
- Chia tài liệu thành các chunk theo đề mục, bullet hoặc câu.
- Gọi model summarization trên Colab để tóm tắt từng phần tài liệu.
- Retrieval các chunk liên quan cho câu hỏi của người dùng.
- Có thể dùng Ollama để tinh chỉnh câu trả lời cuối cùng từ context truy xuất.
- Frontend HTML/CSS/JavaScript chạy trực tiếp, không cần build tool.

## Cấu trúc thư mục

```text
.
├── data/
│   └── download_dataset.py              # Script tải và chuẩn bị dataset VietNews
├── models/
│   ├── vit5-summarization-adapter/      # LoRA adapter đã fine-tune
│   └── vit5-summarization-lora/         # Checkpoint huấn luyện cục bộ
├── notebooks/
│   ├── Model_Serving_API(Colab).ipynb   # Serve model summarization/retrieval qua Colab
│   ├── Ollama_Pinggy(Colab).ipynb       # Tunnel Ollama/refine server
│   ├── Summarization_Evaluate(Colab).ipynb
│   └── Summarization_Train(Kaggle).ipynb
├── src/
│   ├── backend/
│   │   ├── api/routes.py                # Các endpoint FastAPI
│   │   ├── core/                        # Schema và store runtime
│   │   ├── services/                    # Xử lý document và gọi model server
│   │   ├── utils/                       # Grounding check
│   │   ├── run.py                       # Entrypoint backend
│   │   ├── .env.example                 # Mẫu cấu hình môi trường
│   │   └── requirements.txt             # Dependency riêng cho backend
│   └── frontend/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── requirements.txt                     # Dependency tổng cho toàn project
└── .gitignore
```

## Yêu cầu môi trường

- Python 3.10 trở lên.
- Trình duyệt web hiện đại.
- Google Colab có GPU nếu muốn chạy model ViT5-large.
- Ollama là tuỳ chọn, chỉ cần nếu muốn dùng refine server cho phần hỏi đáp.

## Cài đặt

Tạo môi trường ảo và cài dependency từ thư mục gốc project:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

File `requirements.txt` ở root đang include `src/backend/requirements.txt` và thêm `datasets` cho script chuẩn bị dữ liệu. Cách này giúp backend giữ dependency riêng, còn root dùng như file cài đặt tổng cho toàn dự án.

## Cấu hình backend

Sao chép file cấu hình mẫu:

```bash
copy src\backend\.env.example src\backend\.env
```

Các biến môi trường chính:

```env
COLAB_API_URL=http://localhost:8001
OLLAMA_REFINE_URL=http://localhost:11434
BACKEND_PORT=8000
```

- `COLAB_API_URL`: URL của Model Server chạy trên Colab hoặc tunnel Pinggy.
- `OLLAMA_REFINE_URL`: URL Ollama/refine server. Có thể để trống nếu không dùng.
- `BACKEND_PORT`: port chạy backend local.

## Chạy Model Server trên Colab

1. Mở `notebooks/Model_Serving_API(Colab).ipynb` trên Google Colab.
2. Bật GPU trong `Runtime > Change runtime type`.
3. Chạy lần lượt các cell để cài thư viện, load model và mở API.
4. Sao chép URL tunnel được tạo ra, ví dụ:

```text
https://example.a.pinggy.io
```

5. Gán URL đó vào `COLAB_API_URL` trong `src/backend/.env`.

Nếu sử dụng Ollama/refine server, chạy thêm `notebooks/Ollama_Pinggy(Colab).ipynb` và cập nhật `OLLAMA_REFINE_URL`.

## Chạy backend

Chạy từ thư mục backend:

```bash
cd src/backend
python run.py
```

Mặc định backend chạy tại:

```text
http://localhost:8000
```

Kiểm tra trạng thái:

```text
http://localhost:8000/health
```

## Chạy frontend

Frontend là ứng dụng tĩnh, có thể mở trực tiếp file:

```text
src/frontend/index.html
```

Frontend mặc định gọi backend tại:

```javascript
const BACKEND_URL = 'http://localhost:8000';
```

Nếu đổi port backend, cập nhật lại biến `BACKEND_URL` trong `src/frontend/app.js`.

## API chính

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| `GET` | `/health` | Kiểm tra trạng thái backend và model server |
| `POST` | `/upload/text` | Tải văn bản thô lên backend |
| `POST` | `/upload/pdf` | Tải file PDF lên backend |
| `POST` | `/summarize` | Tóm tắt tài liệu đã tải |
| `POST` | `/ask` | Hỏi đáp dựa trên tài liệu đã tải |
| `GET` | `/document/info` | Xem thông tin tài liệu hiện tại |
| `DELETE` | `/document` | Xoá tài liệu khỏi bộ nhớ runtime |

## Chuẩn bị dataset

Script `data/download_dataset.py` dùng thư viện `datasets` để tải dataset `nam194/vietnews` và tách thành train/validation/test.

Chạy script:

```bash
python data/download_dataset.py
```

Lưu ý: script hiện lưu dữ liệu theo đường dẫn tương đối `train/`, `validation/`, `test/` tại thư mục đang chạy lệnh. Các thư mục dữ liệu sinh ra đã được đưa vào `.gitignore` để tránh commit file JSON lớn.

## Quy ước Git

Các file không nên commit:

- File môi trường thật: `.env`, `src/backend/.env`.
- Dataset sinh ra: `data/train/`, `data/validation/`, `data/test/`, `*.json` trong `data`.
- Checkpoint/model artifact nặng: `*.safetensors`, `*.pt`, `*.pth`, `*.bin`, checkpoint training.
- Cache Python, notebook checkpoint, virtual environment.

Các file nên commit:

- Source code trong `src/`.
- Notebook phục vụ train/evaluate/serve.
- Script chuẩn bị dữ liệu.
- File cấu hình mẫu như `.env.example`.
- README và requirements.

## Ghi chú vận hành

- Backend chỉ lưu tài liệu trong bộ nhớ runtime. Khi restart backend, tài liệu đã tải sẽ mất.
- PDF scan dạng ảnh không có text layer sẽ không trích xuất được nội dung. Cần OCR trước khi upload.
- URL tunnel miễn phí như Pinggy thường hết hạn sau một phiên chạy, nên cần cập nhật lại `.env` khi Colab restart.
- Model ViT5-large cần GPU và RAM/VRAM đủ lớn, phù hợp chạy trên Colab hơn là máy local cấu hình thấp.
