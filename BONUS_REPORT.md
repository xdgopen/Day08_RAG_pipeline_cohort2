# Báo Cáo Bonus

Báo cáo này chỉ đề cập các tiêu chí bonus thuộc phần bài cá nhân. Những yêu cầu thuộc bài tập nhóm, triển khai sản phẩm, hoặc UI/UX được ghi rõ là không thực hiện theo yêu cầu.

## Phạm Vi

| Tiêu chí bonus | Điểm | Trạng thái | Ghi chú |
|---|---:|---|---|
| Giải thích lexical search và BM25 | 5 | Đã bổ sung giải thích | Báo cáo phân biệt lexical search là nhóm phương pháp, BM25 là một thuật toán cụ thể. |
| Implement HyDE cho query | 5 | Đã implement | Có module `src/bonus_hyde.py` và đã tích hợp vào pipeline Task 9. |
| Deploy chatbot online | 4 | Bỏ qua | Đây là phần triển khai sản phẩm, thuộc phạm vi bài nhóm. |
| Conversation memory | 3 | Bỏ qua | Bộ nhớ hội thoại nhiều lượt thuộc hướng chatbot của bài nhóm. |
| UI/UX chất lượng | 3 | Bỏ qua | UI/UX thuộc sản phẩm nhóm, không thêm vào phần bài cá nhân. |

## Pipeline Cá Nhân Đã Hoàn Thành

Mặc dù không claim điểm bonus, pipeline cá nhân đã hoàn thành đến Task 10:

- Task 5 semantic search: `src/task5_semantic_search.py`
- Task 6 lexical search bằng BM25: `src/task6_lexical_search.py`
- Task 7 reranking bằng MMR và helper RRF: `src/task7_reranking.py`
- Task 8 tích hợp PageIndex fallback: `src/task8_pageindex_vectorless.py`
- Task 9 hybrid retrieval pipeline có HyDE: `src/task9_retrieval_pipeline.py`
- Task 10 generation có citation: `src/task10_generation.py`
- Bonus HyDE: `src/bonus_hyde.py`

## Lexical Search Khác BM25 Như Thế Nào?

`Lexical search` là tên gọi chung cho nhóm phương pháp tìm kiếm dựa trên từ khóa hoặc token xuất hiện trực tiếp trong văn bản. Nó không cần hiểu ngữ nghĩa sâu bằng embedding; thay vào đó, nó so khớp query với các từ có mặt trong tài liệu.

`BM25` là một thuật toán cụ thể thuộc nhóm lexical search. Nói cách khác:

- lexical search là một loại phương pháp retrieval;
- BM25 là một cách triển khai lexical search;
- các phương pháp lexical khác có thể là boolean search, TF-IDF, Elasticsearch/Lucene scoring hoặc keyword matching đơn giản.

Trong Task 6, hệ thống dùng BM25. BM25 chấm điểm mỗi chunk dựa trên:

- term frequency: từ khóa xuất hiện nhiều hơn trong chunk thì điểm cao hơn;
- inverse document frequency: từ hiếm trong toàn corpus có trọng số cao hơn;
- document length normalization: chunk dài không được ưu tiên quá mức chỉ vì chứa nhiều từ hơn.

Trong code, văn bản tiếng Việt được tokenize bằng regex hỗ trợ Unicode để giữ dấu tiếng Việt và số điều luật. Sau đó hệ thống xây dựng `BM25Okapi` trên các chunk đã tạo ở Task 4 và trả về kết quả theo điểm BM25 giảm dần.

## HyDE Đã Được Implement

HyDE là viết tắt của Hypothetical Document Embeddings. Thay vì embed trực tiếp query ngắn của người dùng, HyDE sinh một tài liệu giả định có dạng gần giống câu trả lời hoặc đoạn nguồn cần tìm, rồi embed tài liệu giả định đó để retrieval.

Luồng đã implement:

1. `generate_hypothetical_document(query)` trong `src/bonus_hyde.py` sinh tài liệu giả định bằng template tiếng Việt, chạy offline không cần API.
2. `hyde_search(query, top_k)` embed tài liệu giả định bằng cùng embedding model của Task 4 thông qua `semantic_search_by_text(...)`.
3. Task 9 chạy HyDE song song với semantic search và BM25.
4. Kết quả semantic, BM25 và HyDE được fusion bằng RRF.
5. Kết quả fusion tiếp tục được rerank bằng MMR.

Lý do chọn HyDE offline bằng template: repo hiện tại có `.env` placeholder cho API key, nên cách này vẫn chạy được trong môi trường chấm/test mà không phụ thuộc LLM bên ngoài. Khi có API key thật, có thể thay template bằng LLM để sinh hypothetical document giàu ngữ cảnh hơn.

## Các Mục Bonus Thuộc Bài Nhóm Đã Bỏ Qua

Các mục sau được cố ý không thực hiện:

- deploy chatbot online;
- conversation memory;
- UI/UX chất lượng.

Những mục này thuộc phạm vi sản phẩm nhóm và chỉ nên xử lý trong `group_project/` khi nhóm quyết định xây dựng demo cuối cùng.
