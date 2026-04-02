# Engineer_Drawings_Detection
III. QUY TRÌNH CÀI ĐẶT CHI TIẾT
* trước khi bắt đầu, bạn hãy vào link này để tải file epoch_23.pth về!
* https://huggingface.co/spaces/khoai3010/engineer_drawings_detection/tree/main

Bước 1: Khởi tạo môi trường từ file cấu hình
Mở công cụ dòng lệnh (Anaconda Prompt hoặc Terminal), di chuyển vào thư mục chứa mã nguồn và thực hiện lệnh:


# Tạo môi trường mới tự động từ file environment.yml
conda env create -f environment.yml
# Kiểm tra danh sách môi trường đã cài đặt
conda env list

Bước 2: Kích hoạt môi trường
Sử dụng tên môi trường đã đặt trong file (mặc định thường là tên dự án):
# Kích hoạt môi trường để bắt đầu sử dụng
conda activate [TEN_MOI_TRUONG_CUA_BAN]
Bước 3: Cài đặt tối ưu hóa (Tùy chọn cho CPU)
Nếu máy cá nhân không có GPU NVIDIA hoặc muốn chạy bản nhẹ hơn, hãy thực thi các lệnh đè sau:

Cài đặt PyTorch bản CPU:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

Cài đặt bộ thư viện OpenMMLab (Dùng MIM):
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.1.0"
mim install "mmdet>=3.3.0"

IV. HƯỚNG DẪN CHẠY ỨNG DỤNG
Sau khi môi trường đã báo "Successfully installed", thực hiện lệnh sau để khởi động giao diện:

Lệnh thực thi: python app.py

Truy cập: Mở trình duyệt và nhập địa chỉ http://127.0.0.1:7860 (địa chỉ mặc định của Gradio).

V. CÁC LỖI THƯỜNG GẶP VÀ CÁCH XỬ LÝ (TROUBLESHOOTING)
Lỗi "ResolvePackageNotFound": Mở file .yml bằng Notepad, xóa các dòng thư viện có phiên bản kèm theo số hiệu lạ của Windows hoặc các dòng liên quan đến vc, vs2015_runtime.

Lỗi thiếu File Model: Đảm bảo các file trọng số .pth đã được tải về và đặt đúng đường dẫn đã khai báo trong code.

Lỗi "ImportError: DLL load failed": Thường do thiếu thư viện Visual C++ Redistributable, hãy tải bản mới nhất từ Microsoft.
