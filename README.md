# LTFSR Meta

Repository này đang được tối ưu để chạy trên Kaggle theo luồng ngắn gọn:
dataset -> baseline -> lưu checkpoint -> kiểm tra nhanh.

Mục tiêu hiện tại là chuẩn hóa phần dữ liệu CIFAR-100-LT để có thể upload lên Kaggle và chạy lại baseline một cách ổn định. Các ý tưởng meta-learning/prototype/contrastive trong `config/` là roadmap, còn script chạy thực tế trong repo hiện tại là baseline ResNet18.

## Hiện có gì chạy được

- Chuẩn bị CIFAR-10, CIFAR-100, hoặc CIFAR-100-LT bằng [data/prepare_datasets.py](data/prepare_datasets.py)
- Kiểm tra layout dataset bằng [data/validate_cifar_lt.py](data/validate_cifar_lt.py)
- Train baseline bằng [training/train_baseline.py](training/train_baseline.py)

## Cấu trúc dataset chuẩn

Sau khi chuẩn bị xong, thư mục dataset sẽ có dạng:

```text
CIFAR-100-LT/
├── class_counts.json
├── dataset_info.json
├── test_manifest.csv
├── train_manifest.csv
├── train/
│   ├── class_000/
│   ├── class_001/
│   └── ...
└── test/
    ├── class_000/
    ├── class_001/
    └── ...
```

Đây là layout `ImageFolder`, nên training code có thể đọc trực tiếp từ `train/` và `test/`.

## Cách dùng trên Kaggle

### 1. Upload dữ liệu

Bạn có thể upload một trong hai kiểu sau lên Kaggle:

- Upload thẳng thư mục dataset đã chuẩn bị, ví dụ `CIFAR-100-LT/`
- Upload cả repo và để script tự tải CIFAR từ `torchvision` trong Kaggle notebook / Kaggle script

Nếu đã có dataset được chuẩn bị sẵn ở máy local, chỉ cần đưa nguyên thư mục `CIFAR-100-LT/` lên Kaggle Dataset.

### 2. Chuẩn bị dataset

Trên Kaggle, chạy:

```bash
python data/prepare_datasets.py --dataset cifar100-lt --data_dir /kaggle/working/data/CIFAR-100-LT --overwrite
```

Nếu muốn cache raw files ở chỗ khác, thêm `--raw_dir`:

```bash
python data/prepare_datasets.py --dataset cifar100-lt --data_dir /kaggle/working/data/CIFAR-100-LT --raw_dir /kaggle/working/raw --overwrite
```

Script sẽ tạo lại `train/`, `test/`, `train_manifest.csv`, `test_manifest.csv`, `class_counts.json`, và `dataset_info.json`.

### 3. Kiểm tra dataset

```bash
python data/validate_cifar_lt.py --data_dir /kaggle/working/data/CIFAR-100-LT
```

### 4. Chạy baseline

```bash
python training/train_baseline.py --config config/config_baseline.yaml --data_dir /kaggle/working/data/CIFAR-100-LT --output_dir /kaggle/working/results/baseline
```

Nếu bạn muốn giảm thời gian chạy trên Kaggle, có thể giới hạn số mẫu bằng:

```bash
python training/train_baseline.py --config config/config_baseline.yaml --data_dir /kaggle/working/data/CIFAR-100-LT --output_dir /kaggle/working/results/baseline --max_train_samples 5000 --max_test_samples 1000
```

## Khuyến nghị cho Kaggle

- Dùng `--device cpu` nếu notebook không có GPU hoặc muốn test nhanh.
- Dùng `--max_train_samples` và `--max_test_samples` để smoke test trước khi chạy full.
- Giữ `data_dir` trỏ đúng thư mục chứa `train/` và `test/`.

## Config baseline

Config baseline hiện tại nằm ở [config/config_baseline.yaml](config/config_baseline.yaml). Nó đã được đặt sẵn cho CIFAR-100-LT và dùng 100 lớp.

## Lưu ý về phần chưa hoàn thiện

- Các file config meta-learning/full model hiện có, nhưng trainer tương ứng chưa có trong repo này.
- Vì vậy, README này chỉ mô tả luồng chạy thực tế hiện tại: dataset -> baseline.
- Khi bạn thêm trainer cho meta-learning, có thể mở rộng README từ cấu trúc này mà không cần đổi lại phần dataset.

## Tóm tắt luồng chạy

1. Chuẩn bị dataset CIFAR-100-LT.
2. Validate layout dataset.
3. Train baseline ResNet18.
4. Lưu checkpoint trong `results/baseline/`.

## Các file chính

- [data/prepare_datasets.py](data/prepare_datasets.py)
- [data/validate_cifar_lt.py](data/validate_cifar_lt.py)
- [training/train_baseline.py](training/train_baseline.py)
- [config/config_baseline.yaml](config/config_baseline.yaml)
