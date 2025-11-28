## Подготовка данных и окружения

Все команды ниже выполняются из корня репозитория `plane-detection-classification`.

### 1. Загрузка COCO 2017

```bash
# перейти в корень репозитория (если ещё нет)
cd /mnt/plane-detection-classification

mkdir -p data/coco
cd data/coco

# Качаем изображения
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Качаем аннотации
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Распаковываем
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# возвращаемся в корень проекта
cd ../..

### 2. Создание окружения

conda create -n planes python=3.11 -y
conda activate planes

# Альтернатива без conda:
# python -m venv venv
# source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate     # Windows

pip install \
  ultralytics \
  torch torchvision \
  pycocotools \
  opencv-python \
  tqdm \
  onnx onnxruntime
