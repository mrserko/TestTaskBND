# Детекция людей на видео с использованием YOLOv5

Этот проект предназначен для обнаружения людей в видео с использованием модели YOLOv5. Программа рисует границы вокруг обнаруженных людей и сохраняет результат в выходном видеофайле.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone git@github.com:mrserko/TestTaskBND.git
   cd TestTaskBND
   ```

2. Создайте виртуальное окружение и активируйте его (необходим python >= 3.10):

   ```bash
   python -m venv venv # на Windows
   python3 -m venv venv # на Linux/Mac
   venv\Scripts\activate # на Windows
   source venv/bin/activate # на Linux/Mac
   ```

3. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

## Запуск

1. Поместите ваш видеофайл (например, crowd.mp4) в директорию проекта.

2. Запустите скрипт для детекции людей:

   ```bash
   python main.py # Windows
   python3 main.py # Linux/Mac
   ```

3. Выходное видео с обнаруженными людьми будет сохранено в файл output_crowd.mp4.

## Анализ результата
Данный проект представляет собой бэйзлайн, который может использоваться в качестве отправной точки для решения задачи детекции людей на видео. В результате В результате работы программы будут отрисованы границы вокруг обнаруженных людей и указано значение уверенности модели в правильности детекции.

## Шаги по улучшению качества работы программы

1. Использование более крупных моделей:

   Для повышения точности можно использовать более крупные модели YOLOv5, такие как yolov5m.pt, yolov5l.pt или yolov5x.pt.

2. Настройка параметров модели:

   Настройка порогового значения, при котором объект считается правильно распознанным, может помочь уменьшить количество ложных срабатываний.

3. Обучение на специфических данных:

   Можно произвести finetuning модели на данных, специально подготовленных для решения задачи распознавания людей.

3. Поддержка GPU:

   Позволит значительно ускорить процесс обработки видео.
