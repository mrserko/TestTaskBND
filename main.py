from ultralytics import YOLO
import cv2

def detect_people_in_video(video_path: str, output_path: str, model_weights: str = "yolov5su.pt"):
    """
    Производит детекцию людей на видео с использованием модели YOLOv5 и сохраняет результат.

    Args:
        video_path (str): Путь к входному файлу.
        output_path (str): Путь к выходному файлу.
        model_weights (str): Путь к файлу с весами модели YOLOv5. По умолчанию "yolov5su.pt".
    """
    # Загружаем модель
    model = YOLO(model_weights)

    # Открываем обрабатываемый файл с видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Сохраняем параметры входного файла
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Определяем объекты для записи результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        # Читаем видео кадр за кадром
        ret, frame = cap.read()
        if not ret:
            break

        # Применяем модель к кадру
        results = model(frame)

        for result in results:
            # Получаем границы объектов, скор и метки классов
            boxes = result.boxes.xyxy.cpu().numpy()  # границы объектов
            confidences = result.boxes.conf.cpu().numpy()  # скор
            class_ids = result.boxes.cls.cpu().numpy()  # метки классов

            # Итерируемся по всем обнаруженным объектам
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                # Проверяем, что объект принадлежит нужному классу
                if class_id == 0:
                    # Координаты границ объекта
                    x1, y1, x2, y2 = map(int, box)

                    # Рисуем границы
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Рисуем скор
                    label = f"Person: {confidence * 100:.2f}%"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Записываем кадр в выходной файл
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "crowd.mp4"
    output_video_path = "output_crowd.mp4"

    detect_people_in_video(input_video_path, output_video_path)
