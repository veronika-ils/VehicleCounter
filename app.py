import sys
import cv2
import time
from PyQt5 import QtCore, QtGui, QtWidgets


class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    counts_updated = QtCore.pyqtSignal(int, int)
    log_msg = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.paused = False
        self.line_y = 300
        self.min_area = 500
        self.cars_in = 0
        self.cars_out = 0

        self.bgs = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=25, detectShadows=True
        )

    def open(self, path_or_index):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path_or_index)
        if not self.cap.isOpened():
            self.log_msg.emit("Could not open video source.")
            return False
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.log_msg.emit(f"Opened video: {w}x{h} @ {fps:.1f} fps")
        return True

    def run(self):
        self.running = True
        while self.running:
            if self.paused or self.cap is None:
                time.sleep(0.02)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break


            frame = cv2.resize(frame, (640, 480))


            fgmask = self.bgs.apply(frame)

            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)


                if cy < self.line_y + 5 and cy > self.line_y - 5:
                    if cx < frame.shape[1] // 2:
                        self.cars_in += 1
                        self.log_msg.emit("Car IN detected")
                    else:
                        self.cars_out += 1
                        self.log_msg.emit("Car OUT detected")
                    self.counts_updated.emit(self.cars_in, self.cars_out)


            cv2.line(frame, (0, self.line_y), (640, self.line_y), (255, 0, 0), 2)


            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            self.frame_ready.emit(img)

        if self.cap:
            self.cap.release()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Counter")
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)

        self.in_label = QtWidgets.QLabel("Cars IN: 0")
        self.out_label = QtWidgets.QLabel("Cars OUT: 0")

        open_btn = QtWidgets.QPushButton("Open Video")
        start_btn = QtWidgets.QPushButton("Start")
        pause_btn = QtWidgets.QPushButton("Pause")
        reset_btn = QtWidgets.QPushButton("Reset")

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(open_btn)
        btns.addWidget(start_btn)
        btns.addWidget(pause_btn)
        btns.addWidget(reset_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addLayout(btns)
        layout.addWidget(self.in_label)
        layout.addWidget(self.out_label)
        layout.addWidget(self.log_box)

        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.counts_updated.connect(self.update_counts)
        self.worker.log_msg.connect(self.log)

        open_btn.clicked.connect(self.open_video)
        start_btn.clicked.connect(self.start_video)
        pause_btn.clicked.connect(self.pause_video)
        reset_btn.clicked.connect(self.reset_counts)

    def update_frame(self, img):
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(img))

    def update_counts(self, cars_in, cars_out):
        self.in_label.setText(f"Cars IN: {cars_in}")
        self.out_label.setText(f"Cars OUT: {cars_out}")

    def log(self, msg):
        self.log_box.append(msg)

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video")
        if path:
            self.worker.open(path)

    def start_video(self):
        if not self.worker.isRunning():
            self.worker.start()
        self.worker.paused = False

    def pause_video(self):
        self.worker.paused = True

    def reset_counts(self):
        self.worker.cars_in = 0
        self.worker.cars_out = 0
        self.update_counts(0, 0)
        self.log("Counts reset.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
