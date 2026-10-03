import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import (
    QImage,
    QPainter,
    QPixmap,
    QColor,
    QFont,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
)

MIN_CONFIDENCE = 0.70

from recognizer import create_recognition_overlay


def cv_to_qpixmap(image):
    if image is None:
        return QPixmap()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, ch = image.shape

    qimage = QImage(
        image.data,
        w,
        h,
        ch * w,
        QImage.Format_RGB888,
    ).copy()

    return QPixmap.fromImage(qimage)


class BoardWidget(QWidget):
    """
    Warped Sudoku görüntüsünü çizer.

    - Tanınan rakamlar: hücreye yeşil (yüksek güven)
      veya kırmızı (düşük güven) yarı saydam renk.
    - Dolu ama tanınamayan: turuncu (sayı yok).
    - Manuel girilen: mavi.
    - Tıkla + rakam tuşuna bas -> hücreyi düzelt.
    """

    cellChanged = Signal()

    def __init__(self, result, parent=None):
        super().__init__(parent)

        self.result = result
        self.selected = None

        self.setMinimumSize(450, 450)
        self.setFocusPolicy(Qt.StrongFocus)

    # --------------------------------------------------------
    # Layout helpers
    # --------------------------------------------------------

    def board_rect(self):
        size = min(self.width(), self.height())
        x = (self.width() - size) // 2
        y = (self.height() - size) // 2
        return QRect(x, y, size, size)

    def cell_rect(self, row, col):
        rect = self.board_rect()
        cw = rect.width() / 9.0
        ch = rect.height() / 9.0
        return QRect(
            round(rect.x() + col * cw),
            round(rect.y() + row * ch),
            round(cw),
            round(ch),
        )

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    def mousePressEvent(self, event):
        rect = self.board_rect()

        if not rect.contains(event.position().toPoint()):
            return

        fx = (event.position().x() - rect.x()) / rect.width()
        fy = (event.position().y() - rect.y()) / rect.height()

        col = min(8, max(0, int(fx * 9)))
        row = min(8, max(0, int(fy * 9)))

        self.selected = (row, col)
        self.setFocus()
        self.update()

    def keyPressEvent(self, event):
        if self.selected is None:
            return

        row, col = self.selected
        key = event.key()

        digit = None

        if Qt.Key_0 <= key <= Qt.Key_9:
            digit = key - Qt.Key_0
        elif Qt.Key_Delete == key or Qt.Key_Backspace == key:
            digit = 0

        if digit is not None:
            if digit == 0:
                self.result["manual_values"][row][col] = 0
            else:
                self.result["manual_values"][row][col] = digit

            self.cellChanged.emit()
            self.update()

    # --------------------------------------------------------
    # Paint
    # --------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor("#1b1b1b"))

        rect = self.board_rect()

        pixmap = cv_to_qpixmap(self.result["warped"])
        painter.drawPixmap(rect, pixmap, pixmap.rect())

        recognition = self.result["recognition"]
        occupied = self.result["occupied"]
        confidences = self.result["confidences"]
        manual = self.result["manual_values"]

        for row in range(9):
            for col in range(9):
                cell_rect = self.cell_rect(row, col)

                value = manual[row][col]
                is_manual = value != 0

                if not is_manual:
                    value = recognition[row][col]

                is_occupied = occupied[row][col] == 1
                confidence = confidences[row][col]

                if is_manual:
                    tint = QColor(52, 152, 219, 110)
                elif value != 0 and confidence >= MIN_CONFIDENCE:
                    tint = QColor(46, 204, 113, 95)
                elif value != 0:
                    # Çok hafif kırmızı
                    #tint = QColor(231, 76, 60, 20)
                    # asıl kırmızı
                    #tint = QColor(231, 76, 60, 70)
                    continue
                elif is_occupied:
                    # Çok hafif turuncu
                    #tint = QColor(243, 156, 18, 15)
                    # asıl turuncu
                    #tint = QColor(243, 156, 18, 70)
                    continue
                else:
                    continue

                painter.fillRect(cell_rect, tint)

        # Re-draw recognized digits on top
        font = QFont("Arial", 20, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(15, 15, 15))

        small = QFont("Arial", 8)

        for row in range(9):
            for col in range(9):
                value = manual[row][col]
                is_manual = value != 0

                if not is_manual:
                    value = recognition[row][col]

                cell_rect = self.cell_rect(row, col)

                if value != 0:
                    painter.setFont(font)
                    painter.drawText(
                        cell_rect,
                        Qt.AlignCenter,
                        str(value),
                    )

                if (
                    not is_manual
                    and recognition[row][col] != 0
                ):
                    painter.setFont(small)
                    painter.setPen(QColor(30, 30, 30))
                    painter.drawText(
                        cell_rect.adjusted(3, 2, -3, -3),
                        Qt.AlignTop | Qt.AlignLeft,
                        f"{confidences[row][col]:.2f}",
                    )
                    painter.setPen(QColor(15, 15, 15))

        # Selection highlight
        if self.selected is not None:
            row, col = self.selected
            pen = QPen(QColor(255, 255, 255), 3)
            painter.setPen(pen)
            painter.drawRect(
                self.cell_rect(row, col).adjusted(1, 1, -1, -1)
            )

        painter.setFont(QFont("Arial", 9))
        painter.setPen(QColor("#cccccc"))
        painter.drawText(
            self.rect().adjusted(8, 4, -8, -4),
            Qt.AlignTop | Qt.AlignLeft,
            "Yeşil: güvenilir tanıma | Kırmızı: düşük güven | "
            "Turuncu: okunamadı (sayı yok) | Mavi: manuel. "
            "Tıkla, rakam gir; Delete = temizle.",
        )


class Thumbnail(QWidget):
    def __init__(self, title, image, parent=None):
        super().__init__(parent)
        self.title = title
        self.pixmap = cv_to_qpixmap(image)
        self.setMinimumSize(160, 160)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#242424"))
        if not self.pixmap.isNull():
            scaled = self.pixmap.scaled(
                self.width() - 8,
                self.height() - 28,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            painter.drawPixmap(
                4,
                20,
                scaled,
            )
        painter.setPen(QColor("#dddddd"))
        painter.drawText(4, 14, self.title)


class MainWindow(QMainWindow):
    def __init__(self, results):
        super().__init__()
        self.results = results
        self.index = 0

        self.setWindowTitle("Sudoku Recognizer")
        self.resize(1400, 900)

        self.board = BoardWidget(self.results[0])
        self.board.cellChanged.connect(self.update_status)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.grids = {}
        for title, key in (
            ("Original", "original"),
            ("Warped", "warped"),
            ("Clean", "clean"),
            ("Grid Debug", "grid_debug"),
            ("Overlay", "overlay"),
        ):
            thumb = Thumbnail(title, None)
            self.grids[key] = thumb
            right_layout.addWidget(thumb)

        nav = QWidget()
        nav_layout = QHBoxLayout(nav)

        self.prev_btn = QPushButton("← Önceki")
        self.next_btn = QPushButton("Sonraki →")
        self.prev_btn.clicked.connect(self.previous)
        self.next_btn.clicked.connect(self.next)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        right_layout.addWidget(nav)

        splitter = QSplitter()
        splitter.addWidget(self.board)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(splitter)

        self.setCentralWidget(container)

        status = QStatusBar()
        self.setStatusBar(status)

        self.show_current()

    def show_current(self):
        result = self.results[self.index]
        self.board.result = result
        self.board.selected = None
        self.board.update()

        key_map = {
            "original": "original",
            "warped": "warped",
            "clean": "clean",
            "grid_debug": "grid_debug",
            "overlay": "overlay",
        }

        for key, thumb in self.grids.items():
            image = result.get(key)
            thumb.pixmap = cv_to_qpixmap(image) if image is not None else QPixmap()
            thumb.title = key
            thumb.update()

        self.update_status()

    def update_status(self):
        result = self.results[self.index]

        filled = sum(
            1
            for r in range(9)
            for c in range(9)
            if result["manual_values"][r][c] != 0
            or result["recognition"][r][c] != 0
        )

        manual_count = sum(
            1
            for r in range(9)
            for c in range(9)
            if result["manual_values"][r][c] != 0
        )

        confs = [
            result["confidences"][r][c]
            for r in range(9)
            for c in range(9)
            if result["recognition"][r][c] != 0
        ]

        avg = sum(confs) / len(confs) if confs else 0.0

        result["overlay"] = create_recognition_overlay(
            result["warped"],
            result["recognition"],
            result["occupied"],
            result["confidences"],
            result["manual_values"],
        )

        self.grids["overlay"].pixmap = cv_to_qpixmap(
            result["overlay"]
        )
        self.grids["overlay"].update()

        self.statusBar().showMessage(
            f"{result['name']}  |  {self.index + 1}/{len(self.results)}"
            f"  |  Tanınan/manuel hücre: {filled}"
            f"  |  Manuel: {manual_count}"
            f"  |  Ort. güven: {avg:.2f}"
        )

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.show_current()

    def next(self):
        if self.index < len(self.results) - 1:
            self.index += 1
            self.show_current()


def run_viewer(results):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(results)
    window.show()
    app.exec()
