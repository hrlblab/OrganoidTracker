"""A source-pixel scene: view zoom and display DPI never change annotations."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from ..services.annotations import AnnotationSet


class AnnotationCanvas(QGraphicsView):
    organoid_clicked = Signal(float, float)
    cyst_drawn = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.image_item = self.scene().addPixmap(QPixmap())
        self.image_item.setZValue(-1)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#20242b"))
        self.annotations = AnnotationSet(())
        self.annotation_items = []
        self.editable = False
        self._press = None
        self._pan = None
        self._box = None

    def set_frame(self, rgb, *, fit=False):
        pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width = pixels.shape[:2]
        image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888).copy()
        self.image_item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(0, 0, width, height)
        if fit:
            self.fit_image()

    def fit_image(self):
        self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor):
        if 0.015 <= self.transform().m11() * factor <= 30:
            self.scale(factor, factor)

    def set_annotations(self, annotations, visible=True):
        self.annotations = annotations
        for item in self.annotation_items:
            self.scene().removeItem(item)
        self.annotation_items = []
        if not visible:
            return
        pen = QPen(QColor("#ffc857"), 2)
        pen.setCosmetic(True)
        for organoid in annotations.organoids:
            x, y = organoid.point
            marker = self.scene().addEllipse(x - 5, y - 5, 10, 10, pen)
            label = self.scene().addSimpleText(str(organoid.organoid_id))
            label.setBrush(QColor("#ffc857"))
            label.setPos(x + 7, y)
            label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations)
            self.annotation_items.extend((marker, label))
            for cyst in organoid.cysts:
                x1, y1, x2, y2 = cyst.bbox
                rect = self.scene().addRect(x1, y1, x2 - x1, y2 - y1, pen)
                label = self.scene().addSimpleText(f"C{cyst.cyst_id}")
                label.setBrush(QColor("#ffc857"))
                label.setPos(x1, y1)
                label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations)
                self.annotation_items.extend((rect, label))

    def source_point(self, viewport_point):
        point = self.mapToScene(viewport_point)
        rect = self.image_item.boundingRect()
        return QPointF(min(max(point.x(), 0), rect.width()), min(max(point.y(), 0), rect.height()))

    def wheelEvent(self, event):
        self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        event.accept()

    def mousePressEvent(self, event):
        position = event.position().toPoint()
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._pan = position
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif (
            event.button() == Qt.MouseButton.LeftButton
            and self.editable
            and self.image_item.contains(self.mapToScene(position))
        ):
            self._press = (position, self.source_point(position))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        position = event.position().toPoint()
        if self._pan is not None:
            delta = position - self._pan
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan = position
        elif self._press is not None:
            rect = QRectF(self._press[1], self.source_point(position)).normalized()
            if self._box is None:
                pen = QPen(QColor("#59c3c3"), 2, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                self._box = self.scene().addRect(rect, pen)
            self._box.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
        elif self._press is not None and event.button() == Qt.MouseButton.LeftButton:
            position, start = self._press
            self._press = None
            if self._box is not None:
                self.scene().removeItem(self._box)
                self._box = None
            if (event.position().toPoint() - position).manhattanLength() < 5:
                self.organoid_clicked.emit(start.x(), start.y())
            else:
                rect = QRectF(start, self.source_point(event.position().toPoint())).normalized()
                if rect.width() >= 1 and rect.height() >= 1:
                    self.cyst_drawn.emit(rect.left(), rect.top(), rect.right(), rect.bottom())
        else:
            super().mouseReleaseEvent(event)
