import sys
import os
import cv2
import json
from PySide6 import QtWidgets, QtGui, QtCore
from UI.ui_form import Ui_MainWindow
from Pipeline.library import extract_frames, checkbox_style, button_style, progress_style, QWidget_style, textbox_style, calculate_pipe_lengths
from multiprocessing import Pool
import segmentation_models_pytorch as smp
import torch
import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ImageLabel(QtWidgets.QLabel):
    squareDrawn = QtCore.Signal(QtCore.QRect)
    lineDrawn = QtCore.Signal(tuple)  # (start, end)

    def __init__(self, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = main_window # store reference
        self.start_pos = None
        self.end_pos = None
        self.drawing_mode = "rect"  # "rect" or "line"
        self.rect = None
        self.line = None
        self._pixmap = None
        self.force_vertical = True
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        super().setPixmap(pixmap)

    def set_mode(self, mode):
        self.drawing_mode = mode
        self.start_pos = None
        self.end_pos = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.start_pos = event.position().toPoint()
            self.end_pos = event.position().toPoint()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            if self.drawing_mode == "rect":
                # force square
                dx = event.position().x() - self.start_pos.x()
                dy = event.position().y() - self.start_pos.y()
                side = max(abs(dx), abs(dy))
                self.end_pos = QtCore.QPoint(
                    self.start_pos.x() + (side if dx >= 0 else -side),
                    self.start_pos.y() + (side if dy >= 0 else -side)
                )
            elif self.drawing_mode == "line":
                if self.main_window.checkVertical.isChecked():
                    self.end_pos = QtCore.QPoint(self.start_pos.x(), event.position().toPoint().y())
                else:
                    self.end_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.start_pos and self.end_pos:
            if self.drawing_mode == "rect":
                self.rect = QtCore.QRect(self.start_pos, self.end_pos).normalized()
                self.squareDrawn.emit(self.rect)
            elif self.drawing_mode == "line":
                self.line = (self.start_pos, self.end_pos)
                self.lineDrawn.emit(self.line)
            self.start_pos = None
            self.end_pos = None
            self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        # draw image
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            painter.drawPixmap(self.contentsRect(), scaled)

        # draw overlays
        pen = QtGui.QPen(QtGui.QColor("red"), 2, QtCore.Qt.SolidLine)
        painter.setPen(pen)

        if self.drawing_mode == "rect" and (self.start_pos and self.end_pos):
            painter.drawRect(QtCore.QRect(self.start_pos, self.end_pos).normalized())
        elif self.drawing_mode == "rect" and self.rect:
            painter.drawRect(self.rect)

        elif self.drawing_mode == "line" and (self.start_pos and self.end_pos):
            painter.drawLine(self.start_pos, self.end_pos)
        elif self.drawing_mode == "line" and self.line:
            painter.drawLine(self.line[0], self.line[1])


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.imageLabel = ImageLabel(self)

        self.btnSelectFolder.setStyleSheet(button_style)
        self.btnSetCrop.setStyleSheet(button_style)
        self.btnSaveCalibration.setStyleSheet(button_style)
        self.btnNextImage.setStyleSheet(button_style)
        self.btnBatchCrop.setStyleSheet(button_style)
        self.btnRunBatch.setStyleSheet(button_style)
        self.progressBarVideo.setStyleSheet(progress_style)
        self.progressBar.setStyleSheet(progress_style)
        self.progressBar_2.setStyleSheet(progress_style)

        self.btnChecks.setStyleSheet(checkbox_style)
        self.btnAmbientLight.setStyleSheet(checkbox_style)
        self.btnDisplayLight.setStyleSheet(checkbox_style)
        self.btnGenerate.setStyleSheet(button_style)
        self.txtComments.setStyleSheet(textbox_style)
        self.widget_12.setStyleSheet(QWidget_style)

        self.checkbox_group = QtWidgets.QButtonGroup(self)
        self.checkbox_group.setExclusive(True)
        self.checkbox_group.addButton(self.checkVertical)
        self.checkbox_group.addButton(self.checkOther)

        self.checkVertical.setStyleSheet(checkbox_style)
        self.checkOther.setStyleSheet(checkbox_style)
        self.checkVertical.setChecked(True)  # Default selection

        self.checkVertical.stateChanged.connect(self.on_vertical_checked)
        self.checkOther.stateChanged.connect(self.on_other_checked)

        self.imageLabel = ImageLabel(main_window=self, parent=self.lblImageDisplay.parent())
        self.imageLabel.setGeometry(self.lblImageDisplay.geometry())  # same size & position
        self.imageLabel.setPixmap(self.lblImageDisplay.pixmap())
        self.imageLabel.setScaledContents(self.lblImageDisplay.hasScaledContents())

        self.lblImageDisplay.hide()  # keep formatting, just hide it

        # Connect drawing signals
        self.imageLabel.squareDrawn.connect(self.on_square_drawn)
        self.imageLabel.lineDrawn.connect(self.on_line_drawn)

        self.show_image("graphics/Thesis - graphics.png")

        self.btnSelectFolder.clicked.connect(self.select_folder)
        self.btnSetCrop.clicked.connect(self.on_set_crop)
        self.btnSaveCalibration.clicked.connect(self.on_save_calibration)
        self.btnNextImage.setEnabled(False)
        self.btnNextImage.clicked.connect(self.next_image)
        self.btnRunBatch.setEnabled(False)
        self.btnBatchCrop.setEnabled(False)
        self.btnBatchCrop.clicked.connect(self.batch_crop_all_folders)
        self.progressBar.setEnabled(False)
        self.progressBar_2.setEnabled(False)
        self.progressBarVideo.setEnabled(False)
        self.btnRunBatch.clicked.connect(self.batch_segment_all_folders)

        self.rect = None
        self.line = None

    def on_vertical_checked(self, checked):
        if checked:
            self.imageLabel.force_vertical = True

    def on_other_checked(self, checked):
        if checked:
            self.imageLabel.force_vertical = False

    def show_image(self, image_path):
        """Display an image in the image label."""
        pixmap = QtGui.QPixmap(image_path)
        self.lblImageGuideCrop.setPixmap(pixmap)
        self.lblImageGuideCrop.setScaledContents(True)

    def select_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Data Folder")
        self.progressBarVideo.setEnabled(True)

        if folder_path:
            scannerName = self.txtScannerName.text().strip()
            self.lblScannerNameResults.setText(str(scannerName))

            self.lblStatus.setText("Status: Extracting frames...")

            self.lblFolderPath.setText(folder_path)
            self.video_folders = [os.path.join(folder_path, f)
                                  for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            self.current_video_idx = 0
            self.cropping_coordinates = {}
            self.conversion_factors = {}

            self.worker = FrameExtractorWorker(folder_path, extract_frames)
            self.worker.progress.connect(self.progressBarVideo.setValue)  # Link progress bar
            self.worker.finished.connect(self.load_first_image_for_cropping)
            self.worker.start()

    def on_square_drawn(self, rect):
        self.rect = rect

    def on_line_drawn(self, line):
        self.line = line

    def update_next_button_state(self):
        self.btnNextImage.setEnabled(self.rect is not None and self.line is not None)

    def load_first_image_for_cropping(self):
        if self.current_video_idx >= len(self.video_folders):
            self.btnNextImage.setEnabled(False)
            self.btnBatchCrop.setEnabled(True)
            return

        current_folder = self.video_folders[self.current_video_idx]
        self.btnSelectFolder.setEnabled(False)

        # Update folder number label
        self.lblFolderNumber.setText(f"{self.current_video_idx + 1} / {len(self.video_folders)}")
        self.lblStatus.setText("Status: Editing frames...")

        frames_dir = os.path.join(current_folder, "frames")
        first_image_path = os.path.join(frames_dir, sorted(os.listdir(frames_dir))[0])

        self.current_image_path = first_image_path
        self.rect = None
        self.line = None
        self.update_next_button_state()

        # Load image into ImageLabel
        pixmap = QtGui.QPixmap(first_image_path)

        # Scale pixmap to fit into the label while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.imageLabel.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.setScaledContents(False)  # make sure QLabel doesn't stretch further

        # Reset drawing stuff
        self.imageLabel.rect = None
        self.imageLabel.line = None

    def on_set_crop(self):
        if not self.imageLabel.rect:
            return

        pm = self.imageLabel._pixmap
        if not pm:
            return
        # scale label rect to image coordinates
        scale_w = pm.width() / self.imageLabel.width()
        scale_h = pm.height() / self.imageLabel.height()
        rect = QtCore.QRect(
            int(self.imageLabel.rect.x() * scale_w),
            int(self.imageLabel.rect.y() * scale_h),
            int(self.imageLabel.rect.width() * scale_w),
            int(self.imageLabel.rect.height() * scale_h),
        )

        # 👉 Store rect in absolute pixel coords
        self.cropping_coordinates[self.current_video_idx] = rect

        cropped = pm.copy(rect)
        final = cropped.scaled(
            256, 256,
            QtCore.Qt.IgnoreAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.imageLabel.setScaledContents(False)
        self.imageLabel.setPixmap(final)
        self.imageLabel.setFixedSize(final.size())

        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.imageLabel.drawing_mode = "line"
        self.imageLabel.line = None

        self.frameConversion.setEnabled(True)

    def on_save_calibration(self):
        """Save calibration line + distance."""

        if not self.imageLabel.line:
            return

        dist_mm_str = self.txtDistanceMm.text().strip()
        if not dist_mm_str:
            return

        try:
            dist_mm = float(dist_mm_str)
        except ValueError:
            return

        # Extract coordinates
        start_point, end_point = self.imageLabel.line
        x1, y1 = start_point.x(), start_point.y()
        x2, y2 = end_point.x(), end_point.y()

        dx = x2 - x1
        dy = y2 - y1

        # Compute total pixel distance along line
        line_length_px = (dx ** 2 + dy ** 2) ** 0.5

        if line_length_px == 0:
            return

        if self.checkVertical.isChecked():
            # Vertical line: full pixel distance corresponds to mm
            vertical_distance_px = abs(dy)
            conversion_factor = dist_mm / vertical_distance_px
        else:
            # Angled line: project user distance onto vertical axis
            vertical_distance_px = abs(dy)
            vertical_distance_mm = dist_mm * (vertical_distance_px / line_length_px)
            conversion_factor = vertical_distance_mm / vertical_distance_px

        # Store result
        self.conversion_factors[self.current_video_idx] = conversion_factor

        self.lblStatus.setText(f"Calibration saved: {conversion_factor:.4f} mm/pixel")
        self.btnNextImage.setEnabled(True)

        self.btnNextImage.setEnabled(True)

    def next_image(self):
        """Go to next folder or finish all."""
        self.current_video_idx += 1
        if self.current_video_idx >= len(self.video_folders):
            # last folder → disable buttons and batch crop
            self.btnNextImage.setEnabled(False)
            self.btnBatchCrop.setEnabled(True)
            self.btnSaveCalibration.setEnabled(True)
            self.btnSetCrop.setEnabled(True)

            # reset label back to full display mode
            self.imageLabel.setScaledContents(True)
            self.imageLabel.setFixedSize(self.lblImageDisplay.size())  # reset to container size
            self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
            pixmap = QtGui.QPixmap("graphics/blank.png")
            self.imageLabel.setPixmap(pixmap)
            return

        # reset drawing state
        self.imageLabel.drawing_mode = "rect"
        self.imageLabel.start_pos = None
        self.imageLabel.end_pos = None
        self.imageLabel.rect = None
        self.imageLabel.line = None

        # reset label back to full display mode
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setFixedSize(self.lblImageDisplay.size())  # reset to container size
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.txtDistanceMm.clear()

        # load new first image
        self.load_first_image_for_cropping()

    def batch_crop_all_folders(self):
        self.lblStatus.setText("Status: Batch cropping frames...")
        self.progressBar_2.setEnabled(True)
        self.btnRunBatch.setEnabled(True)
        self.btnSetCrop.setEnabled(False)
        self.btnSaveCalibration.setEnabled(False)
        self.worker = BatchCropWorker(self.video_folders, self.cropping_coordinates)
        self.worker.progress.connect(self.progressBar_2.setValue)
        self.worker.start()

    def batch_segment_all_folders(self):
        self.lblStatus.setText("Status: Batch segmenting images...")
        self.progressBar.setEnabled(True)
        self.btnRunBatch.setEnabled(False)

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=2).to(device)
        model.load_state_dict(torch.load('Model/ChrisScanners_model_weights_unetXresnet18_310725.pth', map_location=torch.device('cpu')))
        model.eval()

        self.worker = BatchSegmentWorker(self.video_folders, model, device, self.conversion_factors)

        self.worker.progress.connect(self.progressBar.setValue)
        self.worker.finished.connect(self.export_results)
        self.worker.start()

    def update_excel_with_pipe_lengths(xlsx_path, pipe_lengths):
        wb = openpyxl.load_workbook(xlsx_path)

        sheet_name = "PipeLengths"
        # Remove existing sheet if it exists
        if sheet_name in wb.sheetnames:
            std = wb[sheet_name]
            wb.remove(std)

        # Create fresh sheet
        ws = wb.create_sheet(sheet_name)
        ws.append(["Diameter_mm", "PipeLength_mm"])  # headers

        # Insert new rows
        for diameter, length in pipe_lengths:
            ws.append([diameter, length])

        wb.save(xlsx_path)

    def export_results(self):
        xlsx_path = "experiment_results.xlsx"
        exporter = ResultsExporter(xlsx_path)

        # Cropping
        crop_rows = [
            [idx, rect.x(), rect.y(), rect.width(), rect.height()]
            for idx, rect in self.cropping_coordinates.items()
        ]
        exporter.add_sheet("Cropping", ["FolderIdx", "X", "Y", "Width", "Height"], crop_rows)

        # Calibration
        calib_rows = [
            [idx, factor] for idx, factor in self.conversion_factors.items()
        ]
        exporter.add_sheet("Calibration", ["FolderIdx", "ConversionFactor"], calib_rows)

        # Segmentation
        seg_rows = [
            [folder,
             self.worker.min_top_distance.get(folder),
             self.worker.max_bottom_distance.get(folder)]
            for folder in self.video_folders
        ]
        exporter.add_sheet("Segmentation", ["Folder", "MinTopDistance_mm", "MaxBottomDistance_mm"], seg_rows)

        # Save first (so file exists before update_excel_with_pipe_lengths runs)
        exporter.save()

        # PipeLengths
        pipe_lengths = calculate_pipe_lengths(self.lblFolderPath.text())  # data_dir
        self.update_excel_with_pipe_lengths(xlsx_path, pipe_lengths)

        self.lblStatus.setText(f"Status: Results exported to {xlsx_path}")
        self.plot_pipe_lengths_in_label(self.lblResults, xlsx_path="experiment_results.xlsx")

    def plot_pipe_lengths_in_label(self, lbl, xlsx_path="experiment_results.xlsx"):

        # --- Read pipe lengths from Excel ---
        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        if "PipeLengths" not in wb.sheetnames:
            return
        ws = wb["PipeLengths"]
        rows = list(ws.iter_rows(values_only=True))[1:]  # skip header

        if not rows:
            return

        diameters = np.array([float(d) for d, l in rows])
        lengths = np.array([float(l) for d, l in rows])
        inv_diameters = 1 / diameters

        # --- Calculate AUC ---
        auc = np.trapz(lengths, x=inv_diameters)

        # --- Create Matplotlib figure ---
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(inv_diameters, lengths, marker='o', linestyle='-', color='blue', label='Pipe lengths')
        ax.set_xlabel("1 / Pipe Diameter (1/mm)")
        ax.set_ylabel("Pipe Length (mm)")
        ax.set_title(f"Pipe Length vs alpha (R={auc:.3f})")
        ax.grid(True)

        # --- Draw rectangle with area = AUC ---
        D_R = inv_diameters.max()  # choose max x-axis value for width
        L_R = auc / D_R  # height to match area
        rect_x = [0, D_R, D_R, 0, 0]
        rect_y = [0, 0, L_R, L_R, 0]
        ax.plot(rect_x, rect_y, color='green', linestyle='--', label='AUC rectangle')

        # --- Draw diagonal ---
        ax.plot([0, D_R], [L_R, 0], color='red', linestyle=':', label='Diagonal')

        # --- Annotate rectangle intersections ---
        ax.text(-0.01 * D_R, L_R, f"L_R={L_R:.2f} cm", color='green', va='bottom')
        ax.text(D_R, -0.05 * L_R, f"D_R={D_R:.2f} 1/mm", color='green', ha='right')

        ax.legend()

        # --- Display figure in QLabel ---
        for child in lbl.children():
            child.setParent(None)  # remove previous canvas if exists

        canvas = FigureCanvas(fig)
        canvas.setParent(lbl)
        canvas.setFixedSize(lbl.width(), lbl.height())
        canvas.draw()


class FrameExtractorWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()

    def __init__(self, data_dir, extract_frames_func):
        super().__init__()
        self.data_dir = data_dir
        self.extract_frames = extract_frames

    def run(self):
        video_files = [f for f in os.listdir(self.data_dir) if f.endswith(".mp4")]
        video_paths = [os.path.join(self.data_dir, v) for v in video_files]
        frames_dirs = [os.path.join(self.data_dir, os.path.splitext(v)[0], "frames") for v in video_files]
        cropped_dirs = [os.path.join(self.data_dir, os.path.splitext(v)[0], "cropped") for v in video_files]

        with Pool(processes=os.cpu_count()) as pool:
            for i, _ in enumerate(pool.starmap(self.extract_frames, zip(video_paths, frames_dirs)), start=1):
                progress_percent = int((i / len(video_files)) * 100)
                self.progress.emit(progress_percent)

        self.finished.emit()


class BatchCropWorker(QtCore.QThread):
    progress = QtCore.Signal(int)

    def __init__(self, video_folders, cropping_coordinates):
        super().__init__()
        self.video_folders = video_folders
        self.cropping_coordinates = cropping_coordinates

    def run(self):
        total_images = sum(len(os.listdir(os.path.join(f, "frames"))) for f in self.video_folders)
        done = 0

        for idx, folder in enumerate(self.video_folders):
            rect = self.cropping_coordinates.get(idx)
            if not rect:
                continue

            frames_dir = os.path.join(folder, "frames")
            cropped_dir = os.path.join(folder, "cropped")
            os.makedirs(cropped_dir, exist_ok=True)

            for img_name in sorted(os.listdir(frames_dir)):
                img_path = os.path.join(frames_dir, img_name)
                pm = QtGui.QPixmap(img_path)

                cropped = pm.copy(rect)

                save_pixmap = cropped.scaled(256, 256, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                save_pixmap.save(os.path.join(cropped_dir, img_name))

                done += 1
                self.progress.emit(int(done / total_images * 100))


class BatchSegmentWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()

    def __init__(self, video_folders, model, device, conversion_factor):
        super().__init__()
        self.video_folders = video_folders
        self.model = model
        self.device = device
        self.conversion_factor = conversion_factor
        self.max_bottom_distance = {}
        self.min_top_distance = {}

    def run(self):
        total_images = sum(len(os.listdir(os.path.join(f, "cropped"))) for f in self.video_folders)
        done = 0

        for folder in self.video_folders:
            cropped_dir = os.path.join(folder, "cropped")
            pseudolabels_dir = os.path.join(folder, "pseudolabels")
            os.makedirs(pseudolabels_dir, exist_ok=True)
            conversion_factor = self.conversion_factor.get(self.video_folders.index(folder))

            max_bottom_distance = float('-inf')
            min_top_distance = float('inf')

            for img_name in sorted(os.listdir(cropped_dir)):
                img_path = os.path.join(cropped_dir, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (256, 256))
                image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                image_tensor = image_tensor.to(self.device)

                with torch.no_grad():
                    output = self.model(image_tensor)
                    pseudolabel = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8) * 255

                pseudolabel_path = os.path.join(pseudolabels_dir, img_name.replace('.png', '_pseudolabel.png'))
                cv2.imwrite(pseudolabel_path, pseudolabel)
                mask = cv2.imread(pseudolabel_path, cv2.IMREAD_GRAYSCALE)
                row_sums = np.sum(mask > 0, axis=1)
                non_zero_rows = np.where(row_sums > 0)[0]

                if len(non_zero_rows) == 0:
                    continue

                top_row = non_zero_rows[0]
                bottom_row = non_zero_rows[-1]

                # convert row indices to mm
                top_mm = top_row * conversion_factor
                bottom_mm = bottom_row * conversion_factor

                # update per-folder stats
                max_bottom_distance = max(max_bottom_distance, bottom_mm)
                min_top_distance = min(min_top_distance, top_mm)

                done += 1
                self.progress.emit(int(done / total_images * 100))

            # after finishing all images in folder, save results
            self.max_bottom_distance[folder] = (
                max_bottom_distance if max_bottom_distance > float('-inf') else None
            )
            self.min_top_distance[folder] = (
                min_top_distance if min_top_distance < float('inf') else None
            )

            # save per-folder JSON
            def save_distances_json(folder_path, min_top=None, max_bot=None):
                out_path = os.path.join(folder_path, "special_distances_unetXresnet18.json")
                data = {}
                if min_top is not None:
                    data["MinTopDistance_mm"] = min_top
                if max_bot is not None:
                    data["MaxBottomDistance_mm"] = max_bot
                with open(out_path, "w") as f:
                    json.dump(data, f, indent=2)

            save_distances_json(folder, min_top=min_top_distance, max_bot=max_bottom_distance)

        self.finished.emit()


class ResultsExporter:
    def __init__(self, save_path="experiment_results.xlsx"):
        self.save_path = save_path
        # Always create a new workbook
        self.wb = openpyxl.Workbook()
        # remove default sheet
        default = self.wb.active
        self.wb.remove(default)

    def add_sheet(self, name, headers, rows):
        ws = self.wb.create_sheet(title=name)
        ws.append(headers)
        for row in rows:
            ws.append(row)
        # auto-size columns
        for col in ws.columns:
            max_length = max(len(str(cell.value)) if cell.value else 0 for cell in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max_length + 2

    def save(self):
        self.wb.save(self.save_path)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
