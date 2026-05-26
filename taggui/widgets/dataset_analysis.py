from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QDockWidget, QHBoxLayout, QLabel, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)
from transformers import PreTrainedTokenizerBase

from models.image_list_model import ImageListModel
from models.tag_counter_model import TagCounterModel


class DatasetAnalysis(QDockWidget):
    def __init__(self, image_list_model, tag_counter_model, tokenizer, tag_separator):
        super().__init__()
        self.image_list_model = image_list_model
        self.tag_counter_model = tag_counter_model
        self.tokenizer = tokenizer
        self.tag_separator = tag_separator

        self.setObjectName("dataset_analysis")
        self.setWindowTitle("Dataset Analysis")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        top_bar = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Analysis")
        self.refresh_button.setFixedHeight(32)
        self.status_label = QLabel("Click Refresh to analyze the dataset.")
        top_bar.addWidget(self.refresh_button)
        top_bar.addWidget(self.status_label, 1)

        self.tabs = QTabWidget()

        # Overview tab
        overview_layout = QVBoxLayout()
        overview_layout.addWidget(QLabel("Health Score and high-level dataset metrics will appear here."))
        overview_layout.addStretch()
        self.overview_tab = QWidget()
        self.overview_tab.setLayout(overview_layout)
        self.tabs.addTab(self.overview_tab, "Overview")

        # Caption Quality tab
        quality_layout = QVBoxLayout()
        quality_layout.addWidget(QLabel("Caption length stats, token distribution, vocabulary richness, etc."))
        quality_layout.addStretch()
        self.quality_tab = QWidget()
        self.quality_tab.setLayout(quality_layout)
        self.tabs.addTab(self.quality_tab, "Caption Quality")

        # Tag Redundancy tab
        redundancy_layout = QVBoxLayout()
        redundancy_layout.addWidget(QLabel("Similar / redundant tags detected with rapidfuzz will be listed here."))
        redundancy_layout.addStretch()
        self.redundancy_tab = QWidget()
        self.redundancy_tab.setLayout(redundancy_layout)
        self.tabs.addTab(self.redundancy_tab, "Tag Redundancy")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top_bar)
        layout.addWidget(self.tabs, 1)
        self.setWidget(container)

        self.refresh_button.clicked.connect(self.run_analysis)
        self._set_empty_state()

    def _set_empty_state(self):
        self.status_label.setText("No directory loaded or analysis not yet run.")

    @Slot()
    def run_analysis(self):
        images = self.image_list_model.images
        if not images:
            self.status_label.setText("Load a directory first.")
            return
        self.status_label.setText(f"Analysis complete ({len(images)} images).")
