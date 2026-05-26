from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QDockWidget, QFormLayout, QLabel, QVBoxLayout,
                               QWidget)
from transformers import PreTrainedTokenizerBase

from models.image_list_model import ImageListModel
from models.tag_counter_model import TagCounterModel


class TagStatistics(QDockWidget):
    def __init__(self, image_list_model: ImageListModel,
                 tag_counter_model: TagCounterModel,
                 tokenizer: PreTrainedTokenizerBase,
                 tag_separator: str):
        super().__init__()
        self.image_list_model = image_list_model
        self.tag_counter_model = tag_counter_model
        self.tokenizer = tokenizer
        self.tag_separator = tag_separator

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('tag_statistics')
        self.setWindowTitle('Tag Statistics')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.total_images_label = QLabel('0')
        self.tagged_images_label = QLabel('0 (0%)')
        self.untagged_images_label = QLabel('0 (0%)')
        self.unique_tags_label = QLabel('0')
        self.total_tag_instances_label = QLabel('0')
        self.avg_tags_per_image_label = QLabel('0.00')
        self.avg_tokens_per_image_label = QLabel('0.0')

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        form_layout.addRow('Total Images', self.total_images_label)
        form_layout.addRow('Tagged Images', self.tagged_images_label)
        form_layout.addRow('Untagged Images', self.untagged_images_label)
        form_layout.addRow('Unique Tags', self.unique_tags_label)
        form_layout.addRow('Total Tag Instances', self.total_tag_instances_label)
        form_layout.addRow('Avg Tags / Image', self.avg_tags_per_image_label)
        form_layout.addRow('Avg Tokens / Image', self.avg_tokens_per_image_label)

        # A container widget is required to use a layout with a `QDockWidget`.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(form_layout)
        self.setWidget(container)

        self.image_list_model.modelReset.connect(self.update_statistics)
        self.image_list_model.dataChanged.connect(self.update_statistics)
        self.tag_counter_model.modelReset.connect(self.update_statistics)

        self.update_statistics()

    @Slot()
    def update_statistics(self):
        images = self.image_list_model.images
        total_images = len(images)

        if total_images == 0:
            self._set_zero_state()
            return

        tagged_count = sum(1 for image in images if image.tags)
        untagged_count = total_images - tagged_count

        tagged_percent = (tagged_count / total_images) * 100
        untagged_percent = (untagged_count / total_images) * 100

        unique_tags = self.tag_counter_model.rowCount()
        total_instances = sum(self.tag_counter_model.tag_counter.values())

        avg_tags = total_instances / total_images

        total_tokens = 0
        for image in images:
            caption = self.tag_separator.join(image.tags)
            # Subtract 2 for the `<|startoftext|>` and `<|endoftext|>` tokens
            # to match the rest of the application.
            token_count = len(self.tokenizer(caption).input_ids) - 2
            total_tokens += token_count
        avg_tokens = total_tokens / total_images

        self.total_images_label.setText(f'{total_images:,}')
        self.tagged_images_label.setText(
            f'{tagged_count:,} ({tagged_percent:.1f}%)')
        self.untagged_images_label.setText(
            f'{untagged_count:,} ({untagged_percent:.1f}%)')
        self.unique_tags_label.setText(f'{unique_tags:,}')
        self.total_tag_instances_label.setText(f'{total_instances:,}')
        self.avg_tags_per_image_label.setText(f'{avg_tags:.2f}')
        self.avg_tokens_per_image_label.setText(f'{avg_tokens:.1f}')

    def _set_zero_state(self):
        self.total_images_label.setText('0')
        self.tagged_images_label.setText('0 (0%)')
        self.untagged_images_label.setText('0 (0%)')
        self.unique_tags_label.setText('0')
        self.total_tag_instances_label.setText('0')
        self.avg_tags_per_image_label.setText('0.00')
        self.avg_tokens_per_image_label.setText('0.0')
