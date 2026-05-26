from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QDockWidget, QFormLayout, QHBoxLayout, QLabel,
                               QPushButton, QTabWidget, QVBoxLayout, QWidget)
from transformers import PreTrainedTokenizerBase

from models.image_list_model import ImageListModel
from models.tag_counter_model import TagCounterModel

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


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

        # ========== CAPTION QUALITY TAB (real implementation) ==========
        self._create_quality_tab()

        # ========== OVERVIEW / HEALTH SCORE TAB ==========
        self._create_overview_tab()

        # ========== TAG REDUNDANCY TAB ==========
        self._create_redundancy_tab()

        # (Inconsistency metrics are currently shown in the Overview tab for simplicity)

        # ========== INCONSISTENCIES TAB ==========

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top_bar)
        layout.addWidget(self.tabs, 1)
        self.setWidget(container)

        self.refresh_button.clicked.connect(self.run_analysis)
        self._set_empty_state()

    def _create_quality_tab(self):
        """Create the Caption Quality tab with real metrics labels."""
        self.quality_total_images = QLabel("0")
        self.quality_avg_tokens = QLabel("0.0")
        self.quality_avg_words = QLabel("0.0")
        self.quality_avg_chars = QLabel("0.0")
        self.quality_short_pct = QLabel("0%")
        self.quality_long_pct = QLabel("0%")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("Total Images", self.quality_total_images)
        form.addRow("Avg Tokens per Caption", self.quality_avg_tokens)
        form.addRow("Avg Words per Caption", self.quality_avg_words)
        form.addRow("Avg Characters per Caption", self.quality_avg_chars)
        form.addRow("% Very Short Captions (<15 tokens)", self.quality_short_pct)
        form.addRow("% Very Long Captions (>80 tokens)", self.quality_long_pct)

        self.quality_tab = QWidget()
        layout = QVBoxLayout(self.quality_tab)
        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(self.quality_tab, "Caption Quality")

    def _create_overview_tab(self):
        """Create the Overview tab with Health Score."""
        self.health_score_label = QLabel("—")
        self.health_score_label.setStyleSheet("font-size: 32pt; font-weight: bold;")
        self.health_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.health_diversity = QLabel("Diversity: —")
        self.health_consistency = QLabel("Consistency: —")
        self.health_coverage = QLabel("Coverage: —")

        # Inconsistency signals
        self.incon_undertagged = QLabel("—")
        self.incon_rare_tags = QLabel("—")
        self.incon_tag_range = QLabel("—")

        form = QFormLayout()
        form.addRow("Overall Health Score", self.health_score_label)
        form.addRow(self.health_diversity)
        form.addRow(self.health_consistency)
        form.addRow(self.health_coverage)
        form.addRow("Undertagged images (<< 30% avg tags)", self.incon_undertagged)
        form.addRow("Rare tags (appear 1-2 times)", self.incon_rare_tags)
        form.addRow("Tags per image (min / max)", self.incon_tag_range)

        self.overview_tab = QWidget()
        layout = QVBoxLayout(self.overview_tab)
        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(self.overview_tab, "Overview")

    def _create_redundancy_tab(self):
        """Create the Tag Redundancy tab."""
        self.redundancy_status = QLabel()
        self.redundancy_list = QLabel("No data yet. Click Refresh Analysis.")
        self.redundancy_list.setWordWrap(True)
        self.redundancy_list.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout()
        layout.addWidget(self.redundancy_status)
        layout.addWidget(self.redundancy_list, 1)

        self.redundancy_tab = QWidget()
        self.redundancy_tab.setLayout(layout)
        self.tabs.addTab(self.redundancy_tab, "Tag Redundancy")

    def _set_empty_state(self):
        self.status_label.setText("No directory loaded or analysis not yet run.")

    def _compute_health_score(self, images, token_counts):
        """Compute a simple but useful 0-100 health score."""
        if not images:
            return 0, 0, 0, 0

        total = len(images)
        total_tag_instances = sum(len(im.tags) for im in images)
        unique_tags = len(self.tag_counter_model.tag_counter)

        # Diversity: more unique tags relative to instances is generally better
        diversity = 0
        if total_tag_instances > 0:
            diversity = min(100, (unique_tags / total_tag_instances) * 180)

        # Consistency: lower variance in caption length is better for training
        if token_counts:
            mean = sum(token_counts) / len(token_counts)
            variance = sum((x - mean) ** 2 for x in token_counts) / len(token_counts)
            std = variance ** 0.5
            consistency = max(0, 100 - (std * 1.8))   # tuned heuristic
        else:
            consistency = 50

        # Coverage: penalize too many very short or very long captions
        short = sum(1 for t in token_counts if t < 15)
        long = sum(1 for t in token_counts if t > 85)
        bad_ratio = (short + long) / total
        coverage = max(0, 100 - (bad_ratio * 120))

        # Final weighted score
        score = (0.35 * diversity + 0.35 * consistency + 0.30 * coverage)
        score = max(0, min(100, int(score)))

        return score, int(diversity), int(consistency), int(coverage)

    def _compute_redundancies(self, images):
        """Find similar tags using rapidfuzz (only on most common tags for speed)."""
        if not RAPIDFUZZ_AVAILABLE:
            return "rapidfuzz not installed. Run: pip install rapidfuzz"

        if not self.tag_counter_model.most_common_tags:
            return "No tags found."

        # Only look at the top 150 most common tags to keep it fast
        tags_to_check = [tag for tag, _ in self.tag_counter_model.most_common_tags[:150]]

        similar = []
        seen = set()

        for i, tag1 in enumerate(tags_to_check):
            for tag2 in tags_to_check[i+1:]:
                if (tag1, tag2) in seen or (tag2, tag1) in seen:
                    continue
                score = fuzz.ratio(tag1.lower(), tag2.lower())
                if score >= 86:   # fairly strict
                    similar.append((tag1, tag2, score))
                    seen.add((tag1, tag2))

        similar.sort(key=lambda x: x[2], reverse=True)

        if not similar:
            return "No highly similar tags found in the top tags."

        lines = []
        for t1, t2, sc in similar[:25]:   # show top 25
            lines.append(f"• {t1} ≈ {t2}  ({sc:.0f}%)")

        return "\n".join(lines)

    @Slot()
    def run_analysis(self):
        """Run real analysis calculations on the current dataset."""
        images = self.image_list_model.images
        if not images:
            self.status_label.setText("Load a directory first.")
            return

        self.status_label.setText("Analyzing...")

        total = len(images)
        token_counts = []
        word_counts = []
        char_counts = []

        for image in images:
            caption = self.tag_separator.join(image.tags)
            tokens = len(self.tokenizer(caption).input_ids) - 2
            words = len(caption.split())
            chars = len(caption)

            token_counts.append(tokens)
            word_counts.append(words)
            char_counts.append(chars)

        # === Caption Quality ===
        avg_tokens = sum(token_counts) / total
        avg_words = sum(word_counts) / total
        avg_chars = sum(char_counts) / total

        short_count = sum(1 for t in token_counts if t < 15)
        long_count = sum(1 for t in token_counts if t > 80)

        short_pct = (short_count / total) * 100
        long_pct = (long_count / total) * 100

        self.quality_total_images.setText(str(total))
        self.quality_avg_tokens.setText(f"{avg_tokens:.1f}")
        self.quality_avg_words.setText(f"{avg_words:.1f}")
        self.quality_avg_chars.setText(f"{avg_chars:.1f}")
        self.quality_short_pct.setText(f"{short_pct:.1f}%")
        self.quality_long_pct.setText(f"{long_pct:.1f}%")

        # === Health Score ===
        health, div, cons, cov = self._compute_health_score(images, token_counts)
        self.health_score_label.setText(f"{health}")
        self.health_diversity.setText(f"Diversity: {div}")
        self.health_consistency.setText(f"Consistency: {cons}")
        self.health_coverage.setText(f"Coverage: {cov}")

        # === Inconsistency signals ===
        avg_tags = sum(len(im.tags) for im in images) / total
        undertagged = sum(1 for im in images if len(im.tags) < (avg_tags * 0.3))
        rare_tags = sum(1 for _, count in self.tag_counter_model.most_common_tags if count <= 2)
        min_tags = min(len(im.tags) for im in images)
        max_tags = max(len(im.tags) for im in images)

        self.incon_undertagged.setText(f"{undertagged} / {total}")
        self.incon_rare_tags.setText(str(rare_tags))
        self.incon_tag_range.setText(f"{min_tags} / {max_tags}")

        # === Tag Redundancy ===
        redundancy_text = self._compute_redundancies(images)
        self.redundancy_list.setText(redundancy_text)

        if not RAPIDFUZZ_AVAILABLE:
            self.redundancy_status.setText("Install rapidfuzz for full redundancy detection.")
        else:
            self.redundancy_status.setText(f"Found similar tags among the most common ones.")

        self.status_label.setText(f"Analysis complete ({total} images).")
