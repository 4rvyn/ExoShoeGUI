import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import markdown
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QFontInfo
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QSplitter, QTabWidget, QTextBrowser,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget
)

def _load_and_parse_help_md(folder: Path) -> Tuple[Dict[str, str], List[Tuple[str, str, str | None]]]:
    """
    Reads and parses *.md files to build a multi-level tree structure that
    matches the original HTML-based hierarchy.
    """
    md_converter = markdown.Markdown(extensions=["fenced_code", "tables"])
    map_html: Dict[str, str] = {}
    tree_structure: List[Tuple[str, str, str | None]] = []
    
    # Manually define the logical structure to match the original app exactly.
    # This is the most robust way to guarantee the desired hierarchy.
    logical_structure = [
        ('overview', 'Overview', None),
        ('core_customization', 'Core Customization Areas', 'overview'),
        ('scenarios', 'Adding New Features (Scenarios)', 'overview'),
        ('backend_advanced', 'Backend & Core Logic (Advanced)', 'overview')
    ]
    
    # Map these logical keys to the actual filenames
    key_to_filename = {
        'overview': '00_overview.md',
        'core_customization': '01_core_customization.md',
        'scenarios': '02_scenarios.md',
        'backend_advanced': '03_backend_advanced.md'
    }

    # First, build the main parent items from the logical structure
    for key, title, parent in logical_structure:
        tree_structure.append((key, title, parent))
        # Add placeholder content that will be overwritten if the file has intro text
        map_html[key] = f"<h2>{title}</h2><p>Select a sub-topic to see details.</p>"

    # Now, parse each file and add its sub-topics
    for key, filename in key_to_filename.items():
        md_file = folder / filename
        if not md_file.exists():
            logging.warning(f"Help file not found: {md_file}")
            continue
            
        raw_md = md_file.read_text(encoding="utf-8")
        
        # Split the markdown by H3 '###' headings.
        sub_sections = re.split(r'^(### .*)$', raw_md, flags=re.MULTILINE)
        
        # The first chunk is the intro content for the main topic.
        intro_content_md = sub_sections[0].strip()
        if intro_content_md:
            map_html[key] = md_converter.convert(intro_content_md)
            md_converter.reset()

        # Process the sub-sections
        for i in range(1, len(sub_sections), 2):
            heading_line = sub_sections[i]
            sub_topic_content_md = sub_sections[i+1] if i + 1 < len(sub_sections) else ""
            
            sub_topic_title = heading_line.lstrip('#').strip()
            sub_topic_key = f"{key}_sub_{i//2}"
            
            # Add the sub-topic to the tree, parented by the file's logical key
            tree_structure.append((sub_topic_key, sub_topic_title, key))
            
            # The HTML content for the sub-topic includes its heading
            full_section_md = heading_line + "\n" + sub_topic_content_md
            map_html[sub_topic_key] = md_converter.convert(full_section_md)
            md_converter.reset()

    return map_html, tree_structure

class HelpWindow(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Application Guide")
        self.resize(1200, 700)
        self.main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        logging.info("Initializing Help Window...")
        self._setup_gui_tab()
        self._setup_code_tab()
        
        self.setLayout(self.main_layout)
        
        self._help_content_map: Dict[str, str] = {}
        self._populate_code_help_tree()
        
        if self.code_help_tree.topLevelItemCount() > 0:
            self.code_help_tree.expandAll()
            first_item = self.code_help_tree.topLevelItem(0)
            self.code_help_tree.setCurrentItem(first_item)
            self._on_code_help_item_selected(first_item, 0)

    def _setup_gui_tab(self):
        gui_tab_widget = QWidget()
        layout = QVBoxLayout(gui_tab_widget)
        browser = QTextBrowser()
        browser.setReadOnly(True)
        browser.setOpenExternalLinks(True)
        md_path = Path(__file__).parent / "help_content" / "gui_guide.md"
        if not md_path.exists():
            browser.setHtml(f"<h1>Error</h1><p>GUI help file not found:</p><pre>{md_path}</pre>")
        else:
            raw_md  = md_path.read_text(encoding="utf-8")
            browser.setHtml(markdown.markdown(raw_md, extensions=["fenced_code", "tables"]))
        layout.addWidget(browser)
        self.tab_widget.addTab(gui_tab_widget, "GUI Guide")

    def _setup_code_tab(self):
        code_tab_widget = QWidget()
        code_layout = QHBoxLayout(code_tab_widget)
        self.code_help_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.code_help_tree = QTreeWidget()
        self.code_help_tree.setHeaderLabel("Code Customization Topics")
        self.code_help_tree.setMinimumWidth(320)
        self.code_help_tree.setMaximumWidth(500)
        self.code_help_display = QTextBrowser()
        self.code_help_display.setReadOnly(True)
        font = QFont("Consolas", 10)
        font_info = QFontInfo(font)
        if not font_info.fixedPitch():
            font.setFamily("Courier New")
            font_info_fallback = QFontInfo(font)
            if not font_info_fallback.fixedPitch():
                font.setStyleHint(QFont.StyleHint.Monospace)
                font.setFamily("Monospace")
        self.code_help_display.setFont(font)
        self.code_help_display.setOpenExternalLinks(True)
        self.code_help_splitter.addWidget(self.code_help_tree)
        self.code_help_splitter.addWidget(self.code_help_display)
        self.code_help_splitter.setStretchFactor(0, 0)
        self.code_help_splitter.setStretchFactor(1, 1)
        code_layout.addWidget(self.code_help_splitter)
        self.tab_widget.addTab(code_tab_widget, "Code Customization")
        self.code_help_tree.itemClicked.connect(self._on_code_help_item_selected)

    def _add_help_topic(self, text: str, parent_item: QTreeWidget | QTreeWidgetItem, content_key: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent_item, [text])
        item.setData(0, Qt.ItemDataRole.UserRole, content_key)
        # Special styling for the 'Advanced' section to match original
        if 'advanced' in content_key.lower():
            font = item.font(0)
            font.setItalic(True)
            item.setFont(0, font)
            item.setForeground(0, QColor("slateGray"))
        return item

    def _populate_code_help_tree(self):
        folder = Path(__file__).parent / "help_content"
        if not folder.is_dir():
            self.code_help_display.setHtml(f"<h1>Error</h1><p>Help directory not found:</p><pre>{folder}</pre>")
            return

        self._help_content_map, tree_structure = _load_and_parse_help_md(folder)
        if not tree_structure:
            self.code_help_display.setHtml("<h1>Warning</h1><p>No valid topic files found.</p>")
            return
        
        key_to_widget_map: Dict[str, QTreeWidgetItem] = {}
        
        # Loop through the structured data and build the tree
        for key, title, parent_key in tree_structure:
            parent_widget = self.code_help_tree
            if parent_key:
                parent_widget = key_to_widget_map.get(parent_key)
                if not parent_widget:
                    logging.error(f"Logic Error: Could not find parent widget for key '{parent_key}' (child: '{title}').")
                    parent_widget = self.code_help_tree
            
            tree_item = self._add_help_topic(title, parent_widget, key)
            key_to_widget_map[key] = tree_item
    
    def _on_code_help_item_selected(self, item: QTreeWidgetItem, column: int):
        content_key = item.data(0, Qt.ItemDataRole.UserRole)
        if content_key:
            html_content = self._help_content_map.get(content_key, "<p>Content not found.</p>")
            self.code_help_display.setHtml(f"<html><body>{html_content}</body></html>")
            self.code_help_display.verticalScrollBar().setValue(0)