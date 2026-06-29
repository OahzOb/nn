"""
扫雷游戏 - PySide6 入门实战
覆盖概念：QMainWindow, QGridLayout, QPushButton, 信号槽, QMessageBox, QSS, QTimer
"""
import sys
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QMenuBar, QMenu
)
from PySide6.QtCore import Qt, QTimer, Signal

# ========== 游戏核心逻辑 ==========

class MineCell(QPushButton):
    """单个格子 —— 继承 QPushButton，封装格子的状态和行为"""
    leftClicked = Signal(int, int)   # 自定义信号：左键点击，传递行列坐标
    rightClicked = Signal(int, int)  # 自定义信号：右键点击

    def __init__(self, row, col):
        super().__init__()
        self.row = row
        self.col = col
        self.is_mine = False
        self.is_revealed = False
        self.is_flagged = False
        self.adjacent_mines = 0
        self.setFixedSize(32, 32)
        self.setStyleSheet("""
            QPushButton {
                background-color: #bdbdbd;
                border: 2px outset #dcdcdc;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #d0d0d0; }
        """)

    def mousePressEvent(self, event):
        """重写鼠标事件 —— 区分左右键"""
        if event.button() == Qt.LeftButton:
            self.leftClicked.emit(self.row, self.col)
        elif event.button() == Qt.RightButton:
            self.rightClicked.emit(self.row, self.col)

    def reveal(self):
        """揭开格子"""
        if self.is_revealed or self.is_flagged:
            return
        self.is_revealed = True
        self.setEnabled(False)
        if self.is_mine:
            self.setText("💣")
            self.setStyleSheet("""
                QPushButton {
                    background-color: #ff4444;
                    border: 1px solid #999;
                    font-size: 16px;
                }
            """)
        else:
            colors = ["", "#0000ff", "#008000", "#ff0000", "#000080",
                      "#800000", "#008080", "#000000", "#808080"]
            self.setText(str(self.adjacent_mines) if self.adjacent_mines > 0 else "")
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: #e0e0e0;
                    border: 1px solid #999;
                    color: {colors[self.adjacent_mines] if self.adjacent_mines < len(colors) else '#000'};
                    font-weight: bold;
                    font-size: 14px;
                }}
            """)

    def toggle_flag(self):
        """切换旗帜标记"""
        if self.is_revealed:
            return
        self.is_flagged = not self.is_flagged
        if self.is_flagged:
            self.setText("🚩")
            self.setStyleSheet("""
                QPushButton {
                    background-color: #bdbdbd;
                    border: 2px outset #dcdcdc;
                    font-size: 14px;
                }
            """)
        else:
            self.setText("")
        return self.is_flagged  # 返回是否插旗，方便外部统计


class MinesweeperGame(QMainWindow):
    """主窗口 —— QMainWindow 提供菜单栏、状态栏等完整骨架"""

    def __init__(self, rows=9, cols=9, mines=10):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.total_mines = mines
        self.remaining_mines = mines
        self.cells = {}
        self.game_over = False
        self.first_click = True
        self.revealed_count = 0
        self.timer = QTimer()
        self.elapsed_seconds = 0

        self._init_ui()
        self.timer.timeout.connect(self._update_timer)

    def _init_ui(self):
        """初始化界面 —— 布局、菜单、信号连接"""
        self.setWindowTitle(f"扫雷  {self.rows}×{self.cols}  💣×{self.total_mines}")
        self.setFixedSize(self.cols * 32 + 30, self.rows * 32 + 100)

        # 菜单栏
        menu_bar = self.menuBar()
        game_menu = menu_bar.addMenu("游戏")
        for label, r, c, m in [
            ("初级 9×9", 9, 9, 10),
            ("中级 16×16", 16, 16, 40),
            ("高级 16×30", 16, 30, 99),
        ]:
            action = game_menu.addAction(label)
            action.triggered.connect(lambda checked, rr=r, cc=c, mm=m: self._restart(rr, cc, mm))
        game_menu.addSeparator()
        action = game_menu.addAction("重新开始")
        action.triggered.connect(lambda: self._restart(self.rows, self.cols, self.total_mines))

        # 中央控件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 顶部信息栏
        info_layout = QHBoxLayout()
        self.mine_label = QLabel(f"💣 {self.remaining_mines}")
        self.mine_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.time_label = QLabel("⏱ 0")
        self.time_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.mine_label)
        info_layout.addStretch()
        info_layout.addWidget(self.time_label)
        main_layout.addLayout(info_layout)

        # 雷区网格 —— QGridLayout 的核心用法
        grid_widget = QWidget()
        self.grid_layout = QGridLayout(grid_widget)
        self.grid_layout.setSpacing(1)

        for r in range(self.rows):
            for c in range(self.cols):
                cell = MineCell(r, c)
                cell.leftClicked.connect(self._on_left_click)
                cell.rightClicked.connect(self._on_right_click)
                self.grid_layout.addWidget(cell, r, c)
                self.cells[(r, c)] = cell

        main_layout.addWidget(grid_widget, alignment=Qt.AlignCenter)

    # ========== 游戏逻辑 ==========

    def _place_mines(self, safe_row, safe_col):
        """首次点击后布雷 —— 保证第一次点击安全"""
        safe_zone = {(safe_row + dr, safe_col + dc)
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if 0 <= safe_row + dr < self.rows and 0 <= safe_col + dc < self.cols}
        candidates = [(r, c) for r in range(self.rows) for c in range(self.cols)
                      if (r, c) not in safe_zone]
        mine_positions = random.sample(candidates, self.total_mines)
        for r, c in mine_positions:
            self.cells[(r, c)].is_mine = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.cells[(nr, nc)].is_mine:
                        self.cells[(nr, nc)].adjacent_mines += 1

    def _on_left_click(self, row, col):
        """左键点击 —— 信号槽：cell.leftClicked → 这个方法"""
        if self.game_over or self.cells[(row, col)].is_flagged:
            return

        if self.first_click:
            self._place_mines(row, col)
            self.first_click = False
            self.timer.start(1000)  # 启动计时器

        cell = self.cells[(row, col)]
        if cell.is_mine:
            self._game_lost(row, col)
            return

        self._reveal_cell(row, col)
        if self.revealed_count == self.rows * self.cols - self.total_mines:
            self._game_won()

    def _on_right_click(self, row, col):
        """右键点击 —— 插旗/取消"""
        if self.game_over or self.first_click:
            return
        changed = self.cells[(row, col)].toggle_flag()
        self.remaining_mines += -1 if changed else 1
        self.mine_label.setText(f"💣 {self.remaining_mines}")

    def _reveal_cell(self, row, col):
        """揭开格子 + 自动展开空白区（BFS）"""
        cell = self.cells[(row, col)]
        if cell.is_revealed or cell.is_flagged:
            return
        cell.reveal()
        self.revealed_count += 1
        # 空白格自动展开
        if cell.adjacent_mines == 0 and not cell.is_mine:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self._reveal_cell(nr, nc)

    def _game_lost(self, row, col):
        """游戏失败 —— QMessageBox 的使用"""
        self.game_over = True
        self.timer.stop()
        # 揭开所有雷
        for cell in self.cells.values():
            if cell.is_mine:
                cell.reveal()
        self.cells[(row, col)].setStyleSheet("""
            QPushButton {
                background-color: #ff0000;
                border: 1px solid #999;
                font-size: 16px;
            }
        """)
        reply = QMessageBox.question(
            self, "游戏结束", "💥 踩雷了！要再来一局吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._restart(self.rows, self.cols, self.total_mines)

    def _game_won(self):
        """游戏胜利"""
        self.game_over = True
        self.timer.stop()
        for cell in self.cells.values():
            if cell.is_mine:
                cell.setText("🚩")
        reply = QMessageBox.question(
            self, "恭喜！",
            f"🎉 你赢了！用时 {self.elapsed_seconds} 秒\n再来一局？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._restart(self.rows, self.cols, self.total_mines)

    def _update_timer(self):
        """计时器回调"""
        self.elapsed_seconds += 1
        self.time_label.setText(f"⏱ {self.elapsed_seconds}")

    def _restart(self, rows, cols, mines):
        """重新开始 —— 销毁旧界面，创建新游戏"""
        self.timer.stop()
        self.rows, self.cols, self.total_mines = rows, cols, mines
        self.remaining_mines = mines
        self.cells.clear()
        self.game_over = False
        self.first_click = True
        self.revealed_count = 0
        self.elapsed_seconds = 0
        self._init_ui()


# ========== 入口 ==========

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 跨平台统一风格
    game = MinesweeperGame()
    game.show()
    sys.exit(app.exec())
