from PySide6 import QtCore, QtWidgets as W


class Lane(W.QListWidget):
    def __init__(self, board, group):
        super().__init__()
        self.board = board
        self.group = group
        self.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(W.QAbstractItemView.DragDrop)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)
        self.setMinimumWidth(165)
        self.setSpacing(3)

    def dropEvent(self, event):
        source = event.source()
        if isinstance(source, Lane) and source.board is self.board:
            names = [item.data(QtCore.Qt.UserRole) for item in source.selectedItems()]
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            QtCore.QTimer.singleShot(0, lambda: self.board.move_cards(names, self.group))
        else:
            event.ignore()


class GroupBoard(W.QDialog):
    def __init__(self, parent, cells, family, labels, plans):
        super().__init__(parent)
        self.setWindowTitle("设置本次比较 · 分组树与卡片")
        self.resize(960, 570)
        self.cells = {cell.name: cell for cell in cells if cell.family() in ("EPSC", "IPSC")}
        self.family = family
        self.focus_group = None
        self.labels = dict(labels)
        self.decisions = {}
        self.extra = {key: set() for key in ("EPSC", "IPSC")}
        for key, (saved, decision) in plans.items():
            self.labels.update({name: label for name, label in saved.items() if name in self.cells and self.cells[name].family() == key})
            self.decisions[key] = decision
        layout = W.QVBoxLayout(self)
        layout.addWidget(W.QLabel("点击左侧类型或组名查看卡片；Ctrl / Shift 多选后拖到目标组，或点「移入所选」。不修改原始标签。"))
        split = W.QSplitter()
        layout.addWidget(split)
        self.tree = W.QTreeWidget()
        self.tree.setHeaderLabel("类型 / 实验组")
        self.tree.itemClicked.connect(self.navigate)
        split.addWidget(self.tree)
        self.scroll = W.QScrollArea()
        self.scroll.setWidgetResizable(True)
        split.addWidget(self.scroll)
        split.setSizes([220, 720])
        tools = W.QHBoxLayout()
        layout.addLayout(tools)
        add = W.QPushButton("＋ 新建组")
        add.clicked.connect(self.new_group)
        tools.addWidget(add)
        self.identity = W.QComboBox()
        self.identity.addItems(["确认跨组重复编号的含义…", "确实是同一只动物", "各组独立动物，编号重复"])
        self.identity.currentIndexChanged.connect(self.decide)
        tools.addWidget(self.identity)
        self.note = W.QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.buttons = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        self.buttons.button(W.QDialogButtonBox.Ok).setText("应用当前类型的比较方案")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.draw()

    def mapping(self):
        return {name:self.labels.get(name, "") for name, cell in self.cells.items() if cell.family() == self.family}

    def navigate(self, item, column):
        self.family, self.focus_group = item.data(0, QtCore.Qt.UserRole)
        self.draw()

    def new_group(self):
        name, ok = W.QInputDialog.getText(self, "新建实验组", "组名")
        if ok and name.strip():
            self.extra[self.family].add(name.strip())
            self.draw()

    def move_cards(self, names, group):
        for name in names:
            if name in self.mapping():
                self.labels[name] = group
        self.decisions[self.family] = 0
        self.draw()

    def decide(self, index):
        self.decisions[self.family] = index
        self.validate()

    def validate(self):
        assigned = {name:group for name,group in self.mapping().items() if group}
        animals = {}
        for name, group in assigned.items():
            animal = self.cells[name].animal.strip()
            if animal:
                animals.setdefault(animal, set()).add(group)
        overlaps = sorted(key for key, groups in animals.items() if len(groups)>1)
        self.identity.setVisible(bool(overlaps))
        ok = len(set(assigned.values())) >= 2 and (not overlaps or self.identity.currentIndex() > 0)
        self.buttons.button(W.QDialogButtonBox.Ok).setEnabled(bool(ok))
        self.note.setText(f"{self.family} · 纳入 {len(assigned)} 条，未分组 {len(self.mapping())-len(assigned)} 条。" +
            ("跨组重复编号：" + "、".join(overlaps) if overlaps else "至少需要两个非空组。"))

    def draw(self):
        self.tree.clear()
        for family in ("EPSC","IPSC"):
            root = W.QTreeWidgetItem(self.tree, [family])
            root.setData(0, QtCore.Qt.UserRole, (family, None))
            mapping = {n:self.labels.get(n,"") for n,c in self.cells.items() if c.family()==family}
            for group in sorted({"", *mapping.values(), *self.extra[family]}):
                names = [n for n,g in mapping.items() if g==group]
                count = len({self.cells[n].animal.strip() for n in names} - {""})
                item = W.QTreeWidgetItem(root, [f"{group or '未分组'} · {count} 动物 / {len(names)} 细胞"])
                item.setData(0, QtCore.Qt.UserRole, (family,group))
                if family == self.family and group == self.focus_group:
                    self.tree.setCurrentItem(item)
            root.setExpanded(family == self.family)
            if family == self.family and self.focus_group is None:
                self.tree.setCurrentItem(root)
        content = W.QWidget()
        layout = W.QHBoxLayout(content)
        self.lanes = []
        mapping = self.mapping()
        for group in sorted({"", *mapping.values(), *self.extra[self.family]}):
            column = W.QVBoxLayout()
            layout.addLayout(column)
            names = [n for n,g in mapping.items() if g==group]
            column.addWidget(W.QLabel(f"{group or '未分组'}（{len(names)}）"))
            lane = Lane(self, group)
            self.lanes.append(lane)
            column.addWidget(lane)
            if self.focus_group in (None, group):
                for name in names:
                    item = W.QListWidgetItem(f"{name}\n动物：{self.cells[name].animal or '未填写'}", lane)
                    item.setData(QtCore.Qt.UserRole,name)
            button = W.QPushButton("移入所选")
            button.clicked.connect(lambda _, dest=group: self.move_cards(
                [item.data(QtCore.Qt.UserRole) for source in self.lanes for item in source.selectedItems()], dest))
            column.addWidget(button)
        old = self.scroll.takeWidget()
        if old:
            old.deleteLater()
        self.scroll.setWidget(content)
        self.identity.blockSignals(True)
        self.identity.setCurrentIndex(self.decisions.get(self.family, 0))
        self.identity.blockSignals(False)
        self.validate()
