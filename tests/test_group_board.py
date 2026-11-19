from types import SimpleNamespace


def test_card_design_and_isolation():
    from PySide6 import QtWidgets as W
    from patchclamp_gui.gui.group_cards import GroupBoard
    app = W.QApplication.instance() or W.QApplication([])
    cells = [SimpleNamespace(name=name, animal="A", family=lambda: "EPSC", group=group)
             for name,group in [("one","WT"),("two","KO"),("three","")]]
    board = GroupBoard(None,cells,"EPSC",{c.name:c.group for c in cells},{})
    ok = board.buttons.button(W.QDialogButtonBox.Ok)
    assert not ok.isEnabled()
    board.identity.setCurrentIndex(2)
    assert ok.isEnabled()
    board.move_cards(["three"], "KO")
    assert board.mapping()["three"] == "KO" and cells[-1].group == "" and not ok.isEnabled()
    board.reject()
