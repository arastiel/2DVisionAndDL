import sys, os, re
from collections import OrderedDict

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QTreeWidget, QTreeWidgetItem, \
                            QDialogButtonBox, QHeaderView

class ParseError(Exception):
    def __init__(self, error : str, file : str, lineNum : int, line : str):
        super().__init__(error)
        self.file = file
        self.lineNum = lineNum
        self.line = line

def readFile(name):
    try:
        file = open(name, 'r', encoding='utf-8-sig')
    except Exception:
        raise Exception('Die Datei "' + name + '" konnte nicht geöffnet werden.')
    result = OrderedDict()
    uebung = None
    aufgabe = 0
    sub = None
    for lineNum, line in enumerate(file):
        def raiseError(error: str):
            raise ParseError(error, file.name, lineNum + 1, line)
        if re.match('^\s*#', line) or re.match('^\s*$', line):
            continue
        m = re.match('^Uebung ([0-9]+).*', line)
        if m:
            nextUebung = int(m.group(1))
            if uebung:
                if nextUebung < uebung:
                    raiseError('Die genannte Übung steht an falscher Stelle.')
                if nextUebung == uebung:
                    raiseError('Die genannte Übung wurde bereits aufgeführt.')
            uebung = nextUebung
            aufgabe = 0
            continue
        m = re.match('^\s*Aufgabe\s+([1-9])([a-z]{0,1})\s+:\s+([0-9\.]+)\s+([a-zA-Z]+).*', line)
        if m:
            nextAufgabe = int(m.group(1))
            if nextAufgabe < aufgabe:
                raiseError('Die genannte Aufgabe steht an falscher Stelle.')
            nextSub = m.group(2)
            if nextAufgabe == aufgabe:
                if sub == '' or nextSub == '' or sub == nextSub:
                    raiseError('Diese Aufgabe wurde bereits zuvor aufgeführt.')
                if nextSub < sub:
                    raiseError('Die genannte Teilaufgabe steht an falscher Stelle.')
            aufgabe = nextAufgabe
            sub = nextSub
            try:
                punkte = float(m.group(3))
            except ValueError:
                raiseError('"' + m.group(3) + '" ist keine valide Punktzahl.')
            if m.group(4).startswith('Punkt'):
                zusatz = False
            elif m.group(4).startswith('Zusatzpunkt'):
                zusatz = True
            else:
                raiseError('Nach der Punktzahl muss entweder Punkt(e) oder Zusatzpunkt(e) folgen.')
            entry = result.setdefault(uebung, OrderedDict())
            if sub == '':
                entry[aufgabe] = (punkte, zusatz)
            else:
                entry.setdefault(aufgabe, OrderedDict())[sub] = (punkte, zusatz)
            continue
        raiseError('Die Zeile hat eine falsche Syntax.')
    return result

def showError(title : str, text : str):
    QMessageBox.critical(None, title, text)
    sys.exit(1)


class StructureError(Exception):
    def __init__(self, error :  str):
        super().__init__(str)

def isDict(o):
    return hasattr(o, 'values')

def checkStructure(max : OrderedDict, ist : OrderedDict, levelPrefix = None, level : int = 0):
    for key, val in ist.items():
        def getKeyName():
            return (levelPrefix[level] if levelPrefix else '') + ' ' + str(key)
        if key not in max:
            raise StructureError('Für ' + getKeyName() + ' existiert keine maximale Punktzahl.')
        maxVal = max[key]
        if not isDict(val) and isDict(maxVal) and val[0] != 0:
            raise StructureError('Die Punkte für ' + getKeyName() + ' sind nicht korrekt auf die ' \
                                 '(Teil)Aufgaben vergeteilt.')
        if isDict(val) and not isDict(maxVal):
            raise StructureError('Die Teilaufgabe ' + getKeyName() + ' existiert nicht.')
        if isDict(val) and isDict(maxVal):
            checkStructure(maxVal, val, getKeyName() + ' ', level + 1)


def sumPoints(d, mitZusatz : bool = None):
    if mitZusatz == None:
        mitZusatz = (type(d) == tuple)
    if type(d) == tuple:
        return d[0] if not d[1] or mitZusatz else 0
    else: # is OrderedDict
        sum = 0
        for v in d.values():
            sum += sumPoints(v, mitZusatz)
        return sum

def makeDialog(max, ist, levelPrefix = None, resizeLater : bool = True):
    dialog = QDialog()
    dialog.setWindowTitle('Punkte Status')
    dialog.setLayout(QVBoxLayout())
    treeWidget = QTreeWidget()
    firstLabel = ' / '.join(levelPrefix) if levelPrefix else ''
    treeWidget.setHeaderLabels([ firstLabel, 'Ihre Punkzahl', 'max. Punktzahl', 'Prozentsatz' ])
    buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
    buttonBox.accepted.connect(dialog.accept)
    rootItem = makeItem('Gesamt', sumPoints(ist, True), sumPoints(max))
    treeWidget.addTopLevelItem(rootItem)
    addItems(rootItem, max, ist, levelPrefix)

    treeWidget.header().setSectionResizeMode(QHeaderView.ResizeToContents)
    treeWidget.header().setDefaultAlignment(Qt.AlignCenter)
    treeWidget.header().setStretchLastSection(False)

    width = treeWidget.header().length()

    dialog.layout().addWidget(treeWidget)
    dialog.layout().addWidget(buttonBox)

    if resizeLater:
        def resize():
            dialog.resize(dialog.width() - treeWidget.width() + width + 2, dialog.height())
        QTimer.singleShot(0, resize)

    return dialog


def makeItem(name : str, ist : float, max : float):
    item = QTreeWidgetItem()
    item.setData(0, Qt.DisplayRole, name)
    item.setData(1, Qt.DisplayRole, ist)
    item.setData(2, Qt.DisplayRole, max)
    item.setData(3, Qt.DisplayRole, '{:.1f} %'.format(ist/max * 100))
    for col in range(1, 4):
        item.setTextAlignment(col, Qt.AlignRight)
    return item

def addItems(treeItem : QTreeWidgetItem, maxEntry : OrderedDict, istEntry, levelPrefix = None, level : int = 0):
    for key, val in maxEntry.items():
        name = (levelPrefix[level] if levelPrefix else '') + ' ' + str(key)
        if type(val) == tuple and val[1]:
            name += ' (Zusatz)'
        if not isDict(istEntry) or key not in istEntry:
            treeItem.addChild(makeItem(name, 0, sumPoints(val)))
        else:
            istVal = istEntry[key]
            childItem = makeItem(name, sumPoints(istVal, True), sumPoints(val))
            treeItem.addChild(childItem)
            if isDict(val) and isDict(istVal):
                addItems(childItem, val, istVal, levelPrefix, level + 1)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    try:
        max = readFile('max_points.txt')
        ist = readFile('your_points.txt')
    except ParseError as e:
        showError('Fehler in Punktedatei',
                  'Fehler in Datei "' + e.file + '" in Zeile ' + str(e.lineNum) + ':\n\n' + e.line + '\n' + str(e))

    levelPrefix = ['Übung ', 'Aufgabe ', '']
    try:
        checkStructure(max, ist, levelPrefix)
    except StructureError as e:
        showError('Strukturfehler', str(e))

    thisdir = os.path.basename(os.path.abspath('.'))
    for entry in os.listdir('..'):
        if entry == thisdir:
            continue
        dir = '../' + entry
        if os.path.isdir(dir):
            file = dir + '/your_points.txt'
            if os.path.isfile(file):
                try:
                    other = readFile(file)
                except ParseError as e:
                    QMessageBox.critical(None, 'Fehler in Punktedatei',
                         'Fehler in Datei "' + file + '" in Zeile ' + str(e.lineNum) + ':\n\n' + e.line + '\n' +
                         str(e) + '/nDie Datei wird ignoriert.')
                    continue
                try:
                    checkStructure(max, other, levelPrefix)
                except StructureError as e:
                    QMessageBox.critical(None, 'Strukturfehler', 'Strukturfehler in Datei "' + file + '":\n' +
                                         str(e) + '\nDie Datei wird ignoriert.')
                    continue
                common = ist.keys() & other.keys()
                if common:
                    QMessageBox.critical(None, 'Doppelte Einträge', 'Die Datei "' + file + '" enthält bereits ' \
                                         'vorhandene Punktzahleinträge und wird daher ignoriert.')
                    continue
                ist.update(other)

    levelPrefix[2] = 'Teilaufgabe '
    dialog = makeDialog(max, ist, levelPrefix)
    dialog.show()

    sys.exit(app.exec())