from main import *
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from window import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,*args,obj=None,**kwargs):
        super(MainWindow,self).__init__(*args,**kwargs)

        canvas = QtGui.QPixmap(1700, 1400)
        self.setupUi(self)
        self.pixelmap.setPixmap(canvas)
        self.pixelmap.setFixedHeight(1400)
        self.pixelmap.setFixedWidth(1400)
        self.pen = QtGui.QPen()
        self.setFixedSize(1700,1450)

        self.initial_state, self.n = experiment('HILL CLIMBING', 'ORDERED', [5, 10, 25], 1000, [0.2, 0.3, 0.5],
                                                0.5, 22, False)


        self.btn_init_state.clicked.connect(self.draw_init_state)
        self.btn_fin_state.clicked.connect(self.draw_something)


    def draw_something(self):
        painter = QtGui.QPainter(self.pixelmap.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(15)
        pen.setColor(QtGui.QColor('blue'))
        painter.setPen(pen)
        painter.drawLine(
            QtCore.QPoint(100, 100),
            QtCore.QPoint(300, 200)
        )
        painter.end()

    def draw_init_state(self):
        painter = QtGui.QPainter(self.pixelmap.pixmap())
        pen = QtGui.QPen()
        painter.setPen(pen)
        for i in self.initial_state.dict:
            pen.setWidth(10)
            pen.setColor(QtGui.QColor('green'))
            painter.setPen(pen)
            x1 = self.initial_state.centrals[i].CoordX * 14
            y1 = self.initial_state.centrals[i].CoordY * 14

            painter.drawPoint(x1,y1)

            for cl in self.initial_state.dict[i]:
                pen.setWidth(5)
                x2 = self.initial_state.clients[cl].CoordX * 14
                y2 = self.initial_state.clients[cl].CoordY * 14

                if self.initial_state.clients[cl].Contrato == 0:
                    pen.setColor(QtGui.QColor('yellow'))
                    painter.setPen(pen)
                    painter.drawPoint(x2, y2)
                else:
                    pen.setColor(QtGui.QColor('red'))
                    painter.setPen(pen)
                    painter.drawPoint(x2, y2)

                pen.setWidth(1)
                pen.setColor(QtGui.QColor('white'))
                painter.setPen(pen)
                painter.drawLine(x1,y1,x2,y2)

        for i in self.initial_state.left:
            pen.setWidth(5)
            pen.setColor(QtGui.QColor('red'))
            painter.setPen(pen)
            x2 = self.initial_state.clients[i].CoordX * 14
            y2 = self.initial_state.clients[i].CoordY * 14
            painter.drawPoint(x2, y2)
        print('printed')
        painter.end()


    def draw_fin_state(self):
        painter = QtGui.QPainter(self.pixelmap.pixmap())
        pen = QtGui.QPen()
        painter.setPen(pen)
        for i in self.n.dict:
            pen.setWidth(10)
            pen.setColor(QtGui.QColor('green'))
            painter.setPen(pen)
            x1 = self.n.centrals[i].CoordX * 14
            y1 = self.n.centrals[i].CoordY * 14

            painter.drawPoint(x1, y1)

            for cl in self.n.dict[i]:
                pen.setWidth(5)
                x2 = self.n.clients[cl].CoordX * 14
                y2 = self.n.clients[cl].CoordY * 14

                if self.n.clients[cl].Contrato == 0:
                    pen.setColor(QtGui.QColor('yellow'))
                    painter.setPen(pen)
                    painter.drawPoint(x2, y2)
                else:
                    pen.setColor(QtGui.QColor('red'))
                    painter.setPen(pen)
                    painter.drawPoint(x2, y2)

                pen.setWidth(1)
                pen.setColor(QtGui.QColor('white'))
                painter.setPen(pen)
                painter.drawLine(x1, y1, x2, y2)

        for i in self.n.left:
            pen.setWidth(5)
            pen.setColor(QtGui.QColor('red'))
            painter.setPen(pen)
            x2 = self.n.clients[i].CoordX * 14
            y2 = self.n.clients[i].CoordY * 14
            painter.drawPoint(x2, y2)

        painter.end()


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()