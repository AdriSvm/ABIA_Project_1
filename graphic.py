from main import *
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from window import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,*args,obj=None,**kwargs):
        super(MainWindow,self).__init__(*args,**kwargs)

        canvas = QtGui.QPixmap(1000, 1000)
        self.setupUi(self)
        self.pixelmap.setPixmap(canvas)
        self.pixelmap.setFixedHeight(1000)
        self.pixelmap.setFixedWidth(1000)
        self.pen = QtGui.QPen()




        self.initial_state, self.n = experiment1()
        self.btn_init_state.clicked.connect(self.draw_init_state)
        self.btn_fin_state.clicked.connect(self.draw_fin_state)

    def draw_something(self):
        from random import randint
        painter = QtGui.QPainter(self.label.pixmap())
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
        for i in self.initial_state.dict:
            self.pen.setWidth(5)
            self.pen.setColor(QtGui.QColor('green'))
            painter.setPen(self.pen)
            x1 = self.initial_state.centrals[i].CoordX * 10
            y1 = self.initial_state.centrals[i].CoordY * 10

            painter.drawPoint(x1,y1)

            for cl in self.initial_state.dict[i]:
                self.pen.setWidth(10)
                self.pen.setColor(QtGui.QColor('red'))
                painter.setPen(self.pen)
                x2 = self.initial_state.clients[cl].CoordX * 10
                y2 = self.initial_state.clients[cl].CoordY * 10
                painter.drawPoint(x2, y2)
                painter.drawLine(x1,x2,y1,y2)

        painter.end()

    def draw_fin_state(self):
        painter = QtGui.QPainter(self.pixelmap.pixmap())
        for i in self.n.dict:
            self.pen.setWidth(5)
            self.pen.setColor(QtGui.QColor('green'))
            painter.setPen(self.pen)
            x1 = self.n.centrals[i].CoordX * 10
            y1 = self.n.centrals[i].CoordY * 10

            painter.drawPoint(x1, y1)

            for cl in self.n.dict[i]:
                self.pen.setWidth(10)
                self.pen.setColor(QtGui.QColor('red'))
                painter.setPen(self.pen)
                x2 = self.n.clients[cl].CoordX * 10
                y2 = self.n.clients[cl].CoordY * 10
                painter.drawPoint(x2, y2)
                painter.drawLine(x1, x2, y1, y2)
        painter.end()


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()