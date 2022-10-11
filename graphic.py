from main import *
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from window import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,*args,obj=None,**kwargs):
        super(MainWindow,self).__init__(*args,**kwargs)
        self.initial_state = None
        self.n = None

        self.setupUi(self)
        self.pixelmap.setFixedHeight(1050)
        self.pixelmap.setFixedWidth(1050)
        self.setBaseSize(1900,1050)
        self.btn_execute.clicked.connect(self.draw_executing)
        self.btn_execute.clicked.connect(self.execute_experiment)
        self.btn_execute.clicked.connect(self.draw_executing)
        self.btn_time.clicked.connect(self.execute_timming)
        self.btn_init_state.clicked.connect(self.draw_executing)
        self.btn_init_state.clicked.connect(self.draw_init_state)
        self.btn_init_state.clicked.connect(self.draw_executing)
        self.btn_fin_state.clicked.connect(self.draw_executing)
        self.btn_fin_state.clicked.connect(self.draw_fin_state)
        self.btn_fin_state.clicked.connect(self.draw_executing)



    def draw_executing(self):
        if self.lbl_executing.text() != "Executing...":
            self.lbl_executing.setText("Executing...")
        elif self.lbl_executing.text() == "Executing...":
            self.lbl_executing.setText("Executed")
        self.show()


    def execute_experiment(self):
        n = int(self.txt_num_clients.text())
        c_a= int(self.txt_c_a.text())
        c_b = int(self.txt_c_c.text())
        c_c = int(self.txt_c_c.text())
        xg = float(self.txt_clients_XG.text())
        mg = float(self.txt_clients_MG.text())
        g = float(self.txt_clients_G.text())
        propg = float(self.txt_propg.text())
        seed = int(self.txt_seed.text())
        gen = self.box_gen.currentText()
        if gen == "Només garantitzats per ordre d'arribada(ONLY GRANTED)":
            gen = "ONLY GRANTED"
        else:
            gen = "ORDERED"

        method = self.box_method.currentText()
        self.initial_state, self.n = experiment(method,gen,[c_a,c_b,c_c],n,[xg,mg,g],propg,seed,False)

    def execute_timming(self):
        n = self.txt_num_clients.text()
        c_a = self.txt_c_a.text()
        c_b = self.txt_c_c.text()
        c_c = self.txt_c_c.text()
        xg = self.txt_clients_XG.text()
        mg = self.txt_clients_MG.text()
        g = self.txt_clients_G.text()
        propg = self.txt_propg.text()
        seed = self.txt_seed.text()
        gen = self.box_gen.itemText()
        print(gen)
        if gen == "Només garantitzats per ordre d'arribada(ONLY GRANTED)":
            gen = "ONLY GRANTED"
        else:
            gen = "ORDERED"

        method = self.box_method.itemText()
        self.initial_state, self.n = experiment(method, gen, [c_a, c_b, c_c], n, [xg, mg, g], propg, seed, True)
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
        if self.initial_state != None:
            canvas = QtGui.QPixmap(1000, 1000)
            self.pixelmap.setPixmap(canvas)
            painter = QtGui.QPainter(self.pixelmap.pixmap())
            pen = QtGui.QPen()
            painter.setPen(pen)
            for i in self.initial_state.dict:
                pen.setWidth(10)
                pen.setColor(QtGui.QColor('green'))
                painter.setPen(pen)
                x1 = self.initial_state.centrals[i].CoordX * 10
                y1 = self.initial_state.centrals[i].CoordY * 10

                painter.drawPoint(x1,y1)

                for cl in self.initial_state.dict[i]:
                    pen.setWidth(5)
                    x2 = self.initial_state.clients[cl].CoordX * 10
                    y2 = self.initial_state.clients[cl].CoordY * 10

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
                x2 = self.initial_state.clients[i].CoordX * 10
                y2 = self.initial_state.clients[i].CoordY * 10
                painter.drawPoint(x2, y2)
            print('printed')
            painter.end()
            self.show()
        else:
            self.lbl_executing.setText("Primer executa el algorisme")


    def draw_fin_state(self):
        canvas = QtGui.QPixmap(1000, 1000)
        self.pixelmap.setPixmap(canvas)
        painter = QtGui.QPainter(self.pixelmap.pixmap())
        pen = QtGui.QPen()
        painter.setPen(pen)
        for i in self.n.dict:
            pen.setWidth(10)
            pen.setColor(QtGui.QColor('green'))
            painter.setPen(pen)
            x1 = self.n.centrals[i].CoordX * 10
            y1 = self.n.centrals[i].CoordY * 10

            painter.drawPoint(x1, y1)

            for cl in self.n.dict[i]:
                pen.setWidth(5)
                x2 = self.n.clients[cl].CoordX * 10
                y2 = self.n.clients[cl].CoordY * 10

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
            x2 = self.n.clients[i].CoordX * 10
            y2 = self.n.clients[i].CoordY * 10
            painter.drawPoint(x2, y2)
        print("printed")
        painter.end()
        self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()