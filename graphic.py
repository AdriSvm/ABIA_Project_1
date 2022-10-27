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
        self.setWindowTitle("Central Distribution")
        self.setWindowIcon(QtGui.QIcon("Icon/power-plant.ico"))

        self.setupUi(self)
        self.pixelmap.setFixedHeight(500)
        self.pixelmap.setFixedWidth(500)
        self.setFixedSize(900,800)

        self.btn_execute.clicked.connect(self.execute_experiment)


        self.btn_time.clicked.connect(self.execute_timming)

        self.btn_init_state.clicked.connect(self.draw_init_state)

        self.btn_fin_state.clicked.connect(self.draw_fin_state)

        self.lbl_executing.show()




    def execute_experiment(self):
        self.lbl_executing.setText("Executing...")
        n = int(self.txt_num_clients.text())
        c_a= int(self.txt_c_a.text())
        c_b = int(self.txt_c_b.text())
        c_c = int(self.txt_c_c.text())
        xg = float(self.txt_clients_XG.text())
        mg = float(self.txt_clients_MG.text())
        g = float(self.txt_clients_G.text())
        k = int(self.txt_k.text())
        lamda = float(self.txt_lambda.text())
        limit = int(self.txt_limit.text())
        propg = float(self.txt_propg.text())
        seed = int(self.txt_seed.text())
        gen = self.box_gen.currentText()
        if gen == "Només garantitzats per ordre d'arribada(ONLY GRANTED)":
            gen = "ONLY GRANTED"
        else:
            gen = "ORDERED"

        method = self.box_method.currentText()
        self.initial_state, self.n = experiment(method,gen,[c_a,c_b,c_c],n,[xg,mg,g],propg,seed,k=k,lam=lamda,limit=limit,timming=False)
        self.lbl_executing.setText("Executed")
        self.txt_bens.setText(f"Beneficis inicials: {self.initial_state.heuristic()}\n"
                              f"Beneficis finals: {self.n.heuristic()} \nClients no insertats: {len([_ for _ in self.n.left])} \n")
        self.lbl_timing.clear()

    def execute_timming(self):
        n = int(self.txt_num_clients.text())
        c_a = int(self.txt_c_a.text())
        c_b = int(self.txt_c_b.text())
        c_c = int(self.txt_c_c.text())
        xg = float(self.txt_clients_XG.text())
        mg = float(self.txt_clients_MG.text())
        g = float(self.txt_clients_G.text())
        k = int(self.txt_k.text())
        lamda = float(self.txt_lambda.text())
        limit = int(self.txt_limit.text())
        propg = float(self.txt_propg.text())
        seed = int(self.txt_seed.text())
        gen = self.box_gen.currentText()
        if gen == "Només garantitzats per ordre d'arribada(ONLY GRANTED)":
            gen = "ONLY GRANTED"
        else:
            gen = "ORDERED"

        method = self.box_method.currentText()
        time, nothing = experiment(method, gen, [c_a, c_b, c_c], n, [xg, mg, g], propg, seed=seed,k=k,lam=lamda,limit=limit,timming=True,n_iter=1)
        self.lbl_timing.setText(str(time)[:5] + "s")
        self.lbl_executing.clear()


    def draw_init_state(self):
        if self.initial_state != None:
            canvas = QtGui.QPixmap(500, 500)
            self.pixelmap.setPixmap(canvas)
            painter = QtGui.QPainter(self.pixelmap.pixmap())
            pen = QtGui.QPen()
            painter.setPen(pen)
            for i in self.initial_state.dict:
                pen.setWidth(7)
                pen.setColor(QtGui.QColor('green'))
                painter.setPen(pen)
                x1 = self.initial_state.centrals[i].CoordX * 5
                y1 = self.initial_state.centrals[i].CoordY * 5

                painter.drawPoint(x1,y1)

                for cl in self.initial_state.dict[i]:

                    x2 = self.initial_state.clients[cl].CoordX * 5
                    y2 = self.initial_state.clients[cl].CoordY * 5

                    if self.initial_state.clients[cl].Contrato == 0:
                        pen.setColor(QtGui.QColor('yellow'))
                        pen.setWidth(3)
                        painter.setPen(pen)
                        painter.drawPoint(x2, y2)
                    else:
                        pen.setColor(QtGui.QColor('red'))
                        pen.setWidth(3)
                        painter.setPen(pen)
                        painter.drawPoint(x2, y2)

                    pen.setWidth(1)
                    pen.setColor(QtGui.QColor('white'))
                    painter.setPen(pen)
                    painter.drawLine(x1,y1,x2,y2)

            for i in self.initial_state.left:
                pen.setWidth(3)
                pen.setColor(QtGui.QColor('red'))
                painter.setPen(pen)
                x2 = self.initial_state.clients[i].CoordX * 5
                y2 = self.initial_state.clients[i].CoordY * 5
                painter.drawPoint(x2, y2)

            painter.end()
            self.show()
        else:
            self.lbl_executing.setText("Primer executa el algorisme")
        self.lbl_executing.clear()


    def draw_fin_state(self):
        canvas = QtGui.QPixmap(500, 500)
        self.pixelmap.setPixmap(canvas)
        painter = QtGui.QPainter(self.pixelmap.pixmap())
        pen = QtGui.QPen()
        painter.setPen(pen)
        for i in self.n.dict:
            pen.setWidth(7)
            pen.setColor(QtGui.QColor('green'))
            painter.setPen(pen)
            x1 = self.n.centrals[i].CoordX * 5
            y1 = self.n.centrals[i].CoordY * 5

            painter.drawPoint(x1, y1)

            for cl in self.n.dict[i]:

                x2 = self.n.clients[cl].CoordX * 5
                y2 = self.n.clients[cl].CoordY * 5

                if self.n.clients[cl].Contrato == 0:
                    pen.setColor(QtGui.QColor('yellow'))
                    pen.setWidth(3)
                    painter.setPen(pen)
                    painter.drawPoint(x2, y2)
                else:
                    pen.setColor(QtGui.QColor('red'))
                    pen.setWidth(3)
                    painter.setPen(pen)
                    painter.drawPoint(x2, y2)

                pen.setWidth(1)
                pen.setColor(QtGui.QColor('white'))
                painter.setPen(pen)
                painter.drawLine(x1, y1, x2, y2)

        for i in self.n.left:
            pen.setWidth(3)
            pen.setColor(QtGui.QColor('red'))
            painter.setPen(pen)
            x2 = self.n.clients[i].CoordX * 5
            y2 = self.n.clients[i].CoordY * 5
            painter.drawPoint(x2, y2)
        painter.end()
        self.lbl_executing.clear()
        self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()