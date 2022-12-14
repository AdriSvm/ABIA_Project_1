# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'graphics.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2214, 1116)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_init_state = QtWidgets.QPushButton(self.centralwidget)
        self.btn_init_state.setGeometry(QtCore.QRect(10, 640, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_init_state.setFont(font)
        self.btn_init_state.setObjectName("btn_init_state")
        self.btn_fin_state = QtWidgets.QPushButton(self.centralwidget)
        self.btn_fin_state.setGeometry(QtCore.QRect(210, 640, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_fin_state.setFont(font)
        self.btn_fin_state.setObjectName("btn_fin_state")
        self.pixelmap = QtWidgets.QLabel(self.centralwidget)
        self.pixelmap.setGeometry(QtCore.QRect(390, 30, 1461, 361))
        self.pixelmap.setText("")
        self.pixelmap.setObjectName("pixelmap")
        self.btn_execute = QtWidgets.QPushButton(self.centralwidget)
        self.btn_execute.setGeometry(QtCore.QRect(210, 530, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_execute.setFont(font)
        self.btn_execute.setObjectName("btn_execute")
        self.lbl_executing = QtWidgets.QLabel(self.centralwidget)
        self.lbl_executing.setGeometry(QtCore.QRect(210, 590, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_executing.setFont(font)
        self.lbl_executing.setText("")
        self.lbl_executing.setObjectName("lbl_executing")
        self.lbl_timing = QtWidgets.QLabel(self.centralwidget)
        self.lbl_timing.setGeometry(QtCore.QRect(10, 590, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_timing.setFont(font)
        self.lbl_timing.setText("")
        self.lbl_timing.setObjectName("lbl_timing")
        self.btn_time = QtWidgets.QPushButton(self.centralwidget)
        self.btn_time.setGeometry(QtCore.QRect(10, 530, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_time.setFont(font)
        self.btn_time.setObjectName("btn_time")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 151, 201))
        self.groupBox.setObjectName("groupBox")
        self.txt_c_a = QtWidgets.QLineEdit(self.groupBox)
        self.txt_c_a.setGeometry(QtCore.QRect(80, 40, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txt_c_a.setFont(font)
        self.txt_c_a.setObjectName("txt_c_a")
        self.txt_c_b = QtWidgets.QLineEdit(self.groupBox)
        self.txt_c_b.setGeometry(QtCore.QRect(80, 70, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txt_c_b.setFont(font)
        self.txt_c_b.setObjectName("txt_c_b")
        self.txt_c_c = QtWidgets.QLineEdit(self.groupBox)
        self.txt_c_c.setGeometry(QtCore.QRect(80, 100, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txt_c_c.setFont(font)
        self.txt_c_c.setObjectName("txt_c_c")
        self.lbl_num_centrals = QtWidgets.QLabel(self.groupBox)
        self.lbl_num_centrals.setGeometry(QtCore.QRect(20, 30, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lbl_num_centrals.setFont(font)
        self.lbl_num_centrals.setObjectName("lbl_num_centrals")
        self.lbl_num_centrals_2 = QtWidgets.QLabel(self.groupBox)
        self.lbl_num_centrals_2.setGeometry(QtCore.QRect(20, 60, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lbl_num_centrals_2.setFont(font)
        self.lbl_num_centrals_2.setObjectName("lbl_num_centrals_2")
        self.lbl_num_centrals_3 = QtWidgets.QLabel(self.groupBox)
        self.lbl_num_centrals_3.setGeometry(QtCore.QRect(20, 90, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lbl_num_centrals_3.setFont(font)
        self.lbl_num_centrals_3.setObjectName("lbl_num_centrals_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(170, 10, 211, 201))
        self.groupBox_2.setObjectName("groupBox_2")
        self.lbl_num_centrals_4 = QtWidgets.QLabel(self.groupBox_2)
        self.lbl_num_centrals_4.setGeometry(QtCore.QRect(20, 30, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lbl_num_centrals_4.setFont(font)
        self.lbl_num_centrals_4.setObjectName("lbl_num_centrals_4")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 100, 181, 91))
        self.groupBox_3.setObjectName("groupBox_3")
        self.lbl_num_centrals_5 = QtWidgets.QLabel(self.groupBox_3)
        self.lbl_num_centrals_5.setGeometry(QtCore.QRect(20, 30, 31, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lbl_num_centrals_5.setFont(font)
        self.lbl_num_centrals_5.setObjectName("lbl_num_centrals_5")
        self.lbl_num_centrals_6 = QtWidgets.QLabel(self.groupBox_3)
        self.lbl_num_centrals_6.setGeometry(QtCore.QRect(80, 30, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lbl_num_centrals_6.setFont(font)
        self.lbl_num_centrals_6.setObjectName("lbl_num_centrals_6")
        self.lbl_num_centrals_7 = QtWidgets.QLabel(self.groupBox_3)
        self.lbl_num_centrals_7.setGeometry(QtCore.QRect(140, 30, 31, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lbl_num_centrals_7.setFont(font)
        self.lbl_num_centrals_7.setObjectName("lbl_num_centrals_7")
        self.txt_clients_XG = QtWidgets.QLineEdit(self.groupBox_3)
        self.txt_clients_XG.setGeometry(QtCore.QRect(10, 60, 41, 20))
        self.txt_clients_XG.setObjectName("txt_clients_XG")
        self.txt_clients_MG = QtWidgets.QLineEdit(self.groupBox_3)
        self.txt_clients_MG.setGeometry(QtCore.QRect(70, 60, 41, 20))
        self.txt_clients_MG.setObjectName("txt_clients_MG")
        self.txt_clients_G = QtWidgets.QLineEdit(self.groupBox_3)
        self.txt_clients_G.setGeometry(QtCore.QRect(130, 60, 41, 20))
        self.txt_clients_G.setObjectName("txt_clients_G")
        self.txt_num_clients = QtWidgets.QLineEdit(self.groupBox_2)
        self.txt_num_clients.setGeometry(QtCore.QRect(150, 40, 51, 21))
        self.txt_num_clients.setObjectName("txt_num_clients")
        self.txt_propg = QtWidgets.QLineEdit(self.groupBox_2)
        self.txt_propg.setGeometry(QtCore.QRect(150, 70, 51, 20))
        self.txt_propg.setObjectName("txt_propg")
        self.lbl_num_centrals_8 = QtWidgets.QLabel(self.groupBox_2)
        self.lbl_num_centrals_8.setGeometry(QtCore.QRect(20, 60, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lbl_num_centrals_8.setFont(font)
        self.lbl_num_centrals_8.setObjectName("lbl_num_centrals_8")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 230, 371, 201))
        self.groupBox_4.setObjectName("groupBox_4")
        self.lbl_seed = QtWidgets.QLabel(self.groupBox_4)
        self.lbl_seed.setGeometry(QtCore.QRect(10, 20, 291, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_seed.setFont(font)
        self.lbl_seed.setObjectName("lbl_seed")
        self.txt_seed = QtWidgets.QLineEdit(self.groupBox_4)
        self.txt_seed.setGeometry(QtCore.QRect(290, 40, 71, 21))
        self.txt_seed.setObjectName("txt_seed")
        self.box_method = QtWidgets.QComboBox(self.groupBox_4)
        self.box_method.setGeometry(QtCore.QRect(10, 160, 351, 22))
        self.box_method.setObjectName("box_method")
        self.box_method.addItem("")
        self.box_method.addItem("")
        self.box_method.addItem("")
        self.lbl_method = QtWidgets.QLabel(self.groupBox_4)
        self.lbl_method.setGeometry(QtCore.QRect(10, 130, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_method.setFont(font)
        self.lbl_method.setObjectName("lbl_method")
        self.lbl_gen = QtWidgets.QLabel(self.groupBox_4)
        self.lbl_gen.setGeometry(QtCore.QRect(10, 70, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_gen.setFont(font)
        self.lbl_gen.setObjectName("lbl_gen")
        self.box_gen = QtWidgets.QComboBox(self.groupBox_4)
        self.box_gen.setGeometry(QtCore.QRect(10, 110, 351, 22))
        self.box_gen.setObjectName("box_gen")
        self.box_gen.addItem("")
        self.box_gen.addItem("")
        self.box_gen.addItem("")
        self.txt_bens = QtWidgets.QLabel(self.centralwidget)
        self.txt_bens.setGeometry(QtCore.QRect(390, 540, 331, 91))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.txt_bens.setFont(font)
        self.txt_bens.setText("")
        self.txt_bens.setObjectName("txt_bens")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 430, 371, 91))
        self.groupBox_5.setObjectName("groupBox_5")
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setGeometry(QtCore.QRect(10, 20, 61, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label_2.setGeometry(QtCore.QRect(10, 30, 71, 41))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setGeometry(QtCore.QRect(10, 50, 71, 41))
        self.label_3.setObjectName("label_3")
        self.txt_k = QtWidgets.QLineEdit(self.groupBox_5)
        self.txt_k.setGeometry(QtCore.QRect(70, 20, 71, 21))
        self.txt_k.setObjectName("txt_k")
        self.txt_lambda = QtWidgets.QLineEdit(self.groupBox_5)
        self.txt_lambda.setGeometry(QtCore.QRect(70, 40, 71, 21))
        self.txt_lambda.setObjectName("txt_lambda")
        self.txt_limit = QtWidgets.QLineEdit(self.groupBox_5)
        self.txt_limit.setGeometry(QtCore.QRect(70, 60, 71, 21))
        self.txt_limit.setObjectName("txt_limit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 2214, 17))
        self.menubar.setObjectName("menubar")
        self.menuCentral_Distribution = QtWidgets.QMenu(self.menubar)
        self.menuCentral_Distribution.setObjectName("menuCentral_Distribution")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuCentral_Distribution.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_init_state.setText(_translate("MainWindow", "Show Initial State"))
        self.btn_fin_state.setText(_translate("MainWindow", "Show final State"))
        self.btn_execute.setText(_translate("MainWindow", "Executar algorisme"))
        self.btn_time.setText(_translate("MainWindow", "Temps d\'execuci??"))
        self.groupBox.setTitle(_translate("MainWindow", "Centrals"))
        self.txt_c_a.setText(_translate("MainWindow", "5"))
        self.txt_c_b.setText(_translate("MainWindow", "10"))
        self.txt_c_c.setText(_translate("MainWindow", "25"))
        self.lbl_num_centrals.setText(_translate("MainWindow", "Tipus A"))
        self.lbl_num_centrals_2.setText(_translate("MainWindow", "Tipus B"))
        self.lbl_num_centrals_3.setText(_translate("MainWindow", "Tipus C"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Clients"))
        self.lbl_num_centrals_4.setText(_translate("MainWindow", "Total"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Proporcions"))
        self.lbl_num_centrals_5.setText(_translate("MainWindow", "XG"))
        self.lbl_num_centrals_6.setText(_translate("MainWindow", "MG"))
        self.lbl_num_centrals_7.setText(_translate("MainWindow", "G"))
        self.txt_clients_XG.setText(_translate("MainWindow", "0.2"))
        self.txt_clients_MG.setText(_translate("MainWindow", "0.3"))
        self.txt_clients_G.setText(_translate("MainWindow", "0.5"))
        self.txt_num_clients.setText(_translate("MainWindow", "1000"))
        self.txt_propg.setText(_translate("MainWindow", "0.75"))
        self.lbl_num_centrals_8.setText(_translate("MainWindow", "Proporci?? garantitzats"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Par??metres"))
        self.lbl_seed.setText(_translate("MainWindow", "Llavor generadora dels clients i centrals"))
        self.txt_seed.setText(_translate("MainWindow", "22"))
        self.box_method.setItemText(0, _translate("MainWindow", "Seleccionar un m??tode"))
        self.box_method.setItemText(1, _translate("MainWindow", "Hill Climbing"))
        self.box_method.setItemText(2, _translate("MainWindow", "Simulated Annealing"))
        self.lbl_method.setText(_translate("MainWindow", "M??tode"))
        self.lbl_gen.setText(_translate("MainWindow", "Funci?? generadora"))
        self.box_gen.setItemText(0, _translate("MainWindow", "Seleccionar una funci?? generadora"))
        self.box_gen.setItemText(1, _translate("MainWindow", "Primer garantitzats i despr??s tots no garantitzats(ORDERED)"))
        self.box_gen.setItemText(2, _translate("MainWindow", "Nom??s garantitzats per ordre d\'arribada(ONLY GRANTED)"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Par??metres simulated annealing"))
        self.label.setText(_translate("MainWindow", "K"))
        self.label_2.setText(_translate("MainWindow", "Lambda"))
        self.label_3.setText(_translate("MainWindow", "L??mit"))
        self.txt_k.setText(_translate("MainWindow", "5"))
        self.txt_lambda.setText(_translate("MainWindow", "0.0005"))
        self.txt_limit.setText(_translate("MainWindow", "750"))
        self.menuCentral_Distribution.setTitle(_translate("MainWindow", "Central Distribution"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
