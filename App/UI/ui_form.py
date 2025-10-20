# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QTabWidget, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1291, 757)
        MainWindow.setMinimumSize(QSize(1291, 757))
        font = QFont()
        font.setFamilies([u"Verdana"])
        font.setPointSize(11)
        MainWindow.setFont(font)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(u"background-color: rgb(197, 228, 255);")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(10, 70, 1271, 681))
        self.tabWidget.setStyleSheet(u"font: 12pt \"Verdana\";\n"
"color: rgb(48, 48, 48);\n"
"background-color: rgba(215, 236, 255, 180);")
        self.tabWidget.setTabPosition(QTabWidget.TabPosition.North)
        self.tabWidget.setTabShape(QTabWidget.TabShape.Rounded)
        self.tabWidget.setElideMode(Qt.TextElideMode.ElideNone)
        self.tabProcessing = QWidget()
        self.tabProcessing.setObjectName(u"tabProcessing")
        self.frameImageDisplay = QFrame(self.tabProcessing)
        self.frameImageDisplay.setObjectName(u"frameImageDisplay")
        self.frameImageDisplay.setGeometry(QRect(360, 10, 901, 531))
        self.frameImageDisplay.setFrameShape(QFrame.Shape.Box)
        self.lblImageDisplay = QLabel(self.frameImageDisplay)
        self.lblImageDisplay.setObjectName(u"lblImageDisplay")
        self.lblImageDisplay.setGeometry(QRect(10, 30, 881, 451))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblImageDisplay.sizePolicy().hasHeightForWidth())
        self.lblImageDisplay.setSizePolicy(sizePolicy)
        self.lblImageDisplay.setScaledContents(True)
        self.widget_6 = QWidget(self.frameImageDisplay)
        self.widget_6.setObjectName(u"widget_6")
        self.widget_6.setGeometry(QRect(7, 480, 891, 45))
        self.horizontalLayout = QHBoxLayout(self.widget_6)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.btnSetCrop = QPushButton(self.widget_6)
        self.btnSetCrop.setObjectName(u"btnSetCrop")

        self.horizontalLayout.addWidget(self.btnSetCrop)

        self.btnNextImage = QPushButton(self.widget_6)
        self.btnNextImage.setObjectName(u"btnNextImage")

        self.horizontalLayout.addWidget(self.btnNextImage)

        self.btnBatchCrop = QPushButton(self.widget_6)
        self.btnBatchCrop.setObjectName(u"btnBatchCrop")

        self.horizontalLayout.addWidget(self.btnBatchCrop)

        self.progressBar_2 = QProgressBar(self.widget_6)
        self.progressBar_2.setObjectName(u"progressBar_2")
        self.progressBar_2.setValue(0)

        self.horizontalLayout.addWidget(self.progressBar_2)

        self.label_18 = QLabel(self.frameImageDisplay)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(10, 10, 151, 21))
        self.label_18.setStyleSheet(u"font: 700 12pt \"Verdana\";")
        self.lblFolderNumber = QLabel(self.frameImageDisplay)
        self.lblFolderNumber.setObjectName(u"lblFolderNumber")
        self.lblFolderNumber.setGeometry(QRect(830, 10, 61, 21))
        self.label_5 = QLabel(self.frameImageDisplay)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(760, 10, 61, 20))
        self.frameRunBatch = QFrame(self.tabProcessing)
        self.frameRunBatch.setObjectName(u"frameRunBatch")
        self.frameRunBatch.setGeometry(QRect(10, 547, 1251, 91))
        self.frameRunBatch.setFrameShape(QFrame.Shape.Box)
        self.verticalLayout = QVBoxLayout(self.frameRunBatch)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_10 = QLabel(self.frameRunBatch)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setStyleSheet(u"font: 700 12pt \"Verdana\";")

        self.verticalLayout.addWidget(self.label_10)

        self.widget_7 = QWidget(self.frameRunBatch)
        self.widget_7.setObjectName(u"widget_7")
        self.horizontalLayout_2 = QHBoxLayout(self.widget_7)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btnRunBatch = QPushButton(self.widget_7)
        self.btnRunBatch.setObjectName(u"btnRunBatch")

        self.horizontalLayout_2.addWidget(self.btnRunBatch)

        self.progressBar = QProgressBar(self.widget_7)
        self.progressBar.setObjectName(u"progressBar")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy1)
        self.progressBar.setMinimumSize(QSize(0, 25))
        self.progressBar.setValue(0)

        self.horizontalLayout_2.addWidget(self.progressBar)


        self.verticalLayout.addWidget(self.widget_7)

        self.frameDataInitialisation = QFrame(self.tabProcessing)
        self.frameDataInitialisation.setObjectName(u"frameDataInitialisation")
        self.frameDataInitialisation.setGeometry(QRect(10, 10, 341, 321))
        sizePolicy.setHeightForWidth(self.frameDataInitialisation.sizePolicy().hasHeightForWidth())
        self.frameDataInitialisation.setSizePolicy(sizePolicy)
        self.frameDataInitialisation.setStyleSheet(u"")
        self.frameDataInitialisation.setFrameShape(QFrame.Shape.Box)
        self.frameDataInitialisation.setFrameShadow(QFrame.Shadow.Plain)
        self.frameDataInitialisation.setMidLineWidth(0)
        self.verticalLayout_2 = QVBoxLayout(self.frameDataInitialisation)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_14 = QLabel(self.frameDataInitialisation)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setStyleSheet(u"font: 700 12pt \"Verdana\";")

        self.verticalLayout_2.addWidget(self.label_14)

        self.widget_9 = QWidget(self.frameDataInitialisation)
        self.widget_9.setObjectName(u"widget_9")
        self.verticalLayout_6 = QVBoxLayout(self.widget_9)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.label_16 = QLabel(self.widget_9)
        self.label_16.setObjectName(u"label_16")

        self.verticalLayout_6.addWidget(self.label_16)

        self.txtScannerName = QLineEdit(self.widget_9)
        self.txtScannerName.setObjectName(u"txtScannerName")

        self.verticalLayout_6.addWidget(self.txtScannerName)


        self.verticalLayout_2.addWidget(self.widget_9)

        self.widget = QWidget(self.frameDataInitialisation)
        self.widget.setObjectName(u"widget")
        self.verticalLayout_4 = QVBoxLayout(self.widget)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.btnSelectFolder = QPushButton(self.widget)
        self.btnSelectFolder.setObjectName(u"btnSelectFolder")
        self.btnSelectFolder.setAutoFillBackground(False)
        self.btnSelectFolder.setAutoDefault(False)
        self.btnSelectFolder.setFlat(False)

        self.verticalLayout_4.addWidget(self.btnSelectFolder)

        self.lblFolderPath = QLabel(self.widget)
        self.lblFolderPath.setObjectName(u"lblFolderPath")
        self.lblFolderPath.setStyleSheet(u"font: italic 8pt \"Verdana\";")
        self.lblFolderPath.setWordWrap(True)

        self.verticalLayout_4.addWidget(self.lblFolderPath)

        self.progressBarVideo = QProgressBar(self.widget)
        self.progressBarVideo.setObjectName(u"progressBarVideo")
        self.progressBarVideo.setValue(0)

        self.verticalLayout_4.addWidget(self.progressBarVideo)


        self.verticalLayout_2.addWidget(self.widget)

        self.widget_5 = QWidget(self.frameDataInitialisation)
        self.widget_5.setObjectName(u"widget_5")
        self.verticalLayout_5 = QVBoxLayout(self.widget_5)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.label_9 = QLabel(self.widget_5)
        self.label_9.setObjectName(u"label_9")

        self.verticalLayout_5.addWidget(self.label_9)

        self.checkOther = QCheckBox(self.widget_5)
        self.checkOther.setObjectName(u"checkOther")
        self.checkOther.setTristate(False)

        self.verticalLayout_5.addWidget(self.checkOther)

        self.checkVertical = QCheckBox(self.widget_5)
        self.checkVertical.setObjectName(u"checkVertical")

        self.verticalLayout_5.addWidget(self.checkVertical)


        self.verticalLayout_2.addWidget(self.widget_5)

        self.frameConversion = QFrame(self.tabProcessing)
        self.frameConversion.setObjectName(u"frameConversion")
        self.frameConversion.setEnabled(False)
        self.frameConversion.setGeometry(QRect(10, 420, 341, 121))
        self.frameConversion.setFrameShape(QFrame.Shape.Box)
        self.label_17 = QLabel(self.frameConversion)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(10, 10, 315, 18))
        font1 = QFont()
        font1.setFamilies([u"Verdana"])
        font1.setPointSize(11)
        font1.setBold(True)
        font1.setItalic(False)
        self.label_17.setFont(font1)
        self.label_17.setStyleSheet(u"font: 700 11pt \"Verdana\";")
        self.widget_2 = QWidget(self.frameConversion)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setGeometry(QRect(10, 34, 321, 44))
        self.horizontalLayout_5 = QHBoxLayout(self.widget_2)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(self.widget_2)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.txtDistanceMm = QLineEdit(self.widget_2)
        self.txtDistanceMm.setObjectName(u"txtDistanceMm")

        self.horizontalLayout_5.addWidget(self.txtDistanceMm)

        self.btnSaveCalibration = QPushButton(self.frameConversion)
        self.btnSaveCalibration.setObjectName(u"btnSaveCalibration")
        self.btnSaveCalibration.setGeometry(QRect(10, 80, 311, 27))
        self.label_2 = QLabel(self.tabProcessing)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 350, 321, 61))
        self.label_2.setStyleSheet(u"font: italic 9pt \"Verdana\";")
        self.label_2.setWordWrap(True)
        self.tabWidget.addTab(self.tabProcessing, "")
        self.tabInput = QWidget()
        self.tabInput.setObjectName(u"tabInput")
        self.frame = QFrame(self.tabInput)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(0, 20, 1271, 601))
        self.frame.setStyleSheet(u"font: 11pt \"Verdana\";")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.frame.setLineWidth(0)
        self.gridLayout_5 = QGridLayout(self.frame)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.widget_12 = QWidget(self.frame)
        self.widget_12.setObjectName(u"widget_12")
        self.verticalLayout_9 = QVBoxLayout(self.widget_12)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.label_58 = QLabel(self.widget_12)
        self.label_58.setObjectName(u"label_58")
        self.label_58.setStyleSheet(u"font: 700 11pt \"Verdana\";")

        self.verticalLayout_9.addWidget(self.label_58)

        self.btnGenerate = QPushButton(self.widget_12)
        self.btnGenerate.setObjectName(u"btnGenerate")

        self.verticalLayout_9.addWidget(self.btnGenerate)

        self.lblStatusInputSettings = QLabel(self.widget_12)
        self.lblStatusInputSettings.setObjectName(u"lblStatusInputSettings")
        self.lblStatusInputSettings.setStyleSheet(u"font: italic 10pt \"Verdana\";")
        self.lblStatusInputSettings.setWordWrap(True)

        self.verticalLayout_9.addWidget(self.lblStatusInputSettings)


        self.gridLayout_5.addWidget(self.widget_12, 2, 2, 1, 1)

        self.widget_11 = QWidget(self.frame)
        self.widget_11.setObjectName(u"widget_11")
        self.gridLayout_4 = QGridLayout(self.widget_11)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_42 = QLabel(self.widget_11)
        self.label_42.setObjectName(u"label_42")

        self.gridLayout_4.addWidget(self.label_42, 8, 0, 1, 1)

        self.txtFreqRange = QLineEdit(self.widget_11)
        self.txtFreqRange.setObjectName(u"txtFreqRange")

        self.gridLayout_4.addWidget(self.txtFreqRange, 15, 1, 1, 1)

        self.txtFund = QLineEdit(self.widget_11)
        self.txtFund.setObjectName(u"txtFund")

        self.gridLayout_4.addWidget(self.txtFund, 17, 1, 1, 1)

        self.txtFocus = QLineEdit(self.widget_11)
        self.txtFocus.setObjectName(u"txtFocus")

        self.gridLayout_4.addWidget(self.txtFocus, 14, 1, 1, 1)

        self.label_48 = QLabel(self.widget_11)
        self.label_48.setObjectName(u"label_48")

        self.gridLayout_4.addWidget(self.label_48, 9, 0, 1, 1)

        self.txtFPS = QLineEdit(self.widget_11)
        self.txtFPS.setObjectName(u"txtFPS")

        self.gridLayout_4.addWidget(self.txtFPS, 8, 1, 1, 1)

        self.label_46 = QLabel(self.widget_11)
        self.label_46.setObjectName(u"label_46")

        self.gridLayout_4.addWidget(self.label_46, 7, 0, 1, 1)

        self.label_47 = QLabel(self.widget_11)
        self.label_47.setObjectName(u"label_47")

        self.gridLayout_4.addWidget(self.label_47, 2, 0, 1, 1)

        self.label_39 = QLabel(self.widget_11)
        self.label_39.setObjectName(u"label_39")
        self.label_39.setWordWrap(True)

        self.gridLayout_4.addWidget(self.label_39, 12, 0, 1, 1)

        self.txtPower = QLineEdit(self.widget_11)
        self.txtPower.setObjectName(u"txtPower")

        self.gridLayout_4.addWidget(self.txtPower, 3, 1, 1, 1)

        self.txtOthersettings = QLineEdit(self.widget_11)
        self.txtOthersettings.setObjectName(u"txtOthersettings")

        self.gridLayout_4.addWidget(self.txtOthersettings, 12, 1, 1, 1)

        self.txtPersistence = QLineEdit(self.widget_11)
        self.txtPersistence.setObjectName(u"txtPersistence")

        self.gridLayout_4.addWidget(self.txtPersistence, 11, 1, 1, 1)

        self.txtGain = QLineEdit(self.widget_11)
        self.txtGain.setObjectName(u"txtGain")

        self.gridLayout_4.addWidget(self.txtGain, 6, 1, 1, 1)

        self.txtApp = QLineEdit(self.widget_11)
        self.txtApp.setObjectName(u"txtApp")

        self.gridLayout_4.addWidget(self.txtApp, 2, 1, 1, 1)

        self.label_52 = QLabel(self.widget_11)
        self.label_52.setObjectName(u"label_52")

        self.gridLayout_4.addWidget(self.label_52, 16, 0, 1, 1)

        self.txtMITI = QLineEdit(self.widget_11)
        self.txtMITI.setObjectName(u"txtMITI")

        self.gridLayout_4.addWidget(self.txtMITI, 5, 1, 1, 1)

        self.txtCompounding = QLineEdit(self.widget_11)
        self.txtCompounding.setObjectName(u"txtCompounding")

        self.gridLayout_4.addWidget(self.txtCompounding, 9, 1, 1, 1)

        self.label_49 = QLabel(self.widget_11)
        self.label_49.setObjectName(u"label_49")
        self.label_49.setWordWrap(True)

        self.gridLayout_4.addWidget(self.label_49, 13, 0, 1, 1)

        self.label_43 = QLabel(self.widget_11)
        self.label_43.setObjectName(u"label_43")

        self.gridLayout_4.addWidget(self.label_43, 4, 0, 1, 1)

        self.txtDynRange = QLineEdit(self.widget_11)
        self.txtDynRange.setObjectName(u"txtDynRange")

        self.gridLayout_4.addWidget(self.txtDynRange, 7, 1, 1, 1)

        self.txtSmooth = QLineEdit(self.widget_11)
        self.txtSmooth.setObjectName(u"txtSmooth")

        self.gridLayout_4.addWidget(self.txtSmooth, 10, 1, 1, 1)

        self.label_50 = QLabel(self.widget_11)
        self.label_50.setObjectName(u"label_50")
        self.label_50.setWordWrap(True)

        self.gridLayout_4.addWidget(self.label_50, 14, 0, 1, 1)

        self.txtHarm = QLineEdit(self.widget_11)
        self.txtHarm.setObjectName(u"txtHarm")

        self.gridLayout_4.addWidget(self.txtHarm, 16, 1, 1, 1)

        self.txtFrequency = QLineEdit(self.widget_11)
        self.txtFrequency.setObjectName(u"txtFrequency")

        self.gridLayout_4.addWidget(self.txtFrequency, 4, 1, 1, 1)

        self.label_57 = QLabel(self.widget_11)
        self.label_57.setObjectName(u"label_57")
        self.label_57.setStyleSheet(u"font: 700 11pt \"Verdana\";\n"
"")

        self.gridLayout_4.addWidget(self.label_57, 0, 0, 1, 1)

        self.label_45 = QLabel(self.widget_11)
        self.label_45.setObjectName(u"label_45")

        self.gridLayout_4.addWidget(self.label_45, 6, 0, 1, 1)

        self.label_44 = QLabel(self.widget_11)
        self.label_44.setObjectName(u"label_44")

        self.gridLayout_4.addWidget(self.label_44, 17, 0, 1, 1)

        self.label_41 = QLabel(self.widget_11)
        self.label_41.setObjectName(u"label_41")
        self.label_41.setWordWrap(True)

        self.gridLayout_4.addWidget(self.label_41, 10, 0, 1, 1)

        self.label_37 = QLabel(self.widget_11)
        self.label_37.setObjectName(u"label_37")

        self.gridLayout_4.addWidget(self.label_37, 5, 0, 1, 1)

        self.label_38 = QLabel(self.widget_11)
        self.label_38.setObjectName(u"label_38")

        self.gridLayout_4.addWidget(self.label_38, 3, 0, 1, 1)

        self.label_40 = QLabel(self.widget_11)
        self.label_40.setObjectName(u"label_40")

        self.gridLayout_4.addWidget(self.label_40, 11, 0, 1, 1)

        self.label_51 = QLabel(self.widget_11)
        self.label_51.setObjectName(u"label_51")

        self.gridLayout_4.addWidget(self.label_51, 15, 0, 1, 1)

        self.txtDepth = QLineEdit(self.widget_11)
        self.txtDepth.setObjectName(u"txtDepth")

        self.gridLayout_4.addWidget(self.txtDepth, 13, 1, 1, 1)


        self.gridLayout_5.addWidget(self.widget_11, 0, 1, 3, 1)

        self.widget_4 = QWidget(self.frame)
        self.widget_4.setObjectName(u"widget_4")
        self.gridLayout_2 = QGridLayout(self.widget_4)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.txtPhantomMk = QLineEdit(self.widget_4)
        self.txtPhantomMk.setObjectName(u"txtPhantomMk")

        self.gridLayout_2.addWidget(self.txtPhantomMk, 8, 1, 1, 1)

        self.txtReason = QLineEdit(self.widget_4)
        self.txtReason.setObjectName(u"txtReason")

        self.gridLayout_2.addWidget(self.txtReason, 4, 1, 1, 1)

        self.label_24 = QLabel(self.widget_4)
        self.label_24.setObjectName(u"label_24")

        self.gridLayout_2.addWidget(self.label_24, 9, 0, 1, 1)

        self.label_25 = QLabel(self.widget_4)
        self.label_25.setObjectName(u"label_25")

        self.gridLayout_2.addWidget(self.label_25, 6, 0, 1, 1)

        self.label_23 = QLabel(self.widget_4)
        self.label_23.setObjectName(u"label_23")

        self.gridLayout_2.addWidget(self.label_23, 4, 0, 1, 1)

        self.label_27 = QLabel(self.widget_4)
        self.label_27.setObjectName(u"label_27")

        self.gridLayout_2.addWidget(self.label_27, 8, 0, 1, 1)

        self.txtLocation = QLineEdit(self.widget_4)
        self.txtLocation.setObjectName(u"txtLocation")

        self.gridLayout_2.addWidget(self.txtLocation, 7, 1, 1, 1)

        self.txtDetails = QLineEdit(self.widget_4)
        self.txtDetails.setObjectName(u"txtDetails")

        self.gridLayout_2.addWidget(self.txtDetails, 6, 1, 1, 1)

        self.txtDate = QLineEdit(self.widget_4)
        self.txtDate.setObjectName(u"txtDate")

        self.gridLayout_2.addWidget(self.txtDate, 9, 1, 1, 1)

        self.label_26 = QLabel(self.widget_4)
        self.label_26.setObjectName(u"label_26")

        self.gridLayout_2.addWidget(self.label_26, 7, 0, 1, 1)

        self.label_55 = QLabel(self.widget_4)
        self.label_55.setObjectName(u"label_55")
        self.label_55.setStyleSheet(u"font: 700 11pt \"Verdana\";")

        self.gridLayout_2.addWidget(self.label_55, 0, 0, 1, 1)


        self.gridLayout_5.addWidget(self.widget_4, 0, 0, 1, 1)

        self.widget_10 = QWidget(self.frame)
        self.widget_10.setObjectName(u"widget_10")
        self.verticalLayout_8 = QVBoxLayout(self.widget_10)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.label_53 = QLabel(self.widget_10)
        self.label_53.setObjectName(u"label_53")
        self.label_53.setStyleSheet(u"font: 700 11pt \"Verdana\";")

        self.verticalLayout_8.addWidget(self.label_53)

        self.label_54 = QLabel(self.widget_10)
        self.label_54.setObjectName(u"label_54")
        self.label_54.setStyleSheet(u"font: italic 10pt \"Verdana\";")
        self.label_54.setWordWrap(True)

        self.verticalLayout_8.addWidget(self.label_54)

        self.txtComments = QPlainTextEdit(self.widget_10)
        self.txtComments.setObjectName(u"txtComments")

        self.verticalLayout_8.addWidget(self.txtComments)


        self.gridLayout_5.addWidget(self.widget_10, 0, 2, 2, 1)

        self.widget_8 = QWidget(self.frame)
        self.widget_8.setObjectName(u"widget_8")
        self.gridLayout_3 = QGridLayout(self.widget_8)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_29 = QLabel(self.widget_8)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setWordWrap(True)

        self.gridLayout_3.addWidget(self.label_29, 10, 0, 1, 1)

        self.txtSoftwarelvl = QLineEdit(self.widget_8)
        self.txtSoftwarelvl.setObjectName(u"txtSoftwarelvl")

        self.gridLayout_3.addWidget(self.txtSoftwarelvl, 7, 1, 1, 1)

        self.label_34 = QLabel(self.widget_8)
        self.label_34.setObjectName(u"label_34")
        self.label_34.setWordWrap(True)

        self.gridLayout_3.addWidget(self.label_34, 5, 0, 1, 1)

        self.label_31 = QLabel(self.widget_8)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setWordWrap(True)

        self.gridLayout_3.addWidget(self.label_31, 9, 0, 1, 1)

        self.txtProbeYear = QLineEdit(self.widget_8)
        self.txtProbeYear.setObjectName(u"txtProbeYear")

        self.gridLayout_3.addWidget(self.txtProbeYear, 6, 1, 1, 1)

        self.label_56 = QLabel(self.widget_8)
        self.label_56.setObjectName(u"label_56")
        self.label_56.setStyleSheet(u"font: 700 11pt \"Verdana\";")

        self.gridLayout_3.addWidget(self.label_56, 2, 0, 1, 1)

        self.txtBootUp = QLineEdit(self.widget_8)
        self.txtBootUp.setObjectName(u"txtBootUp")

        self.gridLayout_3.addWidget(self.txtBootUp, 8, 1, 1, 1)

        self.txtProbe = QLineEdit(self.widget_8)
        self.txtProbe.setObjectName(u"txtProbe")

        self.gridLayout_3.addWidget(self.txtProbe, 5, 1, 1, 1)

        self.label_35 = QLabel(self.widget_8)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setWordWrap(True)

        self.gridLayout_3.addWidget(self.label_35, 6, 0, 1, 1)

        self.label_28 = QLabel(self.widget_8)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setWordWrap(True)

        self.gridLayout_3.addWidget(self.label_28, 8, 0, 1, 1)

        self.label_32 = QLabel(self.widget_8)
        self.label_32.setObjectName(u"label_32")
        self.label_32.setWordWrap(True)

        self.gridLayout_3.addWidget(self.label_32, 4, 0, 1, 1)

        self.label_33 = QLabel(self.widget_8)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setWordWrap(True)
        self.label_33.setOpenExternalLinks(False)

        self.gridLayout_3.addWidget(self.label_33, 7, 0, 1, 1)

        self.label_30 = QLabel(self.widget_8)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setWordWrap(True)

        self.gridLayout_3.addWidget(self.label_30, 11, 0, 1, 1)

        self.label_36 = QLabel(self.widget_8)
        self.label_36.setObjectName(u"label_36")
        self.label_36.setWordWrap(True)

        self.gridLayout_3.addWidget(self.label_36, 3, 0, 1, 1)

        self.txtScannerType = QLineEdit(self.widget_8)
        self.txtScannerType.setObjectName(u"txtScannerType")

        self.gridLayout_3.addWidget(self.txtScannerType, 3, 1, 1, 1)

        self.txtScannerYear = QLineEdit(self.widget_8)
        self.txtScannerYear.setObjectName(u"txtScannerYear")

        self.gridLayout_3.addWidget(self.txtScannerYear, 4, 1, 1, 1)

        self.btnChecks = QCheckBox(self.widget_8)
        self.btnChecks.setObjectName(u"btnChecks")

        self.gridLayout_3.addWidget(self.btnChecks, 9, 1, 1, 1, Qt.AlignmentFlag.AlignHCenter)

        self.btnAmbientLight = QCheckBox(self.widget_8)
        self.btnAmbientLight.setObjectName(u"btnAmbientLight")

        self.gridLayout_3.addWidget(self.btnAmbientLight, 10, 1, 1, 1, Qt.AlignmentFlag.AlignHCenter)

        self.btnDisplayLight = QCheckBox(self.widget_8)
        self.btnDisplayLight.setObjectName(u"btnDisplayLight")

        self.gridLayout_3.addWidget(self.btnDisplayLight, 11, 1, 1, 1, Qt.AlignmentFlag.AlignHCenter)


        self.gridLayout_5.addWidget(self.widget_8, 1, 0, 2, 1)

        self.tabWidget.addTab(self.tabInput, "")
        self.tabResults = QWidget()
        self.tabResults.setObjectName(u"tabResults")
        self.lblResults = QLabel(self.tabResults)
        self.lblResults.setObjectName(u"lblResults")
        self.lblResults.setGeometry(QRect(20, 20, 831, 601))
        self.groupBox = QGroupBox(self.tabResults)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(950, 270, 221, 221))
        self.groupBox.setStyleSheet(u"font: 700 14pt \"Verdana\";")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_11 = QLabel(self.groupBox)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setStyleSheet(u"font: 700 12pt \"Verdana\";")

        self.gridLayout.addWidget(self.label_11, 0, 0, 1, 1)

        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setStyleSheet(u"font: 700 12pt \"Verdana\";")

        self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)

        self.label_12 = QLabel(self.groupBox)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setStyleSheet(u"font: 700 12pt \"Verdana\";")

        self.gridLayout.addWidget(self.label_12, 2, 0, 1, 1)

        self.lblR = QLabel(self.groupBox)
        self.lblR.setObjectName(u"lblR")

        self.gridLayout.addWidget(self.lblR, 0, 1, 1, 1)

        self.lblLR = QLabel(self.groupBox)
        self.lblLR.setObjectName(u"lblLR")

        self.gridLayout.addWidget(self.lblLR, 2, 1, 1, 1)

        self.lblDR = QLabel(self.groupBox)
        self.lblDR.setObjectName(u"lblDR")

        self.gridLayout.addWidget(self.lblDR, 3, 1, 1, 1)

        self.lblScannerNameResults = QLabel(self.tabResults)
        self.lblScannerNameResults.setObjectName(u"lblScannerNameResults")
        self.lblScannerNameResults.setGeometry(QRect(948, 170, 221, 20))
        self.lblScannerNameResults.setStyleSheet(u"font: 700 16pt \"Verdana\";")
        self.lblScannerNameResults.setWordWrap(True)
        self.tabWidget.addTab(self.tabResults, "")
        self.tabAbout = QWidget()
        self.tabAbout.setObjectName(u"tabAbout")
        self.verticalLayout_7 = QVBoxLayout(self.tabAbout)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.widget_3 = QWidget(self.tabAbout)
        self.widget_3.setObjectName(u"widget_3")
        self.label_4 = QLabel(self.widget_3)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(9, 9, 105, 19))
        self.label_4.setStyleSheet(u"font: 700 12pt \"Verdana\";\n"
"color: rgb(24, 48, 72);")
        self.label_6 = QLabel(self.widget_3)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(10, 210, 701, 41))
        self.label_6.setTextFormat(Qt.TextFormat.AutoText)
        self.label_6.setWordWrap(True)
        self.label_8 = QLabel(self.widget_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(10, 550, 52, 19))
        self.label_8.setStyleSheet(u"font: 700 12pt \"Verdana\";\n"
"color: rgb(24, 48, 72);")
        self.label_15 = QLabel(self.widget_3)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(10, 290, 691, 61))
        self.label_15.setWordWrap(True)
        self.label_19 = QLabel(self.widget_3)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(10, 380, 1231, 38))
        self.label_19.setWordWrap(True)
        self.label_20 = QLabel(self.widget_3)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(10, 460, 1231, 38))
        self.label_20.setWordWrap(True)
        self.label_21 = QLabel(self.widget_3)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(10, 50, 452, 19))
        self.label_22 = QLabel(self.widget_3)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(10, 80, 179, 95))
        self.label_22.setStyleSheet(u"font: italic 12pt \"Verdana\";")
        self.label_13 = QLabel(self.widget_3)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(10, 580, 1231, 38))
        self.label_13.setWordWrap(True)
        self.lblImageGuideCrop = QLabel(self.widget_3)
        self.lblImageGuideCrop.setObjectName(u"lblImageGuideCrop")
        self.lblImageGuideCrop.setGeometry(QRect(730, 20, 491, 321))

        self.verticalLayout_7.addWidget(self.widget_3)

        self.tabWidget.addTab(self.tabAbout, "")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 20, 911, 41))
        self.label.setStyleSheet(u"font: 700 16pt \"Verdana\";\n"
"color: rgb(26, 51, 77);")
        self.lblStatus = QLabel(self.centralwidget)
        self.lblStatus.setObjectName(u"lblStatus")
        self.lblStatus.setGeometry(QRect(960, 15, 311, 51))
        self.lblStatus.setStyleSheet(u"font: italic 11pt \"Verdana\";\n"
"color: rgb(43, 43, 43);")
        self.lblStatus.setWordWrap(True)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.btnSetCrop.setDefault(True)
        self.btnNextImage.setDefault(True)
        self.btnRunBatch.setDefault(True)
        self.btnSelectFolder.setDefault(True)
        self.btnSaveCalibration.setDefault(True)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.lblImageDisplay.setText("")
        self.btnSetCrop.setText(QCoreApplication.translate("MainWindow", u"Set Crop", None))
        self.btnNextImage.setText(QCoreApplication.translate("MainWindow", u"Next Image", None))
        self.btnBatchCrop.setText(QCoreApplication.translate("MainWindow", u"Batch Crop", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"2. Image Display", None))
        self.lblFolderNumber.setText("")
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Folder:", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"4. Batch Processing", None))
        self.btnRunBatch.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"1. Data Initialisation", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Enter scanner & Probe name:", None))
        self.btnSelectFolder.setText(QCoreApplication.translate("MainWindow", u"Select Data Folder", None))
        self.lblFolderPath.setText("")
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Depth scale orientation:", None))
        self.checkOther.setText(QCoreApplication.translate("MainWindow", u"Other", None))
        self.checkVertical.setText(QCoreApplication.translate("MainWindow", u"Vertical", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"3. Pixel to real-life distance conversion", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Enter real distance (mm):", None))
        self.btnSaveCalibration.setText(QCoreApplication.translate("MainWindow", u"Save Calibration", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Step 3 becomes available after cropping the first image in step 2. Repeat steps 2 & 3 for all first images in the dataset.", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabProcessing), QCoreApplication.translate("MainWindow", u"Processing", None))
        self.label_58.setText(QCoreApplication.translate("MainWindow", u"Saving", None))
        self.btnGenerate.setText(QCoreApplication.translate("MainWindow", u"Generate spreadsheet", None))
        self.lblStatusInputSettings.setText(QCoreApplication.translate("MainWindow", u"Status:", None))
        self.label_42.setText(QCoreApplication.translate("MainWindow", u"FPS:", None))
        self.label_48.setText(QCoreApplication.translate("MainWindow", u"Compounding:", None))
        self.label_46.setText(QCoreApplication.translate("MainWindow", u"Dyn. range:", None))
        self.label_47.setText(QCoreApplication.translate("MainWindow", u"Application:", None))
        self.label_39.setText(QCoreApplication.translate("MainWindow", u"Other settings (grey map, tint):", None))
        self.label_52.setText(QCoreApplication.translate("MainWindow", u"Harmonic", None))
        self.label_49.setText(QCoreApplication.translate("MainWindow", u"Depth (min/preset/max):", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"Frequency:", None))
        self.label_50.setText(QCoreApplication.translate("MainWindow", u"Focus (min/preset/max):", None))
        self.label_57.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.label_45.setText(QCoreApplication.translate("MainWindow", u"Gain:", None))
        self.label_44.setText(QCoreApplication.translate("MainWindow", u"Fundamental", None))
        self.label_41.setText(QCoreApplication.translate("MainWindow", u"Smoothing / reduction:", None))
        self.label_37.setText(QCoreApplication.translate("MainWindow", u"MI & TI:", None))
        self.label_38.setText(QCoreApplication.translate("MainWindow", u"Power:", None))
        self.label_40.setText(QCoreApplication.translate("MainWindow", u"Persistence:", None))
        self.label_51.setText(QCoreApplication.translate("MainWindow", u"Frequencies range:", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"Date:", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Further details:", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Reason for testing:", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"Phantom mark:", None))
        self.label_26.setText(QCoreApplication.translate("MainWindow", u"Location:", None))
        self.label_55.setText(QCoreApplication.translate("MainWindow", u"General", None))
        self.label_53.setText(QCoreApplication.translate("MainWindow", u"Comments", None))
        self.label_54.setText(QCoreApplication.translate("MainWindow", u"Please include all comments relating to ergonomics, ease of operation and imaging.", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"Ambient  light < 50 Lux:", None))
        self.label_34.setText(QCoreApplication.translate("MainWindow", u"Probe model & s/n:", None))
        self.label_31.setText(QCoreApplication.translate("MainWindow", u"Visual checks ok (mains plug, cables):", None))
        self.label_56.setText(QCoreApplication.translate("MainWindow", u"Initial checks & specs", None))
        self.txtProbe.setText("")
        self.label_35.setText(QCoreApplication.translate("MainWindow", u"Year of manufacture:", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"Boot-up time (s):", None))
        self.label_32.setText(QCoreApplication.translate("MainWindow", u"Year of manufacture:", None))
        self.label_33.setText(QCoreApplication.translate("MainWindow", u"Software level:", None))
        self.label_30.setText(QCoreApplication.translate("MainWindow", u"At display monitor < 20 Lux: ", None))
        self.label_36.setText(QCoreApplication.translate("MainWindow", u"Scanner type & s/n:", None))
        self.btnChecks.setText(QCoreApplication.translate("MainWindow", u"Yes", None))
        self.btnAmbientLight.setText(QCoreApplication.translate("MainWindow", u"Yes", None))
        self.btnDisplayLight.setText(QCoreApplication.translate("MainWindow", u"Yes", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabInput), QCoreApplication.translate("MainWindow", u"Input settings", None))
        self.lblResults.setText("")
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Measurements", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"R", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"DR", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"LR", None))
        self.lblR.setText("")
        self.lblLR.setText("")
        self.lblDR.setText("")
        self.lblScannerNameResults.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabResults), QCoreApplication.translate("MainWindow", u"Results", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Instructions", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"1. Enter scanner & probe name into the text box, then select the folder containing the dataset. ", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"2. Using your mouse, crop the image loaded in \"2. Image display\" by drawing a square on the image such that the top of the square lines up with the top of the field-of-view and the depth scale is visible (see image on the right).", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"3. After cropping the first image, step 3 will become available. Draw a line on the displayed cropped image on the depth scale, and enter in the allocated box the corresponding real-life distance in mm. Press \"Next image\".", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"4. Once all the images have been cropped and the conversion factors entered, press the \"Run\" button in step 4. The results will be displayed in the \"Results\" tab. If wanting to add in the spreadsheet all the scanner details, do so in the \"Input settings\" tab.", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"Please ensure the data folder is organised as follows: ", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"> ScannerName\n"
"     > 8mm top.mp4\n"
"     > 8mm bot.mp4\n"
"     ...\n"
"     > 0.4mm bot.mp4", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"This application was created by Agathe Bricout as part of her PhD thesis, relying on the work by Pye & Ellis (2004) on the resolution integral. This research is supported by the Precision Medicine Doctoral Training Programme, the Medical Research Council, NHS Lothian and IMV Imaging.", None))
        self.lblImageGuideCrop.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabAbout), QCoreApplication.translate("MainWindow", u"About", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Automated Resolution Integral calculations on the Edinburgh Pipe Phantom", None))
        self.lblStatus.setText("")
    # retranslateUi

