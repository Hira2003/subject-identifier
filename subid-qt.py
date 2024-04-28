import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QDialog,QMessageBox,QTableWidgetItem
from PyQt5.QtGui import QPixmap
from transformers import pipeline

class idn(QDialog):
  def __init__(self):
    super(idn, self).__init__()
    loadUi("subject identifier-.ui", self)
    self.scan.clicked.connect(self.process)
    self.label_2.setPixmap(QPixmap('book.png'))
    self.exit.clicked.connect(self.exiti)
  def exiti(self):
    sys.exit()
  def process(self):
    classifier = pipeline("zero-shot-classification")

    # User input for the paragraph
    user_paragraph = self.pr.toPlainText()

    # Define candidate labels representing different topics or subjects
    candidate_labels = ["educational", "sports", "technology", "science", "business", 
                    "politics", "entertainment", "health", "travel", "food"]  # Added labels

    # Classify the input paragraph into one of the candidate labels
    classification_result = classifier(user_paragraph, candidate_labels)

    # Get the predicted label with the highest score
    predicted_label = classification_result['labels'][0]

    # Print the predicted label
    self.result.setText(f"Predicted Topic:   {predicted_label}")
    
    
  
window = QApplication(sys.argv)
mainwindow = idn()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedHeight(492)
widget.setFixedWidth(757)
widget.show()
sys.exit(window.exec_())
