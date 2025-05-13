from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QDateEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QFormLayout, QHeaderView
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QIcon
import sys, os, csv, atexit
from predictor import predict_result

class MedicalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PNDM")
        self.setFixedSize(800, 650)
        self.setWindowIcon(QIcon("logo.png")) 
        self.records = []
        self.build_ui()
        self.cargar_respaldo()

    def build_ui(self):
        layout = QVBoxLayout()

        barra_color = QLabel()
        barra_color.setFixedHeight(13)
        barra_color.setStyleSheet("background-color: #00796b;")
        layout.addWidget(barra_color)

        encabezado_widget = QWidget()
        encabezado_widget.setStyleSheet("padding: 10px; min-height: 80px;")
        top_header = QHBoxLayout(encabezado_widget)

        titulo = QLabel("Predicting Permanent Neonatal Diabetes Mellitus (PNDM)")
        titulo.setStyleSheet("font-size: 20px; font-weight: bold; color: #00796b;")
        titulo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top_header.addWidget(titulo)
        top_header.addStretch()

        logo = QLabel()
        if os.path.exists("logo.png"):
            pixmap = QPixmap("logo.png")
            logo.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_header.addWidget(logo)
        layout.addWidget(encabezado_widget)

        self.tabs = QTabWidget()
        self.tab_info = QWidget()
        self.tab_pred = QWidget()
        self.tab_reg = QWidget()

        self.tabs.addTab(self.tab_info, "Information")
        self.tabs.addTab(self.tab_pred, "Prediction")
        self.tabs.addTab(self.tab_reg, "Records")
        layout.addWidget(self.tabs)

        self.init_tab_info()
        self.init_tab_pred()
        self.init_tab_reg()

        footer = QLabel("© 2025 PNDM Software Prototype - F1 Score 89.11%")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #00796b; font-size: 11px; margin-top: 10px;")
        layout.addWidget(footer)

        self.setLayout(layout)

    def init_tab_info(self):
        layout = QVBoxLayout()
        texto = QLabel("""
         <h2>Welcome to the Permanent Neonatal Diabetes Mellitus Prediction System</h2>
        <p>
        Permanent Neonatal Diabetes Mellitus (PNDM) is a rare form of diabetes that persists throughout life. 
        It is often genetic in origin and requires timely diagnosis and appropriate treatment.
        </p>
        <p>
        This tool allows clinical staff to predict and register PNDM cases using a neural network-based model.
        </p>
        <h3>How does it work?</h3>
        <p>
        Go to the <strong>"Prediction"</strong> tab and enter the following information:
        </p>
        <ul>
        <li>Age</li>
        <li>HbA1c</li>
        <li>Genetic Info</li>
        <li>Family History</li>
        <li>Birth Weight</li>
        <li>Developmental Delay</li>
        <li>Insulin Level</li>
        </ul>
        <p>
        The system will generate an automatic prediction and save the case in the <strong>"Records"</strong> tab.
        Remember to save changes before closing the application.
        </p>
        <p><strong>This tool is intended as clinical support and does not replace medical judgment.</strong></p>
        """)
        texto.setWordWrap(True)
        texto.setAlignment(Qt.AlignTop)
        layout.addWidget(texto)
        self.tab_info.setLayout(layout)

    def init_tab_pred(self):
        form = QFormLayout()

        self.nombre = QLineEdit()
        self.nombre.setPlaceholderText("Full name")

        self.fecha = QDateEdit()
        self.fecha.setDate(QDate.currentDate())
        self.fecha.setCalendarPopup(True)

        self.age = QSpinBox()
        self.age.setRange(0, 120)
        self.age.setButtonSymbols(QDoubleSpinBox.NoButtons)

        self.hba1c = QDoubleSpinBox()
        self.hba1c.setDecimals(10)
        self.hba1c.setRange(3.0, 20.0)
        self.hba1c.setButtonSymbols(QDoubleSpinBox.NoButtons)

        self.genetic = QLineEdit()
        self.genetic.setPlaceholderText("0 = No mutation, 1 = Mutation")

        self.family = QLineEdit()
        self.family.setPlaceholderText("0 = No, 1 = Yes")

        self.birth_weight = QDoubleSpinBox()
        self.birth_weight.setRange(0.5, 6.0)
        self.birth_weight.setDecimals(10)
        self.birth_weight.setButtonSymbols(QDoubleSpinBox.NoButtons)

        self.delay = QLineEdit()
        self.delay.setPlaceholderText("0 = No, 1 = Yes")

        self.insulin = QDoubleSpinBox()
        self.insulin.setDecimals(10)
        self.insulin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.insulin.setRange(0.1, 100.0)

        self.btn = QPushButton("Ready")
        self.btn.clicked.connect(self.generar_prediccion)
        self.pred_label = QLabel("")

        form.addRow("Neonate's name:", self.nombre)
        form.addRow("Date:", self.fecha)
        form.addRow("Age (months):", self.age)
        form.addRow("HbA1c:", self.hba1c)
        form.addRow("Genetic Info:", self.genetic)
        form.addRow("Family History:", self.family)
        form.addRow("Birth Weight (kg):", self.birth_weight)
        form.addRow("Developmental Delay:", self.delay)
        form.addRow("Insulin Level (µIU/mL):", self.insulin)

        vbox = QVBoxLayout()
        vbox.addLayout(form)
        vbox.addWidget(self.btn)
        vbox.addWidget(self.pred_label)

        self.tab_pred.setLayout(vbox)

    def init_tab_reg(self):
        self.table = QTableWidget() 
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Name", "Date", "Age", "HbA1c", "GeneticInfo",
            "Family History", "BirthWeight", "DevelopmentalDelay",
            "Insulin", "Prediction"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.guardar_btn = QPushButton("Save Records")
        self.eliminar_btn = QPushButton("Delete Selected Record")
        self.eliminar_btn.clicked.connect(self.eliminar_registro)
        self.guardar_btn.clicked.connect(lambda: self.guardar_registros("registros.csv"))

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.guardar_btn)
        layout.addWidget(self.eliminar_btn)
        self.tab_reg.setLayout(layout)

    def eliminar_registro(self):
        fila_seleccionada = self.table.currentRow()
        if fila_seleccionada >= 0:
            self.table.removeRow(fila_seleccionada)
            if fila_seleccionada < len(self.records):
                del self.records[fila_seleccionada]
            print(f"Registro en fila {fila_seleccionada + 1} eliminado.")
        else:
            print("No hay fila seleccionada.")

    def generar_prediccion(self):
        try:
            datos = {
                "nombre": self.nombre.text(),
                "fecha": self.fecha.date().toString("dd/MM/yyyy"),
                "age": self.age.value(),
                "hba1c": self.hba1c.value(),
                "genetic_info": float(self.genetic.text().strip()),
                "family_history": float(self.family.text().strip()),
                "birth_weight": self.birth_weight.value(),
                "developmental_delay": float(self.delay.text().strip()),
                "insulin": self.insulin.value()
            }

            pred = predict_result(datos)
            self.pred_label.setText(f"Prediction: {pred}")

            fila = [
                datos["nombre"], datos["fecha"], str(datos["age"]),
                f"{datos['hba1c']:.2f}", str(int(datos["genetic_info"])),
                str(int(datos["family_history"])), f"{datos['birth_weight']:.2f}",
                str(int(datos["developmental_delay"])), f"{datos['insulin']:.2f}",
                pred
            ]
            self.records.append(fila)
            self.agregar_a_tabla(fila)

        except ValueError:
            self.pred_label.setText("❌ Error: All binary fields must be 0 or 1.")

    def agregar_a_tabla(self, fila):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for i, val in enumerate(fila):
            self.table.setItem(row, i, QTableWidgetItem(val))

    def guardar_registros(self, ruta="registros.csv"):
        try:
            with open(ruta, 'w', newline='', encoding='utf-8') as archivo:
                writer = csv.writer(archivo)
                writer.writerow([
                    "Name", "Date", "Age", "HbA1c", "GeneticInfo",
                    "FamilyHistory", "BirthWeight", "DevelopmentalDelay",
                    "Insulin", "Prediction"
                ])
                writer.writerows(self.records)
            print(f"Records saved to {ruta}")
        except Exception as e:
            print(f"Error saving records: {e}")

    def cargar_respaldo(self):
        respaldo = "respaldo_registros.csv"
        if os.path.exists(respaldo):
            try:
                with open(respaldo, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for fila in reader:
                        self.records.append(fila)
                        self.agregar_a_tabla(fila)
                print("📂 Backup restored successfully.")
            except Exception as e:
                print(f"⚠️ Error reading backup: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if os.path.exists("estilos.css"):
        with open("estilos.css", "r") as f:
            app.setStyleSheet(f.read())
    else:
        print("estilos.css not found.")

    ventana = MedicalApp()

    def respaldo_al_cerrar():
        if hasattr(ventana, 'records') and ventana.records:
            ventana.guardar_registros("respaldo_registros.csv")
            print("Backup saved automatically.")

    atexit.register(respaldo_al_cerrar)

    ventana.show()
    sys.exit(app.exec_())
