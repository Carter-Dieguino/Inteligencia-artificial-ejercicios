import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class ClasificadorMovimiento:
    def __init__(self):
        self.svm_model = None
        self.nb_model = None
        self.scaler = StandardScaler()
        self.situaciones = ['Acostado', 'Caminando', 'Corriendo', 'Sentado']
        
    def cargar_datos_entrenamiento(self, ruta_excel, situacion_forzada=None):
            try:
                # Leer el archivo Excel
                df = pd.read_excel(ruta_excel, na_values=[''], keep_default_na=True)
                
                # Usar la situación proporcionada o intentar leerla del archivo
                if situacion_forzada is not None:
                    situacion = situacion_forzada
                else:
                    try:
                        situacion = pd.read_excel(ruta_excel, usecols="F", nrows=1).iloc[0, 0]
                        situacion = str(situacion).strip()
                        if pd.isna(situacion) or situacion not in self.situaciones:
                            situacion = self.solicitar_situacion()
                    except:
                        situacion = self.solicitar_situacion()
                
                # Convertir las columnas de datos a numérico, empezando desde la fila 2
                columnas = ['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 
                           'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']
                
                # Obtener solo los datos numéricos (desde la fila 2)
                X = df[columnas].iloc[1:].copy()
                
                # Convertir a numérico
                for col in columnas:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                
                # Eliminar cualquier fila con valores NaN después de la conversión
                X = X.dropna()
                
                # Crear el vector de etiquetas del mismo tamaño que X
                y = pd.Series([situacion] * len(X))
                
                # Verificaciones finales
                if len(X) == 0:
                    raise Exception("No hay datos válidos después del procesamiento")
                
                print(f"Datos cargados exitosamente:")
                print(f"- Situación: {situacion}")
                print(f"- Número de registros: {len(X)}")
                return X, y
                
            except Exception as e:
                raise Exception(f"Error al cargar los datos de entrenamiento: {str(e)}")
        
    def solicitar_situacion(self):
            """
            Crea una ventana para que el usuario seleccione la situación
            """
            seleccion = {'situacion': None}
            
            def on_seleccionar():
                seleccion['situacion'] = combo.get()
                ventana.destroy()
            
            # Crear ventana
            ventana = tk.Toplevel()
            ventana.title("Seleccionar Situación")
            ventana.geometry("300x150")
            
            # Hacer la ventana modal
            ventana.transient(ventana.master)
            ventana.grab_set()
            
            # Añadir explicación
            ttk.Label(ventana, 
                     text="No se pudo detectar la situación en el archivo.\nPor favor, seleccione la situación correcta:",
                     wraplength=250,
                     justify="center").pack(pady=10)
            
            # Combo box para seleccionar situación
            combo = ttk.Combobox(ventana, values=self.situaciones, state="readonly")
            combo.pack(pady=10)
            combo.set(self.situaciones[0])  # Valor por defecto
            
            # Botón de confirmación
            ttk.Button(ventana, text="Confirmar", command=on_seleccionar).pack(pady=10)
            
            # Centrar la ventana
            ventana.update_idletasks()
            width = ventana.winfo_width()
            height = ventana.winfo_height()
            x = (ventana.winfo_screenwidth() // 2) - (width // 2)
            y = (ventana.winfo_screenheight() // 2) - (height // 2)
            ventana.geometry(f'{width}x{height}+{x}+{y}')
            
            # Esperar hasta que se cierre la ventana
            ventana.wait_window()
            
            if seleccion['situacion'] is None:
                raise Exception("No se seleccionó ninguna situación")
                
            return seleccion['situacion']        
      
    def cargar_datos_prediccion(self, ruta_excel):
        try:
            df = pd.read_excel(ruta_excel)
            return df[['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 
                      'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']]
        except Exception as e:
            raise Exception(f"Error al cargar los datos para predicción: {e}")
    
    def entrenar_modelos(self, X, y, test_size=0.2):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.svm_model = SVC(kernel='rbf', probability=True)
            self.svm_model.fit(X_train_scaled, y_train)
            
            self.nb_model = GaussianNB()
            self.nb_model.fit(X_train_scaled, y_train)
            
            svm_accuracy = self.svm_model.score(X_test_scaled, y_test)
            nb_accuracy = self.nb_model.score(X_test_scaled, y_test)
            
            return svm_accuracy, nb_accuracy
        except Exception as e:
            raise Exception(f"Error al entrenar los modelos: {e}")
    
    def predecir(self, datos, modelo='svm'):
        try:
            datos_scaled = self.scaler.transform(datos)
            
            if modelo.lower() == 'svm':
                if self.svm_model is None:
                    raise Exception("El modelo SVM no ha sido entrenado")
                predicciones = self.svm_model.predict(datos_scaled)
                probabilidades = self.svm_model.predict_proba(datos_scaled)
            else:
                if self.nb_model is None:
                    raise Exception("El modelo Naive Bayes no ha sido entrenado")
                predicciones = self.nb_model.predict(datos_scaled)
                probabilidades = self.nb_model.predict_proba(datos_scaled)
            
            return predicciones, probabilidades
        except Exception as e:
            raise Exception(f"Error al realizar la predicción: {e}")
    
    def guardar_modelos(self, ruta_base):
        try:
            joblib.dump(self.svm_model, f"{ruta_base}_svm.joblib")
            joblib.dump(self.nb_model, f"{ruta_base}_nb.joblib")
            joblib.dump(self.scaler, f"{ruta_base}_scaler.joblib")
        except Exception as e:
            raise Exception(f"Error al guardar los modelos: {e}")
    
    def cargar_modelos(self, ruta_base):
        try:
            self.svm_model = joblib.load(f"{ruta_base}_svm.joblib")
            self.nb_model = joblib.load(f"{ruta_base}_nb.joblib")
            self.scaler = joblib.load(f"{ruta_base}_scaler.joblib")
        except Exception as e:
            raise Exception(f"Error al cargar los modelos: {e}")

class AplicacionClasificador:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Movimiento")
        self.root.geometry("800x600")
        
        self.clasificador = ClasificadorMovimiento()
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Pestaña de entrenamiento
        self.tab_entrenamiento = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text='Entrenamiento')
        self.crear_tab_entrenamiento()
        
        # Pestaña de predicción
        self.tab_prediccion = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_prediccion, text='Predicción')
        self.crear_tab_prediccion()
    
    def crear_tab_entrenamiento(self):
        # Frame para cargar datos
        frame_datos = ttk.LabelFrame(self.tab_entrenamiento, text="Datos de Entrenamiento", padding=10)
        frame_datos.pack(fill='x', padx=10, pady=5)
        
        # Crear frame para cada situación
        self.archivos_entrenamiento = {}
        for situacion in self.clasificador.situaciones:
            frame_situacion = ttk.Frame(frame_datos)
            frame_situacion.pack(fill='x', padx=5, pady=2)
            
            lbl_situacion = ttk.Label(frame_situacion, text=f"{situacion}:", width=15)
            lbl_situacion.pack(side='left', padx=5)
            
            lbl_archivo = ttk.Label(frame_situacion, text="No seleccionado")
            lbl_archivo.pack(side='left', padx=5, fill='x', expand=True)
            
            btn_cargar = ttk.Button(frame_situacion, text="Seleccionar Excel",
                                  command=lambda s=situacion, l=lbl_archivo: self.cargar_excel_situacion(s, l))
            btn_cargar.pack(side='right', padx=5)
            
            self.archivos_entrenamiento[situacion] = {'label': lbl_archivo, 'ruta': None}
        
        # Frame para entrenamiento
        frame_entrenamiento = ttk.LabelFrame(self.tab_entrenamiento, text="Entrenamiento", padding=10)
        frame_entrenamiento.pack(fill='x', padx=10, pady=5)
        
        btn_entrenar = ttk.Button(frame_entrenamiento, text="Entrenar Modelos", 
                                 command=self.entrenar)
        btn_entrenar.pack(pady=5)
        
        self.lbl_resultados = ttk.Label(frame_entrenamiento, text="")
        self.lbl_resultados.pack(pady=5)
        
        # Frame para guardar modelos
        frame_guardar = ttk.LabelFrame(self.tab_entrenamiento, text="Guardar Modelos", padding=10)
        frame_guardar.pack(fill='x', padx=10, pady=5)
        
        btn_guardar = ttk.Button(frame_guardar, text="Guardar Modelos Entrenados", 
                                command=self.guardar_modelos)
        btn_guardar.pack(pady=5)

    def cargar_excel_situacion(self, situacion, label):
        ruta = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if ruta:
            self.archivos_entrenamiento[situacion]['ruta'] = ruta
            label.config(text=os.path.basename(ruta))

    def entrenar(self):
        try:
            # Verificar que se hayan cargado archivos para al menos dos situaciones
            archivos_cargados = sum(1 for info in self.archivos_entrenamiento.values() if info['ruta'] is not None)
            if archivos_cargados < 2:
                messagebox.showerror("Error", "Debe cargar datos de al menos dos situaciones diferentes")
                return
            
            # Cargar y combinar todos los datos
            X_combined = []
            y_combined = []
            
            for situacion, info in self.archivos_entrenamiento.items():
                if info['ruta'] is not None:
                    try:
                        X, y = self.clasificador.cargar_datos_entrenamiento(info['ruta'], situacion)
                        X_combined.append(X)
                        y_combined.extend(y)
                    except Exception as e:
                        messagebox.showerror("Error", f"Error al cargar datos de {situacion}: {str(e)}")
                        return
            
            # Combinar todos los datos
            X_final = pd.concat(X_combined, ignore_index=True)
            y_final = pd.Series(y_combined)
            
            # Entrenar los modelos
            svm_acc, nb_acc = self.clasificador.entrenar_modelos(X_final, y_final)
            
            resultado = f"Precisión Máquinas de Soporte Vectorial: {svm_acc:.2%}\n"
            resultado += f"Precisión Clasificador Ingenuo Naive Bayes: {nb_acc:.2%}"
            self.lbl_resultados.config(text=resultado)
            messagebox.showinfo("Éxito", "Modelos entrenados correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def crear_tab_prediccion(self):
        # Frame para cargar modelos
        frame_cargar = ttk.LabelFrame(self.tab_prediccion, text="Cargar Modelos", padding=10)
        frame_cargar.pack(fill='x', padx=10, pady=5)
        
        btn_cargar_modelos = ttk.Button(frame_cargar, text="Cargar Modelos Guardados", 
                                       command=self.cargar_modelos)
        btn_cargar_modelos.pack(pady=5)
        
        # Frame para datos de predicción
        frame_datos_pred = ttk.LabelFrame(self.tab_prediccion, text="Datos para Predicción", padding=10)
        frame_datos_pred.pack(fill='x', padx=10, pady=5)
        
        self.lbl_archivo_prediccion = ttk.Label(frame_datos_pred, text="Ningún archivo seleccionado")
        self.lbl_archivo_prediccion.pack(side='left', padx=5)
        
        btn_cargar_pred = ttk.Button(frame_datos_pred, text="Seleccionar Excel para Predicción", 
                                    command=self.cargar_excel_prediccion)
        btn_cargar_pred.pack(side='right', padx=5)
        
        # Frame para predicción
        frame_prediccion = ttk.LabelFrame(self.tab_prediccion, text="Realizar Predicción", padding=10)
        frame_prediccion.pack(fill='x', padx=10, pady=5)
        
        frame_botones = ttk.Frame(frame_prediccion)
        frame_botones.pack(pady=5)
        
        btn_svm = ttk.Button(frame_botones, text="Predecir con Máquinas de Soporte Vectorial", 
                            command=lambda: self.predecir('svm'))
        btn_svm.pack(side='left', padx=5)
        
        btn_nb = ttk.Button(frame_botones, text="Predecir con Clasificador Ingenuo Naive Bayes", 
                           command=lambda: self.predecir('nb'))
        btn_nb.pack(side='left', padx=5)
        
        # Frame para resultados
        self.frame_resultados = ttk.LabelFrame(self.tab_prediccion, text="Resultados", padding=10)
        self.frame_resultados.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Crear Treeview para mostrar resultados
        self.tree = ttk.Treeview(self.frame_resultados, columns=('timestamp', 'prediccion', 'probabilidad'), 
                                show='headings')
        self.tree.heading('timestamp', text='Tiempo (s)')
        self.tree.heading('prediccion', text='Situación Predicha')
        self.tree.heading('probabilidad', text='Probabilidad')
        self.tree.pack(fill='both', expand=True)
        
        # Scrollbar para el Treeview
        scrollbar = ttk.Scrollbar(self.frame_resultados, orient='vertical', command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar.set)
    
    def cargar_excel_prediccion(self):
        ruta = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if ruta:
            self.ruta_excel_prediccion = ruta
            self.lbl_archivo_prediccion.config(text=os.path.basename(ruta))
      
    def guardar_modelos(self):
        try:
            if self.clasificador.svm_model is None:
                messagebox.showerror("Error", "Primero debe entrenar los modelos")
                return
                
            ruta = filedialog.asksaveasfilename(defaultextension=".joblib")
            if ruta:
                self.clasificador.guardar_modelos(ruta[:-7])  # Quitar la extensión .joblib
                messagebox.showinfo("Éxito", "Modelos guardados correctamente")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def cargar_modelos(self):
        try:
            ruta = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib")])
            if ruta:
                self.clasificador.cargar_modelos(ruta[:-11])  # Quitar _svm.joblib
                messagebox.showinfo("Éxito", "Modelos cargados correctamente")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def predecir(self, modelo):
        try:
            # Verificar que los modelos estén cargados
            if self.clasificador.svm_model is None:
                messagebox.showerror("Error", "Primero debe cargar los modelos")
                return
            
            # Verificar que haya datos para predecir
            if not hasattr(self, 'ruta_excel_prediccion'):
                messagebox.showerror("Error", "Por favor, seleccione un archivo Excel para predicción")
                return
            
            # Cargar datos para predicción
            datos = self.clasificador.cargar_datos_prediccion(self.ruta_excel_prediccion)
            
            # Realizar predicción
            predicciones, probabilidades = self.clasificador.predecir(datos, modelo)
            
            # Limpiar tabla anterior
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Mostrar resultados en la tabla
            for i, (pred, probs) in enumerate(zip(predicciones, probabilidades)):
                tiempo = datos.iloc[i]['Time (s)']
                prob_max = max(probs)
                self.tree.insert('', 'end', values=(f"{tiempo:.2f}", pred, f"{prob_max:.2%}"))         
            messagebox.showinfo("Éxito", f"Predicciones realizadas con {modelo.upper()}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionClasificador(root)
    root.mainloop()
