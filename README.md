# Analizador Ocular Modular con Procesamiento Digital de Imágenes

**Prototipo académico para el análisis orientativo de signos oculares externos y retinográficos.**

---

## Descripción general

Este proyecto consiste en el desarrollo de un prototipo académico para el análisis orientativo de imágenes oculares mediante técnicas clásicas de procesamiento digital de imágenes. El sistema permite analizar fotografías externas del ojo y retinografías, generando máscaras binarias, overlays visuales, reportes en texto y archivos CSV con resultados cuantitativos.

El software fue desarrollado en Python y cuenta con una interfaz gráfica que permite cargar imágenes, seleccionar regiones de interés y ejecutar distintos módulos de análisis ocular.

---

## Problema que aborda

El análisis visual del ojo puede proporcionar información útil sobre posibles alteraciones externas como enrojecimiento, opacidad visible, carnosidad, inflamación palpebral o lesiones en párpados. Sin embargo, las imágenes capturadas con cámaras comunes presentan variaciones importantes de iluminación, enfoque, distancia, sombras, resolución y color de piel.

Por ello, se propone un sistema asistido que permite delimitar correctamente las regiones anatómicas del ojo y aplicar técnicas clásicas de procesamiento digital de imágenes para obtener resultados interpretables.

---

## Solución propuesta

La solución propuesta es una aplicación modular en Python que integra:

- Carga de imágenes en distintos formatos.
- Preprocesamiento y normalización.
- Segmentación anatómica del ojo.
- Detección de iris, pupila, esclerótica y bandas palpebrales.
- Análisis orientativo de signos visibles en fotografías externas del ojo.
- Módulo independiente para retinografía o fondo de ojo.
- Generación de máscaras binarias, overlays visuales y archivos CSV.

El sistema no realiza diagnóstico médico. Los resultados son orientativos y deben ser interpretados únicamente con fines académicos.

---

## Objetivo general

Desarrollar un prototipo académico de análisis ocular mediante procesamiento digital de imágenes, capaz de segmentar regiones anatómicas y detectar signos visuales orientativos en fotografías externas del ojo y retinografías.

---

## Objetivos específicos

- Implementar una interfaz gráfica para la carga y análisis de imágenes oculares.
- Aplicar técnicas clásicas de procesamiento digital de imágenes para segmentar estructuras anatómicas.
- Detectar iris, pupila, esclerótica, abertura ocular y bandas palpebrales.
- Analizar signos orientativos asociados a enrojecimiento ocular, catarata visible, pterigión, blefaritis y orzuelo.
- Implementar un módulo separado para retinografía y detección de posibles candidatos a microaneurismas.
- Generar overlays visuales, máscaras binarias y archivos CSV para documentar los resultados.
- Mantener una arquitectura modular, reutilizable y comprensible.

---

## Arquitectura del software

El proyecto está organizado en varios archivos y carpetas, cada uno con una responsabilidad específica:

```text
Analizador_Ocular_Modular/
│
├── main.py
├── gui_app.py
├── config.py
├── requirements.txt
├── run.bat
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── image_io.py
│   ├── preprocess.py
│   ├── iris_pupil_detection.py
│   ├── classical_methods.py
│   ├── quality_control.py
│   └── sclera_segmentation.py
│
├── diseases/
│   ├── __init__.py
│   ├── conjunctivitis.py
│   ├── cataract.py
│   ├── pterygium.py
│   ├── blepharitis.py
│   ├── stye.py
│   └── diabetic_retinopathy_fundus.py
│
├── utils/
│   ├── __init__.py
│   ├── advanced_analysis.py
│   ├── drawing.py
│   └── helpers.py
│
├── notebooks/
│   ├── EdgeDetection.ipynb
│   ├── HoughTransform.ipynb
│   └── RegionSegmentation.ipynb
│
└── docs/
    └── capturas/
```

---

## Descripción de carpetas y archivos principales

### `main.py`

Archivo principal de ejecución. Inicializa la aplicación gráfica y carga la clase principal `ModularEyeApp`.

### `gui_app.py`

Contiene la interfaz gráfica del sistema, los botones de análisis, selección de imágenes, selección manual de regiones, ejecución del pipeline y generación de resultados.

### `config.py`

Define parámetros globales del sistema, como tamaño máximo de procesamiento, extensiones soportadas y directorio base de resultados.

### `core/`

Contiene los módulos de procesamiento base:

- `image_io.py`: carga imágenes en distintos formatos y soporta rutas con espacios o tildes.
- `preprocess.py`: normalización, conversión de canales y redimensionamiento.
- `classical_methods.py`: métodos clásicos como Canny, Otsu, Hough, MSER y limpieza de máscaras.
- `iris_pupil_detection.py`: detección aproximada de pupila e iris.
- `quality_control.py`: validaciones de calidad de imagen.
- `sclera_segmentation.py`: segmentación anatómica de esclerótica, iris, pupila, abertura ocular y bandas palpebrales.

### `diseases/`

Contiene los módulos de análisis orientativo:

- `conjunctivitis.py`: analiza porcentaje y distribución de enrojecimiento.
- `cataract.py`: analiza opacidad visible en la zona pupilar.
- `pterygium.py`: analiza posibles regiones compatibles con pterigión o carnosidad.
- `blepharitis.py`: analiza signos visibles en bandas palpebrales.
- `stye.py`: analiza posibles lesiones compatibles con orzuelo.
- `diabetic_retinopathy_fundus.py`: analiza retinografías y posibles candidatos a microaneurismas.

### `utils/`

Contiene funciones auxiliares:

- `advanced_analysis.py`: relleno de huecos, filtrado geométrico y gráficos.
- `drawing.py`: generación de overlays visuales.
- `helpers.py`: guardado robusto de imágenes.

### `notebooks/`

Contiene notebooks de apoyo utilizados durante el desarrollo del proyecto. Documentan pruebas experimentales relacionadas con técnicas clásicas de procesamiento digital de imágenes:

- Detección de bordes.
- Transformada de Hough.
- Segmentación de regiones.

Estos notebooks son complementarios y no son necesarios para ejecutar la aplicación principal.

---

## Funcionalidades principales

### 1. Modo guiado / corrección manual

Es el modo recomendado para la entrega final. Permite seleccionar manualmente la región ocular, marcar el iris y delimitar la abertura visible del ojo. Esto mejora la estabilidad anatómica de las máscaras.

### 2. Análisis automático completo

Ejecuta una detección automática de la región ocular y aplica los módulos de análisis. Se considera una mejora adicional, aunque puede ser sensible a iluminación, enfoque, sombras o encuadre.

### 3. Análisis de fondo de ojo / retinografía

Módulo independiente para analizar retinografías. Busca posibles candidatos a microaneurismas mediante procesamiento del canal verde, segmentación del campo de visión, exclusión del disco óptico, Black-Hat, Otsu, MSER y análisis de regiones.

---

## Señales analizadas

El sistema analiza signos visuales orientativos asociados a:

- Conjuntivitis / enrojecimiento ocular.
- Catarata visible.
- Pterigión / carnosidad.
- Blefaritis.
- Orzuelo.
- Posibles candidatos a microaneurismas en retinografía.

---

## Técnicas utilizadas

El sistema utiliza técnicas clásicas de procesamiento digital de imágenes, entre ellas:

- Normalización de imagen.
- Conversión de espacios de color.
- Segmentación en HSV.
- Umbralización de Otsu.
- Detección de bordes Canny.
- Transformada de Hough.
- Morfología matemática.
- Componentes conectados.
- Análisis geométrico de regiones.
- CLAHE.
- Black-Hat.
- MSER.
- Generación de máscaras binarias.
- Overlays visuales.

---

## Dataset

No se construyó un dataset clínico propio. Para las pruebas funcionales se utilizaron imágenes locales de validación.

Como referencia académica para el módulo de retinografía se considera el dataset público **mBRSET**, publicado en *Scientific Data*. Este dataset contiene 5,164 imágenes de fondo de ojo de 1,291 pacientes capturadas con cámaras retinianas portátiles en escenarios reales.

Dataset público mBRSET:

```text
https://www.physionet.org/content/mbrset/1.0/
```

Repositorio asociado:

```text
https://github.com/luisnakayama/mBRSET
```

---

## Requisitos del sistema

Sistema operativo usado durante el desarrollo:

```text
Windows 10 / Windows 11
```

Lenguaje de programación:

```text
Python 3.x
```

Entorno de trabajo:

```text
Visual Studio Code
Terminal de Windows / PowerShell
```

---

## Dependencias

Las dependencias principales se encuentran en `requirements.txt`:

```text
opencv-python>=4.7.0
numpy>=1.23.0
Pillow>=9.5.0
pillow_heif>=0.11.0
scikit-image
scipy
matplotlib
```

---

## Instalación

Clonar el repositorio:

```bash
git clone URL_DEL_REPOSITORIO
cd Analizador_Ocular_Modular
```

Crear un entorno virtual opcional:

```bash
python -m venv venv
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución

Ejecutar desde terminal:

```bash
python main.py
```

O ejecutar en Windows:

```bash
run.bat
```

---

## Uso básico

1. Abrir el programa con `python main.py` o `run.bat`.
2. Presionar **Seleccionar Imagen**.
3. Cargar una fotografía del ojo o una retinografía.
4. Para fotografía externa, usar preferentemente **Modo guiado / corrección manual**.
5. Seleccionar el recorte del ojo.
6. Marcar el centro y borde del iris.
7. Marcar la abertura ocular visible.
8. Revisar el overlay generado.
9. Abrir la carpeta de resultados para revisar máscaras y CSV.

---

## Archivos generados

Por cada análisis, el sistema genera una carpeta dentro de:

```text
resultados_ojo_externo/
```

Resultados principales para ojo externo:

```text
normalized_eye_crop.png
eye_opening_mask.png
iris_pupil_mask.png
sclera_roi_mask.png
white_sclera_mask.png
red_mask.png
overlay_general_limpio.png
resumen_general.csv
```

Resultados principales para retinografía:

```text
fundus_preprocessed.png
fundus_fov_mask.png
optic_disc_mask.png
microaneurysm_candidates.png
overlay_retinopathy.png
resumen_retinopatia.csv
```

---

## Pruebas recomendadas

Para validar el sistema se recomienda probar como mínimo:

1. Ojo normal.
2. Ojo rojo.
3. Ojo con párpados visibles.
4. Posible pterigión o carnosidad.
5. Retinografía.

Durante la validación visual se debe verificar:

- Que la máscara de esclerótica no invada piel, cejas o fondo.
- Que la máscara roja esté dentro de la esclerótica.
- Que el iris y la pupila estén correctamente delimitados.
- Que las bandas palpebrales se ubiquen cerca de los párpados.
- Que el overlay general sea visualmente entendible.
- Que el CSV se genere correctamente.

---

## Limitaciones

- El sistema depende de la calidad de la imagen.
- La iluminación, sombras, desenfoque o mala selección de ROI pueden afectar los resultados.
- El modo automático puede fallar en imágenes difíciles.
- El módulo de retinografía no clasifica clínicamente la retinopatía diabética completa.
- El sistema no detecta glaucoma.
- El sistema no confirma diabetes.
- Los resultados son orientativos y no representan diagnóstico médico.
- No reemplaza una evaluación oftalmológica.

---

## Advertencia médica

Este software es un prototipo académico. No confirma enfermedades, no diagnostica diabetes, retinopatía diabética, glaucoma ni otras patologías. Los resultados deben interpretarse únicamente como apoyo visual orientativo.

---

## Estado del proyecto

Versión académica final funcional, pendiente únicamente de validación visual final con imágenes de prueba antes de la exposición.

---

## Curso

Procesamiento Digital de Señales II

---

## Licencia

Proyecto desarrollado con fines académicos.
