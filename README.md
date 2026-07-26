# Analizador Ocular Modular con Procesamiento Digital de Imágenes

## Descripción general

Este proyecto consiste en el desarrollo de un prototipo académico para el análisis orientativo de imágenes oculares mediante técnicas clásicas de procesamiento digital de imágenes. El sistema permite analizar fotografías externas del ojo y retinografías, generando máscaras, overlays visuales, reportes en texto y archivos CSV con resultados cuantitativos.

El software fue desarrollado en Python y cuenta con una interfaz gráfica que permite cargar imágenes, seleccionar regiones de interés y ejecutar distintos módulos de análisis ocular.

## Problema que aborda

El análisis visual del ojo puede proporcionar información útil sobre posibles alteraciones externas como enrojecimiento, opacidad visible, carnosidad, inflamación palpebral o lesiones en párpados. Sin embargo, las imágenes capturadas con cámaras comunes presentan variaciones de iluminación, enfoque, distancia, sombras y color de piel.

Por ello, se propone un sistema asistido que permita delimitar correctamente las regiones anatómicas del ojo y aplicar técnicas clásicas de procesamiento digital de imágenes para obtener resultados interpretables.

## Solución propuesta

La solución propuesta es una aplicación modular en Python que integra:

- Carga de imágenes en distintos formatos.
- Preprocesamiento y normalización.
- Segmentación anatómica del ojo.
- Detección de iris, pupila, esclerótica y bandas palpebrales.
- Análisis orientativo de signos visibles.
- Módulo independiente para retinografía.
- Generación de máscaras, overlays y archivos CSV.

El sistema no realiza diagnóstico médico. Los resultados son orientativos y deben ser interpretados únicamente con fines académicos.

## Objetivo general

Desarrollar un prototipo académico de análisis ocular mediante procesamiento digital de imágenes, capaz de segmentar regiones anatómicas y detectar signos visuales orientativos en fotografías oculares externas y retinografías.

## Objetivos específicos

- Implementar una interfaz gráfica para cargar y analizar imágenes oculares.
- Aplicar técnicas clásicas como Otsu, Canny, Hough, morfología matemática y componentes conectados.
- Segmentar iris, pupila, esclerótica y bandas palpebrales.
- Analizar signos orientativos asociados a conjuntivitis, catarata visible, pterigión, blefaritis y orzuelo.
- Implementar un módulo separado para retinografía y posibles candidatos a microaneurismas.
- Generar resultados visuales mediante overlays y resultados cuantitativos mediante CSV.
- Mantener una arquitectura modular y reutilizable.

## Arquitectura del software

El proyecto se organiza de forma modular:

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
│   ├── image_io.py
│   ├── preprocess.py
│   ├── iris_pupil_detection.py
│   ├── classical_methods.py
│   ├── quality_control.py
│   └── sclera_segmentation.py
│
├── diseases/
│   ├── conjunctivitis.py
│   ├── cataract.py
│   ├── pterygium.py
│   ├── blepharitis.py
│   ├── stye.py
│   └── diabetic_retinopathy_fundus.py
│
├── utils/
│   ├── advanced_analysis.py
│   ├── drawing.py
│   └── helpers.py
│
└── notebooks/
    ├── EdgeDetection.ipynb
    ├── HoughTransform.ipynb
    └── RegionSegmentation.ipynb
