# Detección de retinopatía diabética

## Descripción general

Este proyecto consiste en el desarrollo de un prototipo académico para la detección orientativa de signos asociados a retinopatía diabética mediante procesamiento digital de imágenes. El sistema trabaja principalmente con imágenes de retinografía o fondo de ojo, donde busca posibles candidatos a microaneurismas y regiones oscuras pequeñas mediante técnicas clásicas de procesamiento de imágenes.

Además del módulo principal de retinografía, el software incluye módulos complementarios para análisis ocular externo, como enrojecimiento ocular, catarata visible, pterigión o carnosidad, blefaritis y orzuelo. Estos módulos se mantienen como apoyo académico para demostrar segmentación anatómica, máscaras binarias, overlays visuales y generación de reportes.

El sistema no realiza diagnóstico médico. Los resultados son orientativos y deben interpretarse únicamente como evidencia académica del procesamiento aplicado.

## Problema que aborda

La retinopatía diabética es una alteración ocular que puede presentar signos visibles en imágenes de fondo de ojo, como microaneurismas, hemorragias, alteraciones vasculares y otras lesiones retinianas. El análisis de estas imágenes requiere identificar regiones pequeñas, diferenciar estructuras anatómicas como el disco óptico y trabajar con imágenes que pueden variar en iluminación, contraste, resolución y calidad.

Desde el enfoque del curso de Procesamiento Digital de Señales II, el problema se aborda como una tarea de procesamiento digital de imágenes: cargar una imagen, mejorar su contraste, segmentar regiones relevantes, excluir zonas que pueden generar falsos positivos y producir resultados visuales y cuantitativos.

## Solución propuesta

La solución propuesta es una aplicación de escritorio desarrollada en Python con interfaz gráfica. El sistema permite cargar imágenes oculares, ejecutar un módulo independiente de retinografía y generar salidas visuales como máscaras, overlays y archivos CSV.

El módulo de retinografía utiliza técnicas clásicas como canal verde, CLAHE, segmentación del campo de visión, exclusión del disco óptico, Black-Hat, Otsu, MSER, morfología matemática y análisis geométrico de regiones. De esta manera, el sistema busca posibles candidatos a microaneurismas sin utilizar redes neuronales entrenadas ni modelos de caja negra.

Los módulos complementarios de ojo externo utilizan selección guiada de la región ocular, marcado del iris, delimitación de abertura ocular, segmentación de esclerótica y análisis por máscaras.

## Objetivo general

Desarrollar un prototipo académico de detección orientativa de signos asociados a retinopatía diabética mediante procesamiento digital de imágenes, generando máscaras, overlays visuales y reportes cuantitativos a partir de imágenes de fondo de ojo.

## Objetivos específicos

- Implementar una interfaz gráfica para cargar y analizar imágenes oculares.
- Procesar retinografías mediante técnicas clásicas de procesamiento digital de imágenes.
- Segmentar el campo de visión retiniano y excluir el disco óptico antes de buscar candidatos.
- Detectar posibles regiones oscuras pequeñas compatibles con candidatos a microaneurismas.
- Generar máscaras binarias, overlays visuales y archivos CSV como evidencia del procesamiento.
- Mantener una arquitectura modular organizada en varios archivos reutilizables.
- Incluir módulos complementarios de análisis ocular externo para reforzar la aplicación de segmentación anatómica.

## Arquitectura del software

El proyecto está organizado de manera modular para facilitar su comprensión, mantenimiento y reutilización.

```text
Deteccion_Retinopatia_Diabetica/
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
└── notebooks/
    ├── EdgeDetection.ipynb
    ├── HoughTransform.ipynb
    └── RegionSegmentation.ipynb
```

## Responsabilidad de archivos y carpetas

| Archivo / carpeta | Responsabilidad principal |
|---|---|
| `main.py` | Punto de entrada de la aplicación. Inicializa la interfaz gráfica. |
| `gui_app.py` | Controla la interfaz, carga de imágenes, botones, flujos de análisis y visualización de resultados. |
| `config.py` | Define parámetros generales, extensiones soportadas, dimensiones de procesamiento y rutas de salida. |
| `core/` | Contiene funciones de carga, preprocesamiento, control de calidad, detección de iris/pupila y segmentación anatómica. |
| `diseases/` | Contiene los módulos de análisis: retinografía, conjuntivitis, catarata visible, pterigión, blefaritis y orzuelo. |
| `utils/` | Incluye funciones auxiliares para guardado, overlays, gráficos y análisis avanzado de máscaras. |
| `notebooks/` | Contiene cuadernos experimentales de apoyo para Canny, Hough y segmentación de regiones. No son necesarios para ejecutar la aplicación principal. |

## Funcionalidades principales

### 1. Análisis de fondo de ojo / retinografía

Es el módulo principal del proyecto. Procesa imágenes de retinografía para buscar posibles candidatos a microaneurismas o regiones oscuras pequeñas.

Etapas principales:

- Carga de imagen.
- Extracción del canal verde.
- Realce de contraste mediante CLAHE.
- Segmentación del campo de visión retiniano.
- Detección y exclusión del disco óptico.
- Aplicación de Black-Hat, Otsu y MSER.
- Filtrado geométrico de candidatos.
- Generación de overlay y CSV.

### 2. Modo guiado / corrección manual para ojo externo

Permite seleccionar manualmente la región ocular, marcar el iris y delimitar la abertura visible del ojo. Este flujo ayuda a construir máscaras anatómicas más estables para los módulos complementarios.

En este modo:

- Se selecciona la región ocular mediante recorte.
- Se marca el iris con dos clics: centro y borde.
- Se delimita la abertura ocular con un polígono de 6 a 12 clics.
- Se generan máscaras de iris, pupila, esclerótica y párpados.

### 3. Análisis automático completo

Ejecuta un flujo automático de detección y segmentación. Se mantiene como mejora adicional, aunque puede ser sensible a iluminación, reflejos, sombras o encuadres difíciles.

### 4. Módulos complementarios de ojo externo

El sistema también incluye módulos orientativos para:

- Enrojecimiento ocular / conjuntivitis.
- Catarata visible.
- Pterigión / carnosidad.
- Blefaritis.
- Orzuelo.

Estos módulos no son diagnósticos. Se usan para demostrar segmentación anatómica, análisis por color, morfología matemática y generación de resultados visuales.

## Técnicas utilizadas

- Normalización de imagen.
- Conversión de espacios de color BGR, HSV y escala de grises.
- Canal verde para retinografía.
- CLAHE para realce de contraste.
- Umbralización de Otsu.
- Detección de bordes Canny.
- Transformada de Hough.
- Morfología matemática.
- Black-Hat.
- MSER.
- Componentes conectados.
- Análisis geométrico de regiones.
- Máscaras binarias y overlays visuales.

## Dataset y datos de prueba

No se construyó un dataset clínico propio. Para las pruebas funcionales del sistema se utilizaron imágenes públicas obtenidas de internet, únicamente con fines académicos y de validación visual del prototipo.

Estas imágenes permitieron comprobar la carga de archivos, la segmentación anatómica, la generación de máscaras, overlays y reportes CSV. No se utilizaron para entrenamiento de modelos de inteligencia artificial ni para validación clínica.

Como referencia pública para el módulo de retinografía se considera mBRSET, un dataset de retinografías capturadas con cámaras portátiles y disponible en PhysioNet.

Dataset público mBRSET:
https://physionet.org/content/mbrset/1.0/

Las imágenes públicas usadas para las pruebas no se incluyen en el repositorio por consideraciones de licencia y uso responsable. El usuario puede ejecutar el sistema con imágenes propias o imágenes públicas autorizadas.

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

## Dependencias

Las dependencias principales están indicadas en `requirements.txt`:

```text
opencv-python>=4.7.0
numpy>=1.23.0
Pillow>=9.5.0
pillow_heif>=0.11.0
scikit-image
scipy
matplotlib
```

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/SoloLeveling019/Detecci-n-de-retinopat-a-diab-tica.git
```

Entrar a la carpeta del proyecto:

```bash
cd Deteccion_Retinopatia_Diabetica
```

Crear entorno virtual opcional:

```bash
python -m venv venv
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Ejecutar desde terminal:

```bash
python main.py
```

O en Windows:

```bash
run.bat
```

## Archivos generados

Los resultados se almacenan en carpetas independientes dentro de:

```text
resultados_ojo_externo/
```

Para análisis de ojo externo se pueden generar archivos como:

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

Para retinografía se pueden generar archivos como:

```text
fundus_preprocessed.png
fundus_fov_mask.png
optic_disc_mask.png
microaneurysm_candidates.png
overlay_retinopathy.png
resumen_retinopatia.csv
```

## Limitaciones

- El sistema no realiza diagnóstico médico.
- No confirma diabetes ni retinopatía diabética.
- El módulo de retinografía solo marca posibles candidatos visuales.
- La clasificación clínica completa requiere evaluación especializada.
- La calidad de imagen, iluminación, enfoque y resolución afectan los resultados.
- El modo automático puede fallar en imágenes difíciles.
- Las imágenes públicas usadas para pruebas funcionales no constituyen un dataset clínico propio.
- No se entrenaron redes neuronales ni modelos de aprendizaje profundo.

## Advertencia médica

Este software es un prototipo académico. Sus resultados son orientativos y no reemplazan una evaluación médica, oftalmológica o clínica. No debe utilizarse para tomar decisiones de salud.

## Repositorio del proyecto

El código fuente del proyecto se encuentra disponible en GitHub.

## Curso

Procesamiento Digital de Señales II

## Institución

Universidad Nacional de Piura  
Facultad de Ciencias  
Escuela Profesional de Ingeniería Electrónica y Telecomunicaciones

## Integrantes

- Saba Martínez Luiggi Smith
- Mendoza Delgado Jose Benjamin
- Escobar Gomez Eder

## Docente

MAG. ING. Yonatan Aguirre

## Estado del proyecto

Versión académica final funcional para presentación, documentación y demostración mediante grabación de pantalla.
