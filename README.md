# Detección de Retinopatía Diabética - Avance 50%

## Descripción general

Este repositorio contiene el avance del proyecto de Procesamiento Digital de Señales / Imágenes orientado al análisis ocular mediante Python y OpenCV.

El objetivo principal del proyecto es desarrollar un sistema de apoyo para analizar imágenes de fondo de ojo y detectar candidatos a microaneurismas asociados a retinopatía diabética. Como módulo complementario de avance, también se implementó una interfaz para analizar fotografías externas del ojo y cuantificar enrojecimiento visible en la esclerótica.

> Advertencia: el sistema no diagnostica diabetes ni reemplaza evaluación médica. La diabetes se confirma mediante pruebas clínicas de glucosa/HbA1c y la retinopatía diabética se evalúa correctamente con imágenes de fondo de ojo o retinografía.

## Integrantes

- José Benjamín Mendoza Delgado
- [Agregar nombre de integrante]
- [Agregar nombre de integrante]

## Estado del avance

Avance estimado: 50%.

### Implementado

- Carga robusta de imágenes en formatos comunes: JPG, PNG, BMP, TIFF, WEBP, JP2 y HEIC/HEIF si se instala `pillow-heif`.
- Interfaz gráfica en Tkinter para seleccionar la imagen desde el explorador.
- Preprocesamiento con OpenCV y NumPy.
- Segmentación aproximada de esclerótica en foto externa del ojo.
- Detección de zonas rojizas o venas visibles mediante espacio HSV y dominancia del canal rojo.
- Cálculo de índice de enrojecimiento.
- Exportación de resultados visuales y CSV.
- Módulo base para análisis de retinografías con canal verde, CLAHE, operaciones morfológicas, umbralización y filtrado por área/circularidad.

### Pendiente

- Validar con un conjunto de retinografías reales.
- Separar claramente el módulo de retinografía y el módulo de ojo externo.
- Añadir métricas de validación: sensibilidad, precisión, falsos positivos y comparación con anotaciones.
- Mejorar la exclusión de disco óptico y vasos sanguíneos en retinografía.
- Documentar resultados finales y capturas del sistema.

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── analizador_ojo_externo_interfaz.py
│   ├── detector_microaneurismas_interfaz.py
│   └── detector_microaneurismas_cli.py
├── docs/
│   └── informe_avance_50.pdf
├── data/
│   └── ejemplos/
└── resultados/
```

## Instalación

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

En Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

### Interfaz de ojo externo

```bash
python src/analizador_ojo_externo_interfaz.py
```

### Detector de microaneurismas en retinografía

```bash
python src/detector_microaneurismas_interfaz.py
```

## Requisitos de entrada

Para el análisis de retinopatía diabética, la entrada correcta debe ser una imagen de fondo de ojo o retinografía. Una foto externa del ojo no muestra la retina y, por tanto, no permite evaluar microaneurismas retinales.

## Enlace al informe

El informe de avance se encuentra en:

```text
docs/informe_avance_50.pdf
```

## Licencia

Proyecto académico para fines educativos.
