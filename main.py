import tkinter as tk
from gui_app import ModularEyeApp
from core.image_io import PIL_AVAILABLE

def main():
    if not PIL_AVAILABLE:
        print("Advertencia: Pillow no está instalado. Ejecuta: pip install pillow")
        
    root = tk.Tk()
    app = ModularEyeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
