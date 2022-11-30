from tkinter import filedialog
import tkinter as tk

def gui_dirname(**options):
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    folder_selected = filedialog.askdirectory()
    return folder_selected

def gui_fname(**options):
    print("test")
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    return filedialog.askopenfilename(**options)

def gui_fnames(**options):
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    return filedialog.askopenfilenames()

def gui_fsavename(**options):
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    return filedialog.asksaveasfilename()
    