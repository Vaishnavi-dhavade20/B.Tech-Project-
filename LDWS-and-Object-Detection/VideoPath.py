import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from PIL import Image, ImageTk
import cv2

class DrivingAssistanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driving Assistance System")
        
        self.video_path = None  # To store selected video path or webcam/ipcam option
        
        # Initialize GUI Components
        self.setup_ui()
        
    def setup_ui(self):
        # Image loading and resizing
        image1 = Image.open("assets/drivelogo.png")
        resized_image = image1.resize((170, 170), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(resized_image)
        
        # Window dimensions and positioning
        height = 380
        width = 480
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.overrideredirect(True)
        self.root.config(background="#2F6C60")

        # Labels and progress bar setup
        self.label1 = tk.Label(self.root, text="DRIVING ASSISTANCE", bg="#2F6C60", font=("Trebuchet Ms", 15, "bold"), fg="#FFFFFF")
        self.label1.place(x=140, y=55)

        self.bg_label = tk.Label(self.root, image=self.img, background="#2F6C60")
        self.bg_label.place(x=150, y=95)

        self.label2 = tk.Label(self.root, text="Loading......", bg="#2F6C60", font=("Trebuchet Ms", 15, "bold"), fg="#FFFFFF")
        self.label2.place(x=160, y=270)

        self.progress = ttk.Style()
        self.progress.theme_use("clam")
        self.progress.configure("red.Horizontal.TProgressbar", bg="#108cff")

        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='determinate', style="red.Horizontal.TProgressbar")
        self.progress.place(x=40, y=320)

        self.i = 0
        self.load_progress()

    def load_progress(self):
        if self.i <= 10:
            txt = f'Loading.... {10 * self.i}%'
            self.label2.config(text=txt)
            self.label2.after(600, self.load_progress)
            self.progress['value'] = 10 * self.i
            self.i += 1
        else:
            self.progress.destroy()
            self.show_options()

    def show_options(self):
        self.label2.config(text="Choose The Source:")
        
        file_button = tk.Button(self.root, text="Choose File", command=self.choose_file, bg="#2F6C60", fg="#FFFFFF")
        file_button.place(x=110, y=310)
        
        webcam_button = tk.Button(self.root, text="Use Webcam", command=self.choose_webcam, bg="#2F6C60", fg="#FFFFFF")
        webcam_button.place(x=210, y=310)
        
        ipcam_button = tk.Button(self.root, text="IP Camera", command=self.choose_ipcam, bg="#2F6C60", fg="#FFFFFF")
        ipcam_button.place(x=310, y=310)

    def choose_file(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if self.video_path:
            self.root.destroy()  # Close the main window

    def choose_webcam(self):
        self.video_path = 0
        self.root.destroy()  # Close the main window

    def choose_ipcam(self):
        self.video_path = simpledialog.askstring("IP Cam", "Enter the URL for the IP Camera:")
        if self.video_path:
            self.root.destroy()  # Close the main window
