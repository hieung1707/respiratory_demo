from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter.messagebox as messagebox
from visualization.static_plot import StaticPlot
from visualization.predict_plot import PredictPlot
import threading
from utils import UDP_CLIENT_IP, UDP_CLIENT_PORT


class MainMenuInterface():
    def __init__(self):
        self.exit_flag = False
        self.is_handshaking = True

        self.window = Tk()
        self.window.title('Main menu')
        self.btn_analyze = Button(self.window, text="Analyze File", font=("Arial Bold", 24), height=1, width=15)
        # self.btn_realtime = Button(self.window, text="Realtime", font=("Arial Bold", 24), height=1, width=15)
        self.btn_exit = Button(self.window, text="Exit", font=("Arial Bold", 24), height=1, width=15)
        self.create_interface()
        self.setup_listeners()

    def create_interface(self):
        self.btn_analyze.grid(column=0, row=0, padx=20, pady=20)
        # self.btn_realtime.grid(column=0, row=1, padx=20, pady=20)
        self.btn_exit.grid(column=0, row=2, padx=20, pady=20)
        w = self.window.winfo_reqwidth()
        h = self.window.winfo_reqheight()
        pos_right = int(self.window.winfo_screenwidth() / 2 - w / 2)
        pos_down = int(self.window.winfo_screenheight() / 2 - h / 2)
        self.window.geometry("+{}+{}".format(pos_right, pos_down))
        self.window.resizable(False, False)

    def setup_listeners(self):
        self.btn_analyze.config(command=lambda :self.btn_click(mode=0))
        # self.btn_realtime.config(command=lambda :self.btn_click(mode=1))
        self.btn_exit.config(command=lambda :self.btn_click(mode=2))

    def btn_click(self, mode):
        if mode == 0:
            self.analyze_click()
        elif mode == 1:
            self.realtime_click()
        elif mode == 2:
            self.exit_click()

    def analyze_click(self):
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename(initialdir='/home/hieung1707/projects/respiratory_demo/data/test_files')  # show an "Open" dialog box and return the path to the selected file
        if filename == ():
            return
        if not filename.endswith('.wav'):
            messagebox.showerror('Extension error', 'Only .wav extension allowed')
        self.window.destroy()
        static_plot = StaticPlot()
        static_plot.analyze_and_visualize(filename)
        mm = MainMenuInterface()
        mm.start()

    def realtime_click(self):
        self.window.destroy()
        predict_plot = PredictPlot(sock=None, host=UDP_CLIENT_IP, port=UDP_CLIENT_PORT)
        t = threading.Thread(target=predict_plot.read_data)
        t.start()
        predict_plot.start_plotting()
        mm = MainMenuInterface()
        mm.start()

    def exit_click(self):
        self.exit_flag = True
        self.window.destroy()

    def start(self):
        self.window.mainloop()


if __name__ == "__main__":
    main_menu = MainMenuInterface()
    main_menu.start()